"""Faster Qwen-Image-2.1 inference on Apple silicon.

diffusers' transformer re-derives the sequence layout on every denoising step and, in each of its 32 blocks, spends
more time in the elementwise kernels around the matmuls than the matmuls' size would suggest. `apply` gives the
transformer a forward that computes the layout once per run, keeps it on the KV cache the pipeline already hands to
every step, and folds the modulation and norms into fewer kernels. The arithmetic is the same; only the rounding
points move, since the fused kernels round once where the originals rounded twice.
"""

import math

import torch
import torch.nn.functional as F
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers import transformer_qwenimage21 as reference


class RunLayout:
    """What the transformer derives from the prompt and the image sizes. It is the same for every step of a run,
    and every piece of it costs a device sync to build."""

    def __init__(self, model, img_shapes, img_mask, encoder_hidden_states_mask):
        batch_size, device = img_mask.shape[0], img_mask.device
        # Each vision-language image slot stands for 2x2 latent tokens.
        repeats = torch.where(img_mask, reference._IMG_TOKENS_PER_SLOT, 1)[0]
        self.repeats = repeats
        self.image_pad_mask = torch.repeat_interleave(img_mask[0], repeats)
        self.rotary_emb = model.pos_embed(img_shapes, self.image_pad_mask, device=device).unsqueeze(1)
        image_ids, target_token_mask = model.build_token_metadata(self.image_pad_mask, img_shapes)
        self.target_token_mask = target_token_mask
        self.prefix_len = int((~target_token_mask).sum())
        self.segments = reference._qwenimage21_prefix_segments(image_ids, self.prefix_len)
        self.target_tokens = math.prod(img_shapes[-1])

        # Right-padded prompt positions must never be attended to. Text positions of the joint sequence line up, in
        # order, with the non-image positions of the vision-language sequence. A mask that excludes nothing is
        # dropped: attention runs faster without one.
        self.key_valid = None
        if encoder_hidden_states_mask is not None and not encoder_hidden_states_mask.all():
            key_valid = torch.ones(batch_size, self.image_pad_mask.shape[0], dtype=torch.bool, device=device)
            text_positions = (~self.image_pad_mask).nonzero(as_tuple=True)[0]
            vlm_text_positions = ~img_mask[0][: encoder_hidden_states_mask.shape[1]]
            key_valid[:, text_positions] = encoder_hidden_states_mask.bool()[:, vlm_text_positions]
            self.key_valid = key_valid


class Modulation:
    """The step's shared modulation, arranged for the rows a block will see.

    `params` holds `batch_size + 1` rows: the samples' timestep rows and a trailing `t = 0` row that text and
    condition-image tokens read. Scales stay one row each and are applied to the prefix and target slices
    separately; gates are expanded to the sequence so the residual update is a single `addcmul`.
    """

    def __init__(self, params, prefix_len, target_token_mask):
        scale1, gate1, scale2, gate2 = params.chunk(4, dim=-1)
        self.prefix_len = prefix_len
        target, prefix = slice(None, -1), slice(-1, None)
        self.scale1 = (scale1[prefix][None], scale1[target].unsqueeze(1))
        self.scale2 = (scale2[prefix][None], scale2[target].unsqueeze(1))
        self.gate1 = self._rows(gate1.tanh(), target_token_mask)
        self.gate2 = self._rows(gate2.tanh(), target_token_mask)

    @staticmethod
    def _rows(params, target_token_mask):
        if target_token_mask is None:
            return params[:-1].unsqueeze(1)
        return torch.where(target_token_mask.view(1, -1, 1), params[:-1].unsqueeze(1), params[-1:].unsqueeze(0))

    def norm(self, hidden_states, scales, eps):
        prefix_scale, target_scale = scales
        if self.prefix_len == 0:
            return scaled_layer_norm(hidden_states, target_scale, eps)
        return torch.cat(
            [
                scaled_layer_norm(hidden_states[:, : self.prefix_len], prefix_scale, eps),
                scaled_layer_norm(hidden_states[:, self.prefix_len :], target_scale, eps),
            ],
            dim=1,
        )


def scaled_layer_norm(hidden_states, scale, eps):
    """`LayerNorm(x) * (1 + scale)`. One kernel when the scale is a single row, which it is at batch size 1."""
    if scale.shape[0] == 1:
        return F.layer_norm(hidden_states, hidden_states.shape[-1:], weight=(1 + scale).reshape(-1), eps=eps)
    return F.layer_norm(hidden_states, hidden_states.shape[-1:], eps=eps) * (1 + scale)


def rotate(x, freqs):
    x = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(x * freqs).flatten(3)


def attend(query, key, value, mask=None):
    """SDPA over `(batch, tokens, heads, head_dim)` tensors."""
    out = F.scaled_dot_product_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), mask)
    return out.transpose(1, 2)


def attention(attn, hidden_states, rotary_emb, layer_cache, kv_cache_mode, prefix_len, segments, key_valid):
    """One attention layer. Prefill runs the block-causal structure as one SDPA call per prefix segment plus one
    for the target image; decoding runs the target's queries against the cached prefix and itself."""
    heads = attn.heads
    query = attn.to_q(hidden_states).unflatten(-1, (heads, -1))
    key = attn.to_k(hidden_states).unflatten(-1, (heads, -1))
    value = attn.to_v(hidden_states).unflatten(-1, (heads, -1))
    query = rotate(F.rms_norm(query, query.shape[-1:], attn.norm_q.weight, attn.norm_q.eps), rotary_emb)
    key = rotate(F.rms_norm(key, key.shape[-1:], attn.norm_k.weight, attn.norm_k.eps), rotary_emb)
    query, key = query.type_as(value), key.type_as(value)

    if kv_cache_mode == "cached":
        cached_key, cached_value = layer_cache.get()
        key = torch.cat([cached_key, key], dim=1)
        value = torch.cat([cached_value, value], dim=1)
        out = attend(query, key, value, None if key_valid is None else key_valid[:, None, None, :])
    else:
        if kv_cache_mode == "extract":
            # `clone()`: at batch size 1 the slice counts as contiguous, and the cache would pin the whole prefill.
            layer_cache.store(key[:, :prefix_len].clone(), value[:, :prefix_len].clone())
        outputs = []
        for start, end, is_text in segments:
            mask = None
            if is_text:
                length = end - start
                mask = torch.ones(length, end, dtype=torch.bool, device=query.device)
                mask[:, start:].tril_()
                mask = mask[None, None]
            if key_valid is not None:
                valid = key_valid[:, None, None, :end]
                mask = valid if mask is None else mask & valid
            outputs.append(attend(query[:, start:end], key[:, :end], value[:, :end], mask))
        outputs.append(
            attend(query[:, prefix_len:], key, value, None if key_valid is None else key_valid[:, None, None, :])
        )
        out = torch.cat(outputs, dim=1)
    return attn.to_out[0](out.flatten(2, 3))


def block_forward(block, hidden_states, modulation, rotary_emb, layer_cache, kv_cache_mode, segments, key_valid):
    eps = block.img_norm1.eps
    modulated = modulation.norm(hidden_states, modulation.scale1, eps)
    attn_output = attention(
        block.attn, modulated, rotary_emb, layer_cache, kv_cache_mode, modulation.prefix_len, segments, key_valid
    )
    hidden_states = torch.addcmul(hidden_states, attn_output, modulation.gate1)
    modulated = modulation.norm(hidden_states, modulation.scale2, eps)
    mlp = block.img_mlp
    mlp_output = mlp.out(F.silu(mlp.gate_layer(modulated)) * mlp.proj(modulated))
    return torch.addcmul(hidden_states, mlp_output, modulation.gate2)


def transformer_forward(
    self,
    hidden_states,
    encoder_hidden_states,
    timestep,
    img_shapes,
    img_mask,
    encoder_hidden_states_mask=None,
    attention_kwargs=None,
    kv_cache=None,
    kv_cache_mode=None,
    return_dict=True,
):
    """`QwenImage21Transformer2DModel.forward` with the layout cached on `kv_cache` and only the target image's rows
    projected out, which is all the pipeline reads."""
    batch_size = hidden_states.shape[0]
    layout = getattr(kv_cache, "layout", None)
    if layout is None:
        layout = RunLayout(self, img_shapes[0], img_mask, encoder_hidden_states_mask)
        if kv_cache is not None:
            kv_cache.layout = layout

    # Text and condition-image tokens modulate from the trailing t=0 row, so their activations are the same at
    # every step, which is what makes their keys and values cacheable.
    timestep = timestep.to(hidden_states.dtype)
    timestep = torch.cat([timestep, timestep.new_zeros(1)], dim=0)
    temb = self.time_text_embed(timestep, hidden_states)
    params = self.modulation(temb)

    if kv_cache_mode == "cached":
        # Only the target image's queries are recomputed; the block-causal structure degenerates to full attention
        # for them, since they see the whole prefix and their own block.
        joint_hidden_states = self.img_in(hidden_states[:, -layout.target_tokens :])
        rotary_emb = layout.rotary_emb[layout.prefix_len :]
        modulation = Modulation(params, 0, None)
        segments = None
    else:
        hidden_states = self.img_in(hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)
        joint_hidden_states = torch.cat(
            [
                encoder_hidden_states,
                encoder_hidden_states.new_zeros(batch_size, layout.target_tokens // 4, encoder_hidden_states.shape[2]),
            ],
            dim=1,
        )
        joint_hidden_states = joint_hidden_states.repeat_interleave(layout.repeats, dim=1)
        joint_hidden_states[:, layout.image_pad_mask] = hidden_states
        rotary_emb = layout.rotary_emb
        modulation = Modulation(params, layout.prefix_len, layout.target_token_mask)
        segments = layout.segments

    for index, block in enumerate(self.transformer_blocks):
        layer_cache = kv_cache.get_layer(index) if kv_cache is not None else None
        joint_hidden_states = block_forward(
            block, joint_hidden_states, modulation, rotary_emb, layer_cache, kv_cache_mode, segments, layout.key_valid
        )

    target = joint_hidden_states[:, -layout.target_tokens :]
    norm_out = self.norm_out
    scale = norm_out.linear(norm_out.silu(temb[:-1]).to(target.dtype)).unsqueeze(1)
    output = self.proj_out(scaled_layer_norm(target, scale, norm_out.norm.eps))
    return Transformer2DModelOutput(sample=output) if return_dict else (output,)


def encode_without_logits(text_encoder):
    """The pipeline reads the encoder's last hidden state and nothing else, yet the encoder's forward projects every
    position onto its 152k-token vocabulary."""
    forward = text_encoder.forward

    def encode(*args, **kwargs):
        return forward(*args, logits_to_keep=1, **kwargs)

    text_encoder.forward = encode


def apply(pipeline):
    transformer = pipeline.transformer
    if not transformer.config.causal_condition:
        raise ValueError("The transformer's text and condition-image tokens must modulate from t=0.")
    transformer.forward = transformer_forward.__get__(transformer)
    encode_without_logits(pipeline.text_encoder)
