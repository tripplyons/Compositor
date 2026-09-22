"""Compositor's Qwen-Image-2.1 worker.

The AI helper starts this once, naming the model with --model and --revision, and keeps it running so the model
loads a single time. It reads one JSON request per line on stdin and writes one JSON event per line on stdout:

    {"op": "download"}
    {"op": "generate", "prompt": ..., "images": [paths], "width": ..., "height": ..., "resolution": ...,
     "steps": ..., "seed": ..., "output": path}
    {"op": "cancel"}

    {"event": "progress", "stage": "download" | "load" | "generate", "fraction": 0...1}
    {"event": "done"} | {"event": "cancelled"} | {"event": "error", "message": ...}

stdin closing means the helper is gone, and the worker exits with it.
"""

import argparse
import json
import os
import queue
import sys
import threading
import traceback
from pathlib import Path

arguments = argparse.ArgumentParser()
arguments.add_argument("--model", required=True)
arguments.add_argument("--revision", required=True)
MODEL = arguments.parse_args().model
MODEL_REVISION = arguments.parse_args().revision

# Libraries print to stdout; the protocol keeps the real one and everything else goes to the helper's log.
events = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
sys.stdout = sys.stderr

requests = queue.Queue()
cancelled = threading.Event()
pipeline = None


def emit(event, **fields):
    events.write(json.dumps({"event": event, **fields}) + "\n")


def read_requests():
    for line in sys.stdin:
        request = json.loads(line)
        if request["op"] == "cancel":
            cancelled.set()
            if pipeline is not None:
                pipeline._interrupt = True
        else:
            requests.put(request)
    os._exit(0)


def download():
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE

    info = HfApi().model_info(MODEL, revision=MODEL_REVISION, files_metadata=True)
    total = sum(file.size or 0 for file in info.siblings)
    blobs = Path(HF_HUB_CACHE) / f"models--{MODEL.replace('/', '--')}" / "blobs"
    finished = threading.Event()

    # The hub reports progress per file through tqdm bars of its own; the bytes on disk, partial files included,
    # say the same thing for the whole model.
    def report():
        while not finished.wait(1):
            size = sum(blob.stat().st_size for blob in blobs.glob("*")) if blobs.exists() else 0
            emit("progress", stage="download", fraction=min(1, size / total))

    threading.Thread(target=report, daemon=True).start()
    try:
        snapshot_download(MODEL, revision=MODEL_REVISION)
    finally:
        finished.set()


def load():
    global pipeline
    if pipeline is not None:
        return
    emit("progress", stage="load", fraction=0)
    import torch
    from diffusers import QwenImage21Pipeline

    import speedups

    loaded = QwenImage21Pipeline.from_pretrained(
        MODEL, revision=MODEL_REVISION, torch_dtype=torch.bfloat16, local_files_only=True
    )
    speedups.apply(loaded)
    pipeline = loaded.to("mps")


def generate(request):
    import torch
    from PIL import Image

    load()
    if cancelled.is_set():
        return emit("cancelled")
    steps = request["steps"]
    emit("progress", stage="generate", fraction=0)

    def step_finished(running, index, _timestep, tensors):
        # The pipeline clears its interrupt flag as a run starts, which can swallow a cancel sent just before.
        if cancelled.is_set():
            running._interrupt = True
        emit("progress", stage="generate", fraction=(index + 1) / steps)
        return tensors

    images = [Image.open(path).convert("RGBA") for path in request["images"]]
    result = pipeline(
        prompt=request["prompt"],
        image=images or None,
        width=request["width"],
        height=request["height"],
        output_resolution=request["resolution"],
        num_inference_steps=steps,
        generator=torch.Generator("cpu").manual_seed(request["seed"]),
        callback_on_step_end=step_finished,
    ).images[0]
    if cancelled.is_set():
        return emit("cancelled")
    result.save(request["output"])
    emit("done")


def main():
    threading.Thread(target=read_requests, daemon=True).start()
    emit("ready")
    while True:
        request = requests.get()
        try:
            if request["op"] == "download":
                download()
                emit("done")
            elif request["op"] == "generate":
                generate(request)
            else:
                emit("error", message=f"Unknown request: {request['op']}")
        except Exception as error:
            traceback.print_exc()
            emit("error", message=str(error) or type(error).__name__)
        finally:
            cancelled.clear()
            if "torch" in sys.modules:
                sys.modules["torch"].mps.empty_cache()


if __name__ == "__main__":
    main()
