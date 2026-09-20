# Compositor

Adobe Photoshop costs too much and tools like GIMP don’t feel familiar enough for me to stay in flow. That’s why I built Compositor.

The goal was to create a full-featured image editor that is completely free and open source. I use Photoshop for compositing and post-processing, so Compositor is built around that workflow - with the tools needed to create a pixel-perfect final image.

Because it’s open source, you can download the Xcode project and add, remove, or modify any feature to fit your workflow.

## Features

### Layers
- Layers and folders, with blend modes and opacity
- Layer masks: paint, fill, invert, blur and feather them; link or unlink them to transform a mask on its own
- Clipping masks and folder masks
- Adjustment layers: Hue/Saturation, Levels, Curves, Exposure, Gradient Map and Grain
- Merge Down, Merge Layers and Merge Group (⌘E)
- Duplicate, rename inline, reorder and nest by drag and drop; Option-drag to duplicate
- Drag layers between open projects

### Transform
- Non-destructive move, scale, rotate and flip — images keep their full resolution however small you make them
- Free distort (⌘-drag a handle), with Shift to lock to an axis
- Transform several layers, or a whole folder, together
- Snapping to canvas and layer edges and centers, with guides
- Exact values for position, size, scale and angle, stepped with the arrow keys
- Flip Layer and Flip Canvas, horizontal and vertical

### Selections
- Rectangle and Ellipse Marquee, Freehand and Polygonal Lasso, and Magic Wand
- Add to and subtract from selections, move the outline, or move and duplicate the pixels inside
- Load a layer's pixels or a mask as a selection
- Content-Aware Fill, which can also extend an image past its edges

### Painting and retouching
- Brush with size, hardness and opacity, and Shift for straight lines
- Spot Healing Brush (content-aware)
- Clone Stamp, aligned or not, sampling one layer or all of them
- Blur tool, on pixels or masks
- Gradient tool and Shape tool (rectangles, rounded rectangles and ellipses)
- Type tool (T): inline multiline editing in draggable, resizable paragraph boxes; font, size, color, alignment and spacing in the tool header; transform text and use it as a clipping mask
- Eyedropper and a full color picker

### Adjustments and filters
- Levels (with Auto), Curves, Hue/Saturation, Exposure, Gradient Map, Grain and Invert
- Gaussian Blur and Motion Blur that spread past a layer's edges
- Add Noise, Lens Correction and Remove Background
- Live previews, limited to the selection when there is one

### AI generation and editing
- Generate images and edit layers with [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1), which runs on your Mac (AI > Generate Image…)
- Edit the active layer or the whole canvas from a prompt; with a selection, only that part is sent and changed
- Send up to ten layers or image files as reference images and name them in the prompt by number ("the jacket from image 2")
- Work at 0.5, 1, 2.4 or 4 megapixels; 0.5 is a quick draft
- Transparent backgrounds: the model returns a real alpha channel
- Every result arrives as a new layer, so nothing underneath changes
- Setup is one button: Compositor downloads [uv](https://docs.astral.sh/uv/), a Python environment and the model (about 36 GB) into `~/Library/Application Support/Compositor/AI`, and Uninstall removes them

### Canvas and files
- Multiple projects in tabs
- Crop with snapping, and Option for symmetric cropping
- Canvas Size and Image Size
- Sharp high-quality downsampling when zoomed out, and a pixel grid when zoomed in
- Import JPEG, PNG, HEIC and TIFF — including dropped screenshots and images from other apps
- Export JPEG with a live preview (⇧⌥⌘S); Copy Merged
- Photoshop-style keyboard shortcuts throughout

## Requirements

- macOS 26
- Xcode 26 (to build from source)
- For AI generation: an Apple silicon Mac, 64 GB of memory recommended, and about 40 GB of free disk space

The model is under the Qwen Research License, which doesn't allow commercial use. Compositor's own license doesn't cover it.

## Building

Open `Compositor.xcodeproj` and run the **Compositor** scheme.

### How the AI helper is built

The editor is sandboxed, and the sandbox refuses to run anything a sandboxed process downloaded, Python included. So the model runs from `CompositorAI`, an XPC service inside the app bundle that isn't sandboxed. It installs a pinned, checksummed uv, syncs the locked environment in `CompositorAI/Python`, and keeps one Python worker alive between generations so the model loads once. The editor and the helper share `Shared/AIRuntimeProtocol.swift`. Because the helper is unsandboxed, the app can ship with Developer ID but not on the Mac App Store.

To move to newer packages, edit `CompositorAI/Python/pyproject.toml` and run `uv lock` there; installed copies resync when the lock file changes.

## Releasing

`scripts/release.sh` builds a Release version, signs it with Developer ID, notarizes and staples it, and packages it into `dist/Compositor-<version>.dmg`.

It needs, all kept outside this repository:

- a **Developer ID Application** certificate in the login keychain
- notarization credentials saved with `xcrun notarytool store-credentials "compositor-notary" …`
- [`create-dmg`](https://github.com/create-dmg/create-dmg) (`brew install create-dmg`)

## License

MIT — see [LICENSE](LICENSE).
