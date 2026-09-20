import AppKit

/// What the AI panel is set to. The session keeps it, so the panel reopens as it was left.
struct AIGenerationSettings {
    enum Mode: String, CaseIterable { case generate = "Generate", edit = "Edit" }
    enum Source: String, CaseIterable { case layer = "Active Layer", canvas = "Whole Canvas" }
    var mode = Mode.generate
    var source = Source.layer
    var prompt = ""
    var isTransparent = false
    var aspect = AIAspect.square
    var resolution = 1024
    var steps = 40
    /// Nil picks a new seed for every run.
    var seed: UInt32?
    /// Layers and files sent along as extra images, in the order the prompt can name them.
    var referenceIDs: [UUID] = []
    /// Reference images chosen from disk, which no layer holds.
    var referenceFiles: [AIReference] = []
}

/// An image the model is shown beside the prompt: a layer's pixels, or a file's.
struct AIReference: Identifiable {
    let id: UUID
    let name: String
    let image: CGImage
}

nonisolated enum AIAspect: String, CaseIterable {
    case canvas = "Canvas", square = "1:1", landscape = "4:3", portrait = "3:4", photo = "3:2", tallPhoto = "2:3",
         wide = "16:9", tall = "9:16"
    /// Width over height. Without a canvas to match, Canvas is square.
    func ratio(canvas: CGSize?) -> CGFloat {
        switch self {
        case .canvas: canvas.map { $0.width / $0.height } ?? 1
        case .square: 1
        case .landscape: 4 / 3
        case .portrait: 3 / 4
        case .photo: 3 / 2
        case .tallPhoto: 2 / 3
        case .wide: 16 / 9
        case .tall: 9 / 16
        }
    }
}

nonisolated enum AIImageSize {
    /// The side of the square each resolution matches in area: half a megapixel, 1, 2.4 and 4. The model is
    /// trained from 1 to 4; half is a quick draft.
    static let resolutions = [704, 1024, 1536, 2048]
    /// The model works in blocks of 32 pixels.
    static let block = 32

    /// The size closest to `ratio` with the area of a `resolution` square. Ratios past 4:1 are held there,
    /// which is as far as the model stays coherent.
    static func pixels(ratio: CGFloat, resolution: Int) -> (width: Int, height: Int) {
        let ratio = min(4, max(0.25, ratio))
        let area = CGFloat(resolution * resolution)
        func blocks(_ length: CGFloat) -> Int { max(1, Int((length / CGFloat(block)).rounded())) * block }
        return (blocks((area * ratio).squareRoot()), blocks((area / ratio).squareRoot()))
    }
}

nonisolated enum AIPrompt {
    /// The wording Qwen-Image-2.1 was trained on for images with an alpha channel.
    static func transparent(_ description: String) -> String {
        var description = description.trimmingCharacters(in: .whitespacesAndNewlines)
        while description.hasSuffix(".") { description.removeLast() }
        return "This is an RGBA image with transparency. \(description). The image has alpha channel and the background is transparent."
    }
}

nonisolated enum AIImageInput {
    /// A PNG of the image, reduced to `resolution`'s area when it is larger: the model resizes its inputs to that
    /// anyway, so nothing it would see is lost.
    static func png(of image: CGImage, resolution: Int) throws -> Data {
        let scale = min(1, (CGFloat(resolution * resolution) / CGFloat(image.width * image.height)).squareRoot())
        var image = image
        if scale < 1 {
            let width = max(1, Int(CGFloat(image.width) * scale)), height = max(1, Int(CGFloat(image.height) * scale))
            let context = try BrushRaster.context(width: width, height: height, mask: false)
            AIImageInput.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height), context: context)
            guard let reduced = context.makeImage() else { throw ExportError.render }
            image = reduced
        }
        guard let png = NSBitmapImageRep(cgImage: image).representation(using: .png, properties: [:]) else { throw ExportError.render }
        return png
    }

    /// Draws upright into a `BrushRaster` context, resampling smoothly.
    static func draw(_ image: CGImage, in rect: CGRect, context: CGContext) {
        context.saveGState()
        context.interpolationQuality = .high
        context.translateBy(x: rect.minX, y: rect.maxY)
        context.scaleBy(x: 1, y: -1)
        context.draw(image, in: CGRect(origin: .zero, size: rect.size))
        context.restoreGState()
    }
}

/// A finished image and where it belongs, held when the canvas was in the middle of something else.
struct AIResult {
    let image: CGImage
    let documentID: UUID?
    /// The part of the canvas an edit covered; nil centers a generated image.
    let region: CGRect?
}

extension EditorSession {
    /// What can be sent as a reference image: the chosen files, then the layers, top first as the Layers panel
    /// lists them.
    var aiReferenceCandidates: [AIReference] {
        let layers = document?.layers.reversed().filter { !$0.isGroup && !isAISource($0) } ?? []
        return ai.referenceFiles + layers.compactMap { layer in
            layer.asset.map { AIReference(id: layer.id, name: layer.name, image: $0.image) }
        }
    }

    private func isAISource(_ layer: ImageLayer) -> Bool {
        ai.mode == .edit && ai.source == .layer && layer.id == activeLayerID
    }

    /// The chosen references that still exist, in the order they were chosen.
    var aiReferences: [AIReference] {
        let candidates = aiReferenceCandidates
        return ai.referenceIDs.compactMap { id in candidates.first { $0.id == id } }
    }

    /// How many images a run would send: the edit's source, then the references.
    var aiInputCount: Int { (ai.mode == .edit ? 1 : 0) + aiReferences.count }

    /// Why a run can't start, or nil when it can.
    var aiBlocker: String? {
        if ai.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return "Describe the image first." }
        if aiInputCount > AIRuntime.maximumInputs { return "The model takes at most \(AIRuntime.maximumInputs) images." }
        guard ai.mode == .edit else { return nil }
        if document == nil { return "Open an image to edit." }
        if selection?.isEmpty == true { return "The selection is empty." }
        if ai.source == .layer, activeLayer?.asset == nil || activeLayer?.isGroup == true { return "Select a layer with pixels to edit." }
        return nil
    }

    /// The part of the canvas an edit sends: the selection's bounds, else the layer's, else all of it.
    private func aiEditRegion() -> CGRect? {
        guard let document else { return nil }
        if selection != nil { return selectionCopyRegion() }
        let canvas = CGRect(origin: .zero, size: document.size)
        guard ai.source == .layer, let layer = activeLayer else { return canvas }
        let transform = displayedTransform(for: layer)
        let corners = [CGPoint(x: 0, y: 0), CGPoint(x: 1, y: 0), CGPoint(x: 0, y: 1), CGPoint(x: 1, y: 1)].map(transform.point)
        let xs = corners.map(\.x), ys = corners.map(\.y)
        let bounds = CGRect(x: xs.min()!, y: ys.min()!, width: xs.max()! - xs.min()!, height: ys.max()! - ys.min()!)
        let region = bounds.integral.intersection(canvas)
        return region.isNull || region.width < 1 || region.height < 1 ? nil : region
    }

    /// The source as it sits on the canvas. The whole rectangle goes, not just the selected part of it, so the
    /// model sees the surroundings; the selection shapes the result instead.
    private func aiSourcePixels(in region: CGRect) throws -> CGImage {
        guard let document else { throw ExportError.render }
        let context = try BrushRaster.context(width: Int(region.width), height: Int(region.height), mask: false)
        context.translateBy(x: -region.minX, y: -region.minY)
        if ai.source == .layer, let layer = activeLayer, let image = layer.asset?.image {
            let transform = displayedTransform(for: layer)
            LayerRenderer.draw(image, transform: transform, center: transform.center, in: context)
        } else {
            drawLiveComposite(document, in: context)
        }
        guard let image = context.makeImage() else { throw ExportError.render }
        return image
    }

    /// Runs the panel's settings through the model and adds the result as a new layer. The canvas stays usable
    /// meanwhile, so everything the run needs is taken from it up front.
    func runAI() async {
        let runtime = AIRuntimeConnection.shared
        guard aiBlocker == nil, runtime.activity == nil else { return }
        aiFailure = nil
        aiResult = nil
        do {
            var images: [CGImage] = []
            var region: CGRect?
            if ai.mode == .edit {
                guard let editRegion = aiEditRegion() else { aiFailure = "There is nothing on the canvas there to edit."; return }
                region = editRegion
                images.append(try aiSourcePixels(in: editRegion))
            }
            images += aiReferences.map(\.image)
            let ratio = region.map { $0.width / $0.height } ?? ai.aspect.ratio(canvas: document?.size)
            let size = AIImageSize.pixels(ratio: ratio, resolution: ai.resolution)
            let seed = ai.seed ?? UInt32.random(in: 0...UInt32.max)
            let resolution = ai.resolution
            let inputs = SendableImages(images)
            let request = AIImageRequest(
                prompt: ai.isTransparent ? AIPrompt.transparent(ai.prompt) : ai.prompt,
                images: try await Task.detached { try inputs.images.map { try AIImageInput.png(of: $0, resolution: resolution) } }.value,
                width: size.width, height: size.height, resolution: resolution, steps: ai.steps, seed: seed)
            let documentID = document?.id
            // The helper may have been busy for another tab since the check above.
            guard runtime.activity == nil else { return }
            guard let image = try await runtime.generate(request) else { return }
            aiLastSeed = seed
            aiResult = AIResult(image: image, documentID: documentID, region: region)
            addAIResult()
        } catch { aiFailure = error.localizedDescription }
    }

    var canAddAIResult: Bool {
        // An edit belongs to the canvas it was taken from; a generated image can go anywhere.
        guard let aiResult, aiResult.region == nil || aiResult.documentID == document?.id else { return false }
        return document == nil ? !isProjectBusy && !isImporting : canEditLayers
    }

    /// Adds the finished image as a layer, or leaves it with the panel when the canvas is busy with something else.
    func addAIResult() {
        guard canAddAIResult, let result = aiResult else { return }
        do {
            let name = nextLayerName()
            if let region = result.region {
                try addAIEdit(result.image, over: region, name: name)
            } else {
                if document == nil { createDocument(width: result.image.width, height: result.image.height) }
                guard let canvas = document?.size else { return }
                let image = try Self.sRGBCopy(of: result.image)
                let scale = min(1, canvas.width / CGFloat(image.width), canvas.height / CGFloat(image.height))
                let size = CGSize(width: (CGFloat(image.width) * scale).rounded(), height: (CGFloat(image.height) * scale).rounded())
                let origin = CGPoint(x: floor((canvas.width - size.width) / 2), y: floor((canvas.height - size.height) / 2))
                addPixelLayer(image, at: origin, size: size, name: name, editName: "Generate Image")
            }
            aiResult = nil
        } catch { aiFailure = error.localizedDescription }
    }

    /// An edit goes back over the region it came from. With a selection it is cut to the selection's shape, which
    /// takes the region's own pixel grid; without one it keeps every pixel the model made and is scaled to fit.
    private func addAIEdit(_ result: CGImage, over region: CGRect, name: String) throws {
        guard let document else { return }
        guard let clip = try selection?.clip(canvas: document.size) else {
            addPixelLayer(try Self.sRGBCopy(of: result), at: region.origin, size: region.size, name: name, editName: "Edit with AI")
            return
        }
        let context = try BrushRaster.context(width: Int(region.width), height: Int(region.height), mask: false)
        context.translateBy(x: -region.minX, y: -region.minY)
        clip.apply(to: context)
        AIImageInput.draw(result, in: region, context: context)
        guard let image = context.makeImage() else { throw ExportError.render }
        addPixelLayer(image, at: region.origin, name: name, editName: "Edit with AI", dropsSelection: false)
    }

    /// Layers are switched on and off; a file is only listed while it is on.
    func toggleAIReference(_ id: UUID) {
        if let index = ai.referenceIDs.firstIndex(of: id) {
            ai.referenceIDs.remove(at: index)
            ai.referenceFiles.removeAll { $0.id == id }
        } else { ai.referenceIDs.append(id) }
    }

    /// Adds image files as references, each one chosen as it arrives.
    func addAIReferenceFiles(_ urls: [URL]) async {
        aiFailure = nil
        for url in urls {
            let scoped = url.startAccessingSecurityScopedResource()
            defer { if scoped { url.stopAccessingSecurityScopedResource() } }
            do {
                let asset = try await ImageImporter.shared.decode(url)
                let reference = AIReference(id: UUID(), name: asset.name, image: asset.image)
                ai.referenceFiles.append(reference)
                ai.referenceIDs.append(reference.id)
            } catch { aiFailure = "\(url.lastPathComponent): \(error.localizedDescription)" }
        }
    }
}

/// Core Graphics images are immutable, so they can cross to the task that encodes them.
private nonisolated struct SendableImages: @unchecked Sendable {
    let images: [CGImage]
    init(_ images: [CGImage]) { self.images = images }
}
