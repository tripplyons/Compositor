import AppKit
import Testing
@testable import Compositor

@MainActor
struct AIGenerationTests {
    private func solid(width: Int, height: Int, _ red: CGFloat, _ green: CGFloat, _ blue: CGFloat) throws -> CGImage {
        let context = try BrushRaster.context(width: width, height: height, mask: false)
        context.setFillColor(CGColor(srgbRed: red, green: green, blue: blue, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        return try #require(context.makeImage())
    }
    private func pixel(_ image: CGImage, x: Int, y: Int) throws -> [Int] {
        let context = try #require(CGContext(data: nil, width: image.width, height: image.height, bitsPerComponent: 8,
            bytesPerRow: image.width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue))
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        let bytes = try #require(context.data).assumingMemoryBound(to: UInt8.self)
        let index = (y * image.width + x) * 4
        return (0..<4).map { Int(bytes[index + $0]) }
    }
    /// A 200×100 canvas with a red layer under a blue one.
    private func makeSession() throws -> EditorSession {
        let session = EditorSession()
        session.createDocument(width: 200, height: 100)
        session.addPixelLayer(try solid(width: 200, height: 100, 1, 0, 0), at: .zero, name: "Red", editName: "Add")
        session.addPixelLayer(try solid(width: 50, height: 50, 0, 0, 1), at: CGPoint(x: 10, y: 20), name: "Blue", editName: "Add")
        return session
    }

    @Test func sizesKeepTheAreaInBlocksOf32() {
        #expect(AIImageSize.pixels(ratio: 1, resolution: 1024) == (1024, 1024))
        #expect(AIImageSize.pixels(ratio: 1, resolution: 2048) == (2048, 2048))
        let wide = AIImageSize.pixels(ratio: 16 / 9, resolution: 1024)
        #expect(wide == (1376, 768))
        let tall = AIImageSize.pixels(ratio: 9 / 16, resolution: 1024)
        #expect(tall == (768, 1376))
        for ratio in [0.3, 0.75, 1.3, 1.5, 2.2] {
            let size = AIImageSize.pixels(ratio: ratio, resolution: 1536)
            #expect(size.width % 32 == 0 && size.height % 32 == 0)
            #expect(abs(Double(size.width * size.height) / Double(1536 * 1536) - 1) < 0.06)
        }
    }

    @Test func extremeRatiosStopAtFourToOne() {
        #expect(AIImageSize.pixels(ratio: 40, resolution: 1024) == (2048, 512))
        #expect(AIImageSize.pixels(ratio: 0.01, resolution: 1024) == (512, 2048))
    }

    @Test func canvasAspectFollowsTheCanvas() {
        #expect(AIAspect.canvas.ratio(canvas: CGSize(width: 300, height: 100)) == 3)
        #expect(AIAspect.canvas.ratio(canvas: nil) == 1)
    }

    @Test func transparentPromptUsesTheModelsWording() {
        #expect(AIPrompt.transparent(" A red apple. ")
            == "This is an RGBA image with transparency. A red apple. The image has alpha channel and the background is transparent.")
    }

    @Test func largeInputsAreReducedToTheWorkingArea() throws {
        let png = try AIImageInput.png(of: try solid(width: 400, height: 100, 0, 1, 0), resolution: 100)
        let image = try #require(NSBitmapImageRep(data: png)?.cgImage)
        #expect(image.width == 200 && image.height == 50)
        #expect(try pixel(image, x: 100, y: 25) == [0, 255, 0, 255])
        let small = try AIImageInput.png(of: try solid(width: 40, height: 10, 0, 1, 0), resolution: 100)
        #expect(NSBitmapImageRep(data: small)?.pixelsWide == 40)
    }

    @Test func referencesKeepTheOrderTheyWereChosenInAndLeaveOutTheSource() throws {
        let session = try makeSession()
        let red = try #require(session.document?.layers.first { $0.name == "Red" }?.id)
        let blue = try #require(session.document?.layers.first { $0.name == "Blue" }?.id)
        session.ai.mode = .generate
        session.toggleAIReference(red)
        session.toggleAIReference(blue)
        #expect(session.aiReferences.map(\.id) == [red, blue] && session.aiInputCount == 2)
        // Blue is the active layer, so as the edit's source it is image 1 already.
        session.ai.mode = .edit
        #expect(session.aiReferences.map(\.id) == [red] && session.aiInputCount == 2)
        session.toggleAIReference(red)
        #expect(session.aiReferences.isEmpty)
    }

    @Test func imageFilesAreReferencesWithoutADocument() async throws {
        let url = FileManager.default.temporaryDirectory.appending(path: "\(UUID().uuidString).png")
        try AIImageInput.png(of: try solid(width: 40, height: 20, 0, 1, 0), resolution: 100).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let session = EditorSession()
        session.ai.mode = .generate
        await session.addAIReferenceFiles([url, url])
        #expect(session.aiFailure == nil)
        #expect(session.aiReferences.map(\.image.width) == [40, 40] && session.aiInputCount == 2)
        // Switching a file off takes it out of the list; the other copy stays.
        session.toggleAIReference(try #require(session.aiReferences.first?.id))
        #expect(session.aiReferenceCandidates.count == 1 && session.aiReferences.count == 1)
    }

    @Test func theDraftSizeIsHalfAMegapixel() {
        let size = AIImageSize.pixels(ratio: 1.5, resolution: 704)
        #expect(size.width % 32 == 0 && size.height % 32 == 0)
        #expect(abs(size.width * size.height - 500_000) < 40_000)
    }

    @Test func aRunNeedsAPromptAndSomethingToEdit() throws {
        let empty = EditorSession()
        #expect(empty.aiBlocker != nil)
        empty.ai.prompt = "a lighthouse"
        #expect(empty.aiBlocker == nil)
        empty.ai.mode = .edit
        #expect(empty.aiBlocker != nil)
        let session = try makeSession()
        session.ai.prompt = "make it green"
        session.ai.mode = .edit
        #expect(session.aiBlocker == nil)
    }

    @Test func aGeneratedImageWithoutACanvasMakesOne() throws {
        let session = EditorSession()
        session.aiResult = AIResult(image: try solid(width: 64, height: 32, 0, 1, 0), documentID: nil, region: nil)
        session.addAIResult()
        #expect(session.document?.size == CGSize(width: 64, height: 32) && session.aiResult == nil)
        #expect(session.activeLayer?.transform.origin == .zero && session.activeLayer?.size == CGSize(width: 64, height: 32))
    }

    @Test func aGeneratedImageIsFittedAndCenteredOnTheCanvas() throws {
        let session = try makeSession()
        session.aiResult = AIResult(image: try solid(width: 400, height: 400, 0, 1, 0), documentID: nil, region: nil)
        session.addAIResult()
        let layer = try #require(session.activeLayer)
        #expect(layer.size == CGSize(width: 100, height: 100) && layer.transform.origin == CGPoint(x: 50, y: 0))
        #expect(layer.asset?.image.width == 400 && session.history.undoName == "Generate Image")
    }

    @Test func anEditCoversItsRegionAtTheModelsResolution() throws {
        let session = try makeSession()
        let region = CGRect(x: 10, y: 20, width: 50, height: 50)
        session.aiResult = AIResult(image: try solid(width: 320, height: 320, 0, 1, 0), documentID: session.document?.id, region: region)
        session.addAIResult()
        let layer = try #require(session.activeLayer)
        #expect(layer.transform.origin == region.origin && layer.size == region.size && layer.asset?.image.width == 320)
        #expect(session.history.undoName == "Edit with AI")
    }

    @Test func anEditIsCutToTheSelection() throws {
        let session = try makeSession()
        session.applySelection(CGPath(ellipseIn: CGRect(x: 20, y: 20, width: 40, height: 40), transform: nil), mode: .replace, name: "Select")
        let region = try #require(session.selectionCopyRegion())
        session.aiResult = AIResult(image: try solid(width: 320, height: 320, 0, 1, 0), documentID: session.document?.id, region: region)
        session.addAIResult()
        let layer = try #require(session.activeLayer)
        let image = try #require(layer.asset?.image)
        #expect(layer.transform.origin == region.origin && image.width == Int(region.width) && session.selection != nil)
        #expect(try pixel(image, x: image.width / 2, y: image.height / 2) == [0, 255, 0, 255])
        #expect(try pixel(image, x: 0, y: 0)[3] == 0)
    }

    @Test func anEditWaitsForTheCanvasItCameFrom() throws {
        let session = try makeSession()
        session.aiResult = AIResult(image: try solid(width: 32, height: 32, 0, 1, 0), documentID: UUID(),
                                    region: CGRect(x: 0, y: 0, width: 32, height: 32))
        let layers = session.document?.layers.count
        session.addAIResult()
        #expect(!session.canAddAIResult && session.aiResult != nil && session.document?.layers.count == layers)
    }
}
