import Foundation

/// What the editor asks of the AI helper, an XPC service inside the app. The editor is sandboxed, and the sandbox
/// won't run anything a sandboxed process downloaded, so the helper — which isn't — owns uv, the Python environment,
/// the model and the worker that runs it. Values cross as JSON, which keeps the Objective-C surface XPC needs small.
@objc nonisolated protocol AIRuntimeService {
    /// An encoded `AIRuntimeStatus`.
    func status(reply: @escaping @Sendable (Data) -> Void)
    /// Installs whatever is missing: uv, the Python environment, then the model. Replies with an error, if any.
    func install(reply: @escaping @Sendable (String?) -> Void)
    /// Takes an encoded `AIImageRequest`; replies with the PNG or an error. Both nil means it was cancelled.
    func generate(_ request: Data, reply: @escaping @Sendable (Data?, String?) -> Void)
    /// Stops the install or generation in progress.
    func cancel()
    /// Deletes the environment and the model.
    func uninstall(reply: @escaping @Sendable (String?) -> Void)
}

/// What the helper calls back on while it works.
@objc nonisolated protocol AIRuntimeClient {
    /// An encoded `AIProgress`.
    func report(_ progress: Data)
}

nonisolated enum AIRuntime {
    static let serviceName = "com.wonderassembly.compositor.ai"
    /// The model takes up to ten condition images.
    static let maximumInputs = 10
}

nonisolated struct AIRuntimeStatus: Codable, Equatable, Sendable {
    var environmentInstalled: Bool
    var modelInstalled: Bool
    var isInstalled: Bool { environmentInstalled && modelInstalled }
}

nonisolated struct AIImageRequest: Codable, Sendable {
    var prompt: String
    /// PNGs the model conditions on, in the order the prompt can name them: image 1, image 2, …
    var images: [Data]
    var width: Int
    var height: Int
    /// The side of the square whose area condition images are resized to.
    var resolution: Int
    var steps: Int
    var seed: UInt32
}

nonisolated struct AIProgress: Codable, Equatable, Sendable {
    enum Stage: String, Codable, Sendable {
        case tools, environment, download, load, generate
    }
    var stage: Stage
    /// Nil while the stage can't say how far along it is.
    var fraction: Double?
    /// The installer's latest line of output.
    var detail: String?
}
