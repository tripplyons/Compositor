import AppKit
import Observation

/// The editor's end of the AI helper: one connection for the whole app, since there is one helper and one model.
@Observable
final class AIRuntimeConnection {
    static let shared = AIRuntimeConnection()
    enum Activity { case installing, generating, uninstalling }
    enum Failure: LocalizedError {
        case helper(String), unreadableResult
        var errorDescription: String? {
            switch self {
            case .helper(let message): message
            case .unreadableResult: "The generated image couldn’t be read."
            }
        }
    }

    /// Nil until the helper has answered.
    private(set) var status: AIRuntimeStatus?
    private(set) var activity: Activity?
    private(set) var progress: AIProgress?
    @ObservationIgnored private var connection: NSXPCConnection?

    private func service(_ fail: @escaping @Sendable (Error) -> Void) -> AIRuntimeService {
        if connection == nil {
            let opened = NSXPCConnection(serviceName: AIRuntime.serviceName)
            opened.remoteObjectInterface = NSXPCInterface(with: AIRuntimeService.self)
            opened.exportedInterface = NSXPCInterface(with: AIRuntimeClient.self)
            opened.exportedObject = ProgressReceiver { [weak self] progress in
                Task { @MainActor [weak self] in
                    if self?.activity != nil { self?.progress = progress }
                }
            }
            opened.invalidationHandler = { [weak self] in
                Task { @MainActor in self?.connection = nil }
            }
            opened.resume()
            connection = opened
        }
        return connection!.remoteObjectProxyWithErrorHandler(fail) as! AIRuntimeService
    }

    /// XPC calls either the reply or the error handler, never both, so the continuation resumes once.
    private func finish(_ call: (AIRuntimeService, @escaping @Sendable (String?) -> Void) -> Void) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            call(service { continuation.resume(throwing: $0) }) { message in
                continuation.resume(with: message.map { .failure(Failure.helper($0)) } ?? .success(()))
            }
        }
    }

    private func begin(_ next: Activity) {
        activity = next
        progress = nil
    }
    private func end() {
        activity = nil
        progress = nil
    }

    func refresh() async throws {
        status = try await withCheckedThrowingContinuation { continuation in
            service { continuation.resume(throwing: $0) }.status { data in
                continuation.resume(with: Result { try JSONDecoder().decode(AIRuntimeStatus.self, from: data) })
            }
        }
    }

    /// Returns normally when cancelled too; `status` says how far it got.
    func install() async throws {
        begin(.installing)
        defer { end() }
        do { try await finish { $0.install(reply: $1) } } catch {
            try? await refresh()
            throw error
        }
        try await refresh()
    }

    func uninstall() async throws {
        begin(.uninstalling)
        defer { end() }
        try await finish { $0.uninstall(reply: $1) }
        try await refresh()
    }

    /// The generated image, or nil when it was cancelled.
    func generate(_ request: AIImageRequest) async throws -> CGImage? {
        begin(.generating)
        defer { end() }
        let encoded = try JSONEncoder().encode(request)
        let png: Data? = try await withCheckedThrowingContinuation { continuation in
            service { continuation.resume(throwing: $0) }.generate(encoded) { png, message in
                continuation.resume(with: message.map { .failure(Failure.helper($0)) } ?? .success(png))
            }
        }
        guard let png else { return nil }
        guard let image = NSBitmapImageRep(data: png)?.cgImage else { throw Failure.unreadableResult }
        return image
    }

    func cancel() {
        guard activity == .installing || activity == .generating else { return }
        service { _ in }.cancel()
    }
}

private nonisolated final class ProgressReceiver: NSObject, AIRuntimeClient {
    private let deliver: @Sendable (AIProgress) -> Void
    init(deliver: @escaping @Sendable (AIProgress) -> Void) { self.deliver = deliver }
    func report(_ progress: Data) {
        if let progress = try? JSONDecoder().decode(AIProgress.self, from: progress) { deliver(progress) }
    }
}
