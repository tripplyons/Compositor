import Foundation

/// The object the editor talks to: decodes each request, hands it to the engine, and passes progress back.
final class AIRuntimeHost: NSObject, AIRuntimeService, @unchecked Sendable {
    private let engine = AIRuntimeEngine()
    private let lock = NSLock()
    private var currentClient: AIRuntimeClient?
    var client: AIRuntimeClient? {
        get { lock.withLock { currentClient } }
        set { lock.withLock { currentClient = newValue } }
    }

    private func report(_ progress: AIProgress) {
        guard let data = try? JSONEncoder().encode(progress) else { return }
        client?.report(data)
    }

    func status(reply: @escaping @Sendable (Data) -> Void) {
        Task {
            let status = await engine.status()
            reply((try? JSONEncoder().encode(status)) ?? Data())
        }
    }

    func install(reply: @escaping @Sendable (String?) -> Void) {
        Task {
            do {
                try await engine.install { self.report($0) }
                reply(nil)
            } catch { reply(error.localizedDescription) }
        }
    }

    func generate(_ request: Data, reply: @escaping @Sendable (Data?, String?) -> Void) {
        Task {
            do {
                let request = try JSONDecoder().decode(AIImageRequest.self, from: request)
                reply(try await engine.generate(request) { self.report($0) }, nil)
            } catch { reply(nil, error.localizedDescription) }
        }
    }

    func cancel() {
        Task { await engine.cancel() }
    }

    func disconnect() {
        client = nil
        Task { await engine.disconnect() }
    }

    func uninstall(reply: @escaping @Sendable (String?) -> Void) {
        Task {
            do {
                try await engine.uninstall()
                reply(nil)
            } catch { reply(error.localizedDescription) }
        }
    }
}
