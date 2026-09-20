import Foundation

/// The AI helper's entry point: an XPC service that serves the one app it is bundled in.
final class ServiceDelegate: NSObject, NSXPCListenerDelegate {
    private let host = AIRuntimeHost()

    func listener(_ listener: NSXPCListener, shouldAcceptNewConnection connection: NSXPCConnection) -> Bool {
        connection.exportedInterface = NSXPCInterface(with: AIRuntimeService.self)
        connection.exportedObject = host
        connection.remoteObjectInterface = NSXPCInterface(with: AIRuntimeClient.self)
        host.client = connection.remoteObjectProxy as? AIRuntimeClient
        // The editor quit or crashed: nothing is left to hand a result to, so the model's memory goes back.
        connection.invalidationHandler = { [host] in host.disconnect() }
        connection.resume()
        return true
    }
}

let delegate = ServiceDelegate()
let listener = NSXPCListener.service()
listener.delegate = delegate
listener.resume()
