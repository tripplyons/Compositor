import Foundation

/// The long-lived Python process that holds the model. Requests go in and events come out as JSON lines; see
/// worker.py for the protocol.
final class PythonWorker: @unchecked Sendable {
    enum Outcome { case done, cancelled }
    enum Failure: LocalizedError {
        case quit(URL), failed(String)
        var errorDescription: String? {
            switch self {
            case .quit(let log): "The AI worker quit unexpectedly, which usually means it ran out of memory. Its log is at \(log.path)."
            case .failed(let message): message
            }
        }
    }
    private struct Event: Decodable, Sendable {
        var event: String
        var stage: AIProgress.Stage?
        var fraction: Double?
        var message: String?
    }

    private let process = Process()
    private let input = Pipe()
    private let events: AsyncStream<Event>
    private let log: URL

    init(environment: PythonEnvironment) throws {
        log = environment.workerLog
        FileManager.default.createFile(atPath: log.path, contents: nil)
        let output = Pipe()
        process.executableURL = environment.python
        process.arguments = [try PythonEnvironment.resource("worker", "py").path,
                             "--model", PythonEnvironment.model, "--revision", PythonEnvironment.modelRevision]
        process.environment = environment.workerVariables
        process.standardInput = input
        process.standardOutput = output
        process.standardError = try FileHandle(forWritingTo: log)
        let (events, continuation) = AsyncStream.makeStream(of: Event.self)
        self.events = events
        let lines = LineSplitter { line in
            if let event = try? JSONDecoder().decode(Event.self, from: Data(line.utf8)) { continuation.yield(event) }
        }
        output.fileHandleForReading.readabilityHandler = { lines.append($0.availableData) }
        process.terminationHandler = { _ in
            output.fileHandleForReading.readabilityHandler = nil
            continuation.finish()
        }
        try process.run()
    }

    var isRunning: Bool { process.isRunning }

    func send(_ request: [String: Any]) {
        guard var line = try? JSONSerialization.data(withJSONObject: request) else { return }
        line.append(0x0A)
        try? input.fileHandleForWriting.write(contentsOf: line)
    }

    /// Sends a request and follows it to its end. The engine runs one request at a time, so every event until then
    /// belongs to this one.
    func perform(_ request: [String: Any], report: (AIProgress) -> Void) async throws -> Outcome {
        send(request)
        for await event in events {
            switch event.event {
            case "progress":
                if let stage = event.stage { report(AIProgress(stage: stage, fraction: event.fraction)) }
            case "done": return .done
            case "cancelled": return .cancelled
            case "error": throw Failure.failed(event.message ?? "The AI worker failed.")
            default: break
            }
        }
        throw Failure.quit(log)
    }

    func terminate() { process.terminate() }
}
