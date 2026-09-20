import Foundation

extension Process {
    /// Runs to the end and returns the exit status, handing over each line of output (stdout and stderr together)
    /// as it arrives.
    func run(onLine: @escaping @Sendable (String) -> Void) async throws -> Int32 {
        let pipe = Pipe()
        let lines = LineSplitter(onLine: onLine)
        standardOutput = pipe
        standardError = pipe
        standardInput = FileHandle.nullDevice
        pipe.fileHandleForReading.readabilityHandler = { lines.append($0.availableData) }
        return try await withCheckedThrowingContinuation { continuation in
            terminationHandler = { process in
                pipe.fileHandleForReading.readabilityHandler = nil
                lines.append((try? pipe.fileHandleForReading.readToEnd()) ?? Data())
                lines.finish()
                continuation.resume(returning: process.terminationStatus)
            }
            do { try run() } catch {
                terminationHandler = nil
                pipe.fileHandleForReading.readabilityHandler = nil
                continuation.resume(throwing: error)
            }
        }
    }
}

/// Cuts a stream of bytes into lines. A carriage return ends a line too, which is how progress bars redraw.
final class LineSplitter: @unchecked Sendable {
    private let lock = NSLock()
    private var pending = Data()
    private let onLine: @Sendable (String) -> Void

    init(onLine: @escaping @Sendable (String) -> Void) { self.onLine = onLine }

    func append(_ data: Data) {
        lock.lock()
        pending.append(data)
        var lines: [Data] = []
        while let end = pending.firstIndex(where: { $0 == 0x0A || $0 == 0x0D }) {
            lines.append(pending[pending.startIndex..<end])
            pending = pending[pending.index(after: end)...]
        }
        lock.unlock()
        for line in lines { deliver(line) }
    }

    func finish() {
        lock.lock()
        let rest = pending
        pending = Data()
        lock.unlock()
        deliver(rest)
    }

    private func deliver(_ line: Data) {
        guard let text = String(data: line, encoding: .utf8)?.trimmingCharacters(in: .whitespaces), !text.isEmpty else { return }
        onLine(text)
    }
}
