import Foundation

/// Runs one install or generation at a time, and keeps the worker (and the model inside it) around between
/// generations.
actor AIRuntimeEngine {
    enum Failure: LocalizedError {
        case busy, notInstalled
        var errorDescription: String? {
            switch self {
            case .busy: "The AI helper is already working."
            case .notInstalled: "Qwen-Image-2.1 isn't installed yet."
            }
        }
    }
    private enum Work { case installing, downloading, generating }

    /// How long an idle worker keeps the model's memory, some 35 GB, before giving it back.
    private static let idleLimit: Duration = .seconds(600)
    private let environment = PythonEnvironment()
    private var work: Work?
    private var isCancelled = false
    private var installer: Process?
    private var worker: PythonWorker?
    private var idleShutdown: Task<Void, Never>?
    /// Holds off the system's idle exit, which would take the loaded model with it.
    private var activity: NSObjectProtocol?

    func status() -> AIRuntimeStatus {
        AIRuntimeStatus(environmentInstalled: environment.isInstalled, modelInstalled: environment.isModelInstalled)
    }

    private func begin(_ next: Work) throws {
        guard work == nil else { throw Failure.busy }
        work = next
        isCancelled = false
        idleShutdown?.cancel()
    }
    private func end() {
        work = nil
        installer = nil
        idleShutdown = Task {
            try? await Task.sleep(for: Self.idleLimit)
            if !Task.isCancelled { shutDown() }
        }
    }

    private func runningWorker() throws -> PythonWorker {
        if let worker, worker.isRunning { return worker }
        stopWorker()
        let started = try PythonWorker(environment: environment)
        worker = started
        activity = ProcessInfo.processInfo.beginActivity(options: [.automaticTerminationDisabled, .suddenTerminationDisabled],
                                                         reason: "Qwen-Image-2.1 is loaded")
        return started
    }
    private func shutDown() {
        guard work == nil else { return }
        stopWorker()
    }
    private func stopWorker() {
        worker?.terminate()
        worker = nil
        if let activity { ProcessInfo.processInfo.endActivity(activity) }
        activity = nil
    }

    func install(report: @escaping @Sendable (AIProgress) -> Void) async throws {
        try begin(.installing)
        defer { end() }
        if !environment.isInstalled {
            report(AIProgress(stage: .tools))
            try await environment.installUV()
            guard !isCancelled else { return }
            report(AIProgress(stage: .environment))
            let sync = try environment.syncProcess()
            installer = sync
            // uv writes what it is doing line by line; the last one is also the best account of a failure.
            let lastLine = LastLine()
            let status = try await sync.run { line in
                lastLine.value = line
                report(AIProgress(stage: .environment, detail: line))
            }
            guard !isCancelled else { return }
            guard status == 0 else { throw PythonEnvironment.Failure.uv(lastLine.value) }
            try environment.markInstalled()
        }
        if !environment.isModelInstalled {
            work = .downloading
            report(AIProgress(stage: .download, fraction: 0))
            do {
                guard try await runningWorker().perform(["op": "download"], report: report) == .done else { return }
            } catch where isCancelled { return }
            try environment.markModelInstalled()
        }
    }

    /// The PNG, or nil when the generation was cancelled.
    func generate(_ request: AIImageRequest, report: @escaping @Sendable (AIProgress) -> Void) async throws -> Data? {
        guard status().isInstalled else { throw Failure.notInstalled }
        try begin(.generating)
        defer { end() }
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        var inputs: [String] = []
        for (index, image) in request.images.enumerated() {
            let file = folder.appendingPathComponent("input-\(index + 1).png")
            try image.write(to: file)
            inputs.append(file.path)
        }
        let output = folder.appendingPathComponent("output.png")
        let outcome = try await runningWorker().perform([
            "op": "generate", "prompt": request.prompt, "images": inputs, "width": request.width,
            "height": request.height, "resolution": request.resolution, "steps": request.steps,
            "seed": Int(request.seed), "output": output.path,
        ], report: report)
        guard outcome == .done else { return nil }
        return try Data(contentsOf: output)
    }

    func cancel() {
        guard let work else { return }
        isCancelled = true
        switch work {
        case .installing: installer?.terminate()
        // A download can't be interrupted from inside Python; what it has fetched stays for the next try.
        case .downloading: worker?.terminate()
        case .generating: worker?.send(["op": "cancel"])
        }
    }

    /// The editor went away: stop, and give the model's memory back.
    func disconnect() {
        cancel()
        stopWorker()
    }

    func uninstall() throws {
        guard work == nil else { throw Failure.busy }
        shutDown()
        if FileManager.default.fileExists(atPath: environment.root.path) {
            try FileManager.default.removeItem(at: environment.root)
        }
    }
}

private final class LastLine: @unchecked Sendable {
    private let lock = NSLock()
    private var line = ""
    var value: String {
        get { lock.withLock { line } }
        set { lock.withLock { line = newValue } }
    }
}
