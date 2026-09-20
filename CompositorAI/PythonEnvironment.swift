import CryptoKit
import Foundation

/// Where the helper keeps uv, the Python environment it builds with it, and the model, and how each gets there.
/// Everything lives in one folder, so removing the feature is removing that folder.
struct PythonEnvironment: Sendable {
    static let uvVersion = "0.12.17"
    static let uvArchive = URL(string: "https://github.com/astral-sh/uv/releases/download/\(uvVersion)/uv-aarch64-apple-darwin.tar.gz")!
    static let uvArchiveSHA256 = "85f00cbdc6dd3e97eba4c31b4d014375a9fdfe8f570023b84e5102fc3456896b"
    static let model = "Qwen/Qwen-Image-2.1"
    static let modelRevision = "b3179ad355be050328e483a9dfdd9e60cd62adfa"

    enum Failure: LocalizedError {
        case unsupported, download(String), checksum, uv(String), missingResource(String)
        var errorDescription: String? {
            switch self {
            case .unsupported: "Qwen-Image-2.1 needs a Mac with Apple silicon."
            case .download(let reason): "uv could not be downloaded: \(reason)"
            case .checksum: "The uv download didn't match its checksum, so it was discarded."
            case .uv(let line): "The Python environment could not be installed: \(line)"
            case .missingResource(let name): "The AI helper is missing \(name)."
            }
        }
    }

    let root = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("Compositor/AI", isDirectory: true)
    var uv: URL { root.appendingPathComponent("uv-\(Self.uvVersion)/uv") }
    var project: URL { root.appendingPathComponent("environment", isDirectory: true) }
    var python: URL { project.appendingPathComponent(".venv/bin/python") }
    var models: URL { root.appendingPathComponent("models", isDirectory: true) }
    var workerLog: URL { root.appendingPathComponent("worker.log") }
    /// A copy of the lock file the environment was last built from, written once the build succeeds. An app update
    /// that changes the lock makes the two differ, and the environment is synced again.
    private var installedLock: URL { project.appendingPathComponent("installed.lock") }
    private var modelStamp: URL { models.appendingPathComponent("\(Self.modelRevision).installed") }

    static func resource(_ name: String, _ type: String) throws -> URL {
        guard let url = Bundle.main.url(forResource: name, withExtension: type) else {
            throw Failure.missingResource("\(name).\(type)")
        }
        return url
    }

    var isInstalled: Bool {
        guard let bundled = try? Data(contentsOf: Self.resource("uv", "lock")) else { return false }
        return FileManager.default.isExecutableFile(atPath: python.path) && (try? Data(contentsOf: installedLock)) == bundled
    }
    var isModelInstalled: Bool { FileManager.default.fileExists(atPath: modelStamp.path) }
    func markModelInstalled() throws { try Data().write(to: modelStamp) }

    /// uv's own environment: its cache and the Python it downloads stay in the helper's folder, and nothing of the
    /// user's — their uv settings, their Pythons — is read.
    private var uvVariables: [String: String] {
        ["HOME": NSHomeDirectory(), "PATH": "/usr/bin:/bin",
         "UV_CACHE_DIR": root.appendingPathComponent("cache").path,
         "UV_PYTHON_INSTALL_DIR": root.appendingPathComponent("python").path,
         "UV_PYTHON_PREFERENCE": "only-managed", "UV_NO_CONFIG": "1", "UV_NO_PROGRESS": "1"]
    }
    var workerVariables: [String: String] {
        ["HOME": NSHomeDirectory(), "PATH": "/usr/bin:/bin", "HF_HOME": models.path,
         "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"]
    }

    func installUV() async throws {
        #if !arch(arm64)
        throw Failure.unsupported
        #else
        guard !FileManager.default.isExecutableFile(atPath: uv.path) else { return }
        let archive: URL
        do { archive = try await URLSession.shared.download(from: Self.uvArchive).0 }
        catch { throw Failure.download(error.localizedDescription) }
        defer { try? FileManager.default.removeItem(at: archive) }
        let digest = SHA256.hash(data: try Data(contentsOf: archive)).map { String(format: "%02x", $0) }.joined()
        guard digest == Self.uvArchiveSHA256 else { throw Failure.checksum }
        let folder = uv.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let tar = Process()
        tar.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
        // The archive holds one folder; its contents go straight into ours.
        tar.arguments = ["-xzf", archive.path, "-C", folder.path, "--strip-components", "1"]
        guard try await tar.run(onLine: { _ in }) == 0 else { throw Failure.download("the archive could not be unpacked.") }
        #endif
    }

    /// `uv sync` for the bundled project; the caller runs the process so that it can stop it.
    func syncProcess() throws -> Process {
        try FileManager.default.createDirectory(at: project, withIntermediateDirectories: true)
        try? FileManager.default.removeItem(at: installedLock)
        for (name, type) in [("pyproject", "toml"), ("uv", "lock")] {
            let target = project.appendingPathComponent("\(name).\(type)")
            try? FileManager.default.removeItem(at: target)
            try FileManager.default.copyItem(at: Self.resource(name, type), to: target)
        }
        let process = Process()
        process.executableURL = uv
        process.arguments = ["sync", "--frozen", "--project", project.path]
        process.environment = uvVariables
        return process
    }
    func markInstalled() throws {
        try FileManager.default.copyItem(at: project.appendingPathComponent("uv.lock"), to: installedLock)
    }
}
