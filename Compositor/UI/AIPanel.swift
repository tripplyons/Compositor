import SwiftUI
import UniformTypeIdentifiers

/// Generate with AI: sets the model up the first time, then prompts it, with layers as its input images.
struct AIPanel: View {
    @Bindable var session: EditorSession
    /// Set when the helper couldn't be asked at all, which leaves no status to show.
    @State private var connectionFailure: String?
    @State private var confirmsUninstall = false
    private var runtime: AIRuntimeConnection { .shared }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            if let status = runtime.status {
                if status.isInstalled { prompt } else { setup(status) }
            } else if let connectionFailure {
                Text(connectionFailure).foregroundStyle(.orange).fixedSize(horizontal: false, vertical: true)
            } else {
                ProgressView().controlSize(.small)
            }
        }
        .padding(24).frame(width: 420).fixedSize()
        .task {
            do { try await runtime.refresh() } catch { connectionFailure = error.localizedDescription }
        }
    }

    // MARK: Setup

    @ViewBuilder private func setup(_ status: AIRuntimeStatus) -> some View {
        Text("Qwen-Image-2.1 runs on this Mac. Nothing you make with it leaves the computer.")
            .fixedSize(horizontal: false, vertical: true)
        Text("Setting it up downloads a Python environment and the model, about 36 GB in all, into Application Support. The model needs an Apple silicon Mac, and 64 GB of memory is recommended.")
            .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        Text("The model is under the [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE), which doesn’t allow commercial use.")
            .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        if runtime.activity == .installing { progress }
        failure
        Divider()
        HStack {
            Spacer()
            if runtime.activity == .installing {
                Button("Cancel") { runtime.cancel() }.keyboardShortcut(.cancelAction)
            } else {
                Button(status.environmentInstalled ? "Download Model" : "Download and Install") { perform { try await runtime.install() } }
                    .keyboardShortcut(.defaultAction).buttonStyle(.borderedProminent)
                    .disabled(runtime.activity != nil)
            }
        }
    }

    // MARK: Prompt

    @ViewBuilder private var prompt: some View {
        Picker("Mode", selection: $session.ai.mode) {
            ForEach(AIGenerationSettings.Mode.allCases, id: \.self) { Text($0.rawValue).tag($0) }
        }
        .pickerStyle(.segmented).labelsHidden()
        TextField(session.ai.mode == .edit ? "Describe the change" : "Describe the image", text: $session.ai.prompt, axis: .vertical)
            .lineLimit(4...8).textFieldStyle(.roundedBorder)
        if session.ai.mode == .edit {
            Picker("Edit", selection: $session.ai.source) {
                ForEach(AIGenerationSettings.Source.allCases, id: \.self) { Text($0.rawValue).tag($0) }
            }
            .help("What the model sees as image 1. With a selection, only that part of the canvas is sent and changed")
            if session.selection != nil {
                Text("Limited to the selection").font(.callout).foregroundStyle(.secondary)
            }
        }
        references
        Toggle("Transparent background", isOn: $session.ai.isTransparent)
            .help("Asks for an image with an alpha channel, in the wording the model was trained on")
        HStack(spacing: 16) {
            if session.ai.mode == .generate {
                Picker("Shape", selection: $session.ai.aspect) {
                    ForEach(AIAspect.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                }
            }
            Picker("Detail", selection: $session.ai.resolution) {
                ForEach(AIImageSize.resolutions, id: \.self) { Text(megapixels($0)).tag($0) }
            }
            .help("How many pixels the model works at. Time grows with them; 0.5 MP is a quick draft, below what the model was trained on")
        }
        HStack(spacing: 16) {
            LabeledContent("Steps") {
                TextField("Steps", value: Binding(get: { session.ai.steps }, set: { session.ai.steps = min(100, max(1, $0)) }), format: .number)
                    .frame(width: 56).textFieldStyle(.roundedBorder).multilineTextAlignment(.trailing)
            }
            LabeledContent("Seed") {
                TextField("Random", value: $session.ai.seed, format: .number.grouping(.never))
                    .frame(width: 110).textFieldStyle(.roundedBorder).multilineTextAlignment(.trailing)
            }
            .help("The same seed, prompt and settings make the same image")
            Spacer()
        }
        if let seed = session.aiLastSeed, session.ai.seed == nil {
            Text("The last image used seed \(String(seed))").font(.callout).foregroundStyle(.secondary)
        }
        if runtime.activity == .generating { progress }
        failure
        if session.aiResult != nil {
            HStack {
                Text("The image is ready.").fixedSize()
                Spacer()
                Button("Discard") { session.aiResult = nil }
                Button("Add as Layer") { session.addAIResult() }.disabled(!session.canAddAIResult)
            }
        }
        Divider()
        HStack {
            Button("Uninstall…") { confirmsUninstall = true }.disabled(runtime.activity != nil)
            Spacer()
            if runtime.activity == .generating {
                Button("Cancel") { runtime.cancel() }.keyboardShortcut(.cancelAction)
            } else {
                if let blocker = session.aiBlocker, !session.ai.prompt.isEmpty {
                    Text(blocker).font(.callout).foregroundStyle(.secondary)
                }
                Button(session.ai.mode == .edit ? "Edit" : "Generate") { Task { await session.runAI() } }
                    .keyboardShortcut(.defaultAction).buttonStyle(.borderedProminent)
                    .disabled(session.aiBlocker != nil || runtime.activity != nil)
            }
        }
        .confirmationDialog("Remove Qwen-Image-2.1 and its Python environment?", isPresented: $confirmsUninstall) {
            Button("Uninstall", role: .destructive) { perform { try await runtime.uninstall() } }
        } message: { Text("This frees about 36 GB. It can be downloaded again at any time.") }
    }

    /// Layers and files to send along. The prompt can call them by number, so each chosen one shows the number
    /// it has.
    @ViewBuilder private var references: some View {
        let candidates = session.aiReferenceCandidates
        let chosen = session.aiReferences.map(\.id)
        let first = session.ai.mode == .edit ? 2 : 1
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Reference Images").font(.callout).foregroundStyle(.secondary)
                Spacer()
                Button("Add Images…", action: chooseReferenceFiles).controlSize(.small)
            }
            if !candidates.isEmpty {
                ScrollView {
                    VStack(alignment: .leading, spacing: 4) {
                        ForEach(candidates) { layer in
                            Toggle(isOn: Binding(get: { chosen.contains(layer.id) }, set: { _ in session.toggleAIReference(layer.id) })) {
                                HStack {
                                    Text(layer.name).lineLimit(1)
                                    Spacer()
                                    if let index = chosen.firstIndex(of: layer.id) {
                                        Text("image \(index + first)").font(.callout).foregroundStyle(.secondary)
                                    }
                                }
                            }
                        }
                    }
                }
                .frame(height: min(CGFloat(candidates.count), 5) * 22)
            }
        }
        .help("Tick layers or add image files, then refer to them in the prompt by number: “the jacket from image 2”")
    }

    private func chooseReferenceFiles() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.jpeg, .png, .heic, .tiff]
        panel.allowsMultipleSelection = true
        panel.begin { response in
            guard response == .OK else { return }
            Task { await session.addAIReferenceFiles(panel.urls) }
        }
    }

    private func megapixels(_ resolution: Int) -> String {
        let value = Double(resolution * resolution) / 1_000_000
        return "\(value.formatted(.number.precision(.fractionLength(0...1)))) MP"
    }

    // MARK: Shared

    @ViewBuilder private var progress: some View {
        let current = runtime.progress
        VStack(alignment: .leading, spacing: 4) {
            if let fraction = current?.fraction { ProgressView(value: fraction) } else { ProgressView().progressViewStyle(.linear) }
            Text(current.map(title) ?? "Starting…").font(.callout).foregroundStyle(.secondary)
            if let detail = current?.detail {
                Text(detail).font(.caption).foregroundStyle(.tertiary).lineLimit(1).truncationMode(.middle)
            }
        }
    }

    private func title(_ progress: AIProgress) -> String {
        switch progress.stage {
        case .tools: "Downloading uv…"
        case .environment: "Installing Python and PyTorch…"
        case .download: "Downloading the model…"
        case .load: "Loading the model, which takes a minute…"
        case .generate: "Generating…"
        }
    }

    @ViewBuilder private var failure: some View {
        if let failure = session.aiFailure {
            Text(failure).foregroundStyle(.orange).fixedSize(horizontal: false, vertical: true).textSelection(.enabled)
        }
    }

    private func perform(_ work: @escaping () async throws -> Void) {
        session.aiFailure = nil
        Task {
            do { try await work() } catch { session.aiFailure = error.localizedDescription }
        }
    }
}
