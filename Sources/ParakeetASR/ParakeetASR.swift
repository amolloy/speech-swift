import CoreML
import Foundation
import AudioCommon

/// Parakeet TDT 0.6B v3 — CoreML-based automatic speech recognition.
///
/// Uses a FastConformer encoder with a Token-and-Duration Transducer (TDT) decoder.
/// Mel preprocessing is done in Swift using Accelerate/vDSP. The encoder, decoder, and
/// joint network run on CoreML with INT8-quantized encoder for Neural Engine acceleration.
///
/// - Warning: This class is not thread-safe. Create separate instances for concurrent use.
public class ParakeetASRModel {
    /// Model configuration.
    public let config: ParakeetConfig

    /// Default HuggingFace model ID (INT8 quantized encoder, 30s max).
    public static let defaultModelId = "aufklarer/Parakeet-TDT-v3-CoreML-INT8"

    /// iOS-optimized model: single 500-frame shape (5s max), no EnumeratedShapes overhead.
    /// Saves ~600MB runtime memory vs the default model.
    public static let iosModelId = "aufklarer/Parakeet-TDT-v3-CoreML-INT8-iOS-5s"

    /// Whether the model is loaded and ready for inference.
    var _isLoaded = true

    private let melPreprocessor: MelPreprocessor
    var encoder: MLModel?
    var decoder: MLModel?
    var joint: MLModel?
    private let vocabulary: ParakeetVocabulary
    /// Mel frame counts the loaded encoder accepts, sorted ascending.
    /// Derived from the encoder's CoreML input constraint at load time, so
    /// fixed-shape exports (e.g. iOS-5s = `[500]`) and enumerated-shape
    /// exports (e.g. macOS = `[100, 200, 300, 400, 500, 750, 1000, 1500,
    /// 2000, 3000]`) both work without any per-variant Swift logic.
    /// Internal access so tests can assert discovery results.
    let supportedMelLengths: [Int]
    /// Confidence from the last transcription (0.0–1.0).
    public private(set) var lastConfidence: Float = 0
    /// Per-word confidence scores from the last transcription.
    public private(set) var lastWordConfidences: [WordConfidence]?

    private init(
        config: ParakeetConfig,
        encoder: MLModel?,
        decoder: MLModel?,
        joint: MLModel?,
        vocabulary: ParakeetVocabulary,
        supportedMelLengths: [Int]
    ) {
        self.config = config
        self.melPreprocessor = MelPreprocessor(config: config)
        self.encoder = encoder
        self.decoder = decoder
        self.joint = joint
        self.vocabulary = vocabulary
        self.supportedMelLengths = supportedMelLengths
    }

    // MARK: - Warmup

    /// Warm up CoreML models by running a dummy inference.
    ///
    /// CoreML compiles computation graphs on the first `prediction()` call, which causes
    /// a ~4x latency penalty on cold inference. Calling `warmUp()` triggers this compilation
    /// on a minimal 1-second silence input so that subsequent real transcriptions run at full speed.
    public func warmUp() throws {
        let dummyAudio = [Float](repeating: 0, count: config.sampleRate)  // 1s silence
        _ = try transcribeAudio(dummyAudio, sampleRate: config.sampleRate)
    }

    // MARK: - Transcription

    /// Transcribe audio to text.
    ///
    /// - Parameters:
    ///   - audio: PCM Float32 audio samples
    ///   - sampleRate: Sample rate of the input audio in Hz
    ///   - language: Language hint (unused, Parakeet auto-detects from 25 European languages)
    /// - Returns: Transcribed text
    /// - Throws: `AudioModelError` on CoreML inference failure
    public func transcribeAudio(_ audio: [Float], sampleRate: Int, language: String? = nil) throws -> String {
        // Resample to 16kHz if needed
        let samples: [Float]
        if sampleRate != config.sampleRate {
            samples = AudioFileLoader.resample(audio, from: sampleRate, to: config.sampleRate)
        } else {
            samples = audio
        }

        // Step 1: Mel spectrogram extraction (Swift/Accelerate)
        AudioLog.inference.debug("Parakeet: preprocessing \(samples.count) samples")
        let tMel0 = CFAbsoluteTimeGetCurrent()
        let (mel, melLength) = try melPreprocessor.extract(samples)
        let tMel1 = CFAbsoluteTimeGetCurrent()

        // Step 2: Encoder — mel → encoded representations
        // Pad / truncate mel to a shape the encoder accepts. `effectiveLength`
        // is what we mask the encoder with — it equals `melLength` for normal
        // pad-up cases and the encoder's max for truncated overflows.
        let (paddedMel, effectiveLength) = try padMelToEnumeratedShape(mel: mel, actualLength: melLength)
        AudioLog.inference.debug("Parakeet: encoding \(effectiveLength) mel frames, tensor width \(paddedMel.shape[2])")
        let tEnc0 = CFAbsoluteTimeGetCurrent()
        let encoderOutput = try runEncoder(mel: paddedMel, length: effectiveLength)
        let tEnc1 = CFAbsoluteTimeGetCurrent()
        let encoded = encoderOutput.featureValue(for: "encoded")!.multiArrayValue!
        let encodedLengthArray = encoderOutput.featureValue(for: "encoded_length")!.multiArrayValue!
        let encodedLength = encodedLengthArray[0].intValue

        // Step 3: TDT greedy decode
        let tdtDecoder = TDTGreedyDecoder(config: config, decoder: decoder!, joint: joint!)
        let tDec0 = CFAbsoluteTimeGetCurrent()
        let (tokenIds, tokenLogProbs, confidence) = try tdtDecoder.decode(encoded: encoded, encodedLength: encodedLength)
        let tDec1 = CFAbsoluteTimeGetCurrent()

        // Step 4: Vocabulary decode with per-word confidence
        let text = vocabulary.decode(tokenIds)
        lastConfidence = confidence
        lastWordConfidences = vocabulary.decodeWords(tokenIds, logProbs: tokenLogProbs)

        let melMs = (tMel1 - tMel0) * 1000
        let encMs = (tEnc1 - tEnc0) * 1000
        let decMs = (tDec1 - tDec0) * 1000
        let totalMs = melMs + encMs + decMs
        AudioLog.inference.info("Parakeet: mel=\(String(format: "%.1f", melMs))ms enc=\(String(format: "%.1f", encMs))ms dec=\(String(format: "%.1f", decMs))ms total=\(String(format: "%.1f", totalMs))ms (\(tokenIds.count) tokens, \(encodedLength) frames)")

        return text
    }

    // MARK: - CoreML Inference Helpers

    /// Fallback used only if the encoder's input constraint can't be introspected.
    /// Real values come from `discoverSupportedMelLengths(from:)` at load time.
    private static let defaultMelLengths = [100, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000]

    /// Pad / truncate mel spectrogram to the smallest supported shape that
    /// fits. Supported shapes come from the encoder's CoreML input constraint
    /// at load time (see `discoverSupportedMelLengths(from:)`), so fixed-shape
    /// exports work without forcing the caller to know about them.
    ///
    /// If the mel exceeds the encoder's max supported length, we keep the
    /// most recent frames (i.e., drop from the start). For voice pipelines
    /// this preserves the actual speech (which sits at the end of the
    /// utterance after pre-roll silence) and turns force-cut overflows into
    /// partial-but-useful transcriptions instead of hard errors.
    ///
    /// Returns `(padded, effectiveLength)` because truncation also reduces
    /// the masked length the encoder uses.
    private func padMelToEnumeratedShape(
        mel: MLMultiArray, actualLength: Int
    ) throws -> (padded: MLMultiArray, length: Int) {
        let melFrames = mel.shape[2].intValue
        let maxSupported = supportedMelLengths.last ?? 0

        // Truncation path: input exceeds the encoder's max supported size.
        if melFrames > maxSupported {
            AudioLog.inference.notice(
                "Parakeet: mel \(melFrames) > max \(maxSupported); truncating to last \(maxSupported) frames")
            return try truncateMel(mel: mel, melFrames: melFrames, targetLength: maxSupported)
        }

        // Find the smallest supported length >= melFrames
        guard let targetLength = supportedMelLengths.first(where: { $0 >= melFrames }) else {
            // Should be unreachable given the truncation guard above, but
            // keep the typed error for any future shape config that breaks
            // the invariant.
            throw AudioModelError.inferenceFailed(
                operation: "mel padding",
                reason: "No supported shape >= \(melFrames) frames in \(supportedMelLengths)")
        }

        if targetLength == melFrames {
            return (mel, actualLength)
        }

        // Create zero-padded mel array [1, 128, targetLength]
        let padded = try MLMultiArray(
            shape: [1, 128, targetLength as NSNumber], dataType: mel.dataType)
        let srcPtr = mel.dataPointer.assumingMemoryBound(to: Float16.self)
        let dstPtr = padded.dataPointer.assumingMemoryBound(to: Float16.self)

        let numMelBins = config.numMelBins
        for bin in 0..<numMelBins {
            let srcOffset = bin * melFrames
            let dstOffset = bin * targetLength
            dstPtr.advanced(by: dstOffset)
                .update(from: srcPtr.advanced(by: srcOffset), count: melFrames)
            dstPtr.advanced(by: dstOffset + melFrames)
                .update(repeating: Float16(0), count: targetLength - melFrames)
        }

        return (padded, actualLength)
    }

    /// Drop the leading `melFrames - targetLength` frames and copy the tail
    /// into a `[1, 128, targetLength]` array. Effective length passed to the
    /// encoder is clamped to `targetLength` so the model's mask covers the
    /// whole tensor (no zero padding region to ignore).
    private func truncateMel(
        mel: MLMultiArray, melFrames: Int, targetLength: Int
    ) throws -> (padded: MLMultiArray, length: Int) {
        let dropPrefix = melFrames - targetLength
        let truncated = try MLMultiArray(
            shape: [1, 128, targetLength as NSNumber], dataType: mel.dataType)
        let srcPtr = mel.dataPointer.assumingMemoryBound(to: Float16.self)
        let dstPtr = truncated.dataPointer.assumingMemoryBound(to: Float16.self)
        let numMelBins = config.numMelBins
        for bin in 0..<numMelBins {
            let srcOffset = bin * melFrames + dropPrefix
            let dstOffset = bin * targetLength
            dstPtr.advanced(by: dstOffset)
                .update(from: srcPtr.advanced(by: srcOffset), count: targetLength)
        }
        return (truncated, targetLength)
    }

    private func runEncoder(mel: MLMultiArray, length: Int) throws -> MLFeatureProvider {
        let lengthArray = try MLMultiArray(shape: [1], dataType: .int32)
        lengthArray[0] = NSNumber(value: Int32(length))

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "mel": MLFeatureValue(multiArray: mel),
            "length": MLFeatureValue(multiArray: lengthArray),
        ])
        return try encoder!.prediction(from: input)
    }

    // MARK: - Model Loading

    /// Load a pretrained Parakeet model from HuggingFace.
    ///
    /// - Parameters:
    ///   - modelId: HuggingFace model identifier
    ///   - progressHandler: Optional callback for download/load progress `(fraction, status)`
    /// - Returns: Initialized model ready for transcription
    public static func fromPretrained(
        modelId: String? = nil,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> ParakeetASRModel {
        let effectiveModelId: String
        if let modelId {
            effectiveModelId = modelId
        } else {
            #if os(iOS)
            effectiveModelId = iosModelId
            #else
            effectiveModelId = defaultModelId
            #endif
        }

        AudioLog.modelLoading.info("Loading Parakeet model: \(effectiveModelId)")

        // Step 1: Get/create cache directory
        let resolvedCacheDir: URL
        do {
            resolvedCacheDir = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: effectiveModelId)
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: effectiveModelId, reason: "Failed to resolve cache directory", underlying: error)
        }

        // Step 2: Download model files (no preprocessor needed — mel is computed in Swift)
        progressHandler?(0.0, "Downloading model...")
        do {
            try await HuggingFaceDownloader.downloadWeights(
                modelId: effectiveModelId,
                to: resolvedCacheDir,
                additionalFiles: [
                    "encoder.mlmodelc/**",
                    "decoder.mlmodelc/**",
                    "joint.mlmodelc/**",
                    "vocab.json",
                    "config.json",
                ],
                offlineMode: offlineMode
            ) { fraction in
                progressHandler?(fraction * 0.7, "Downloading model...")
            }
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: effectiveModelId, reason: "Download failed", underlying: error)
        }

        // Step 3: Load config
        progressHandler?(0.70, "Loading configuration...")
        let config: ParakeetConfig
        let configURL = resolvedCacheDir.appendingPathComponent("config.json")
        if FileManager.default.fileExists(atPath: configURL.path) {
            let data = try Data(contentsOf: configURL)
            config = try JSONDecoder().decode(ParakeetConfig.self, from: data)
            AudioLog.modelLoading.debug("Loaded config from \(configURL.path)")
        } else {
            config = .default
            AudioLog.modelLoading.debug("Using default config")
        }

        // Step 4: Load vocabulary
        progressHandler?(0.75, "Loading vocabulary...")
        let vocabURL = resolvedCacheDir.appendingPathComponent("vocab.json")
        let vocabulary: ParakeetVocabulary
        do {
            vocabulary = try ParakeetVocabulary.load(from: vocabURL)
            AudioLog.modelLoading.debug("Loaded vocabulary: \(vocabulary.count) tokens")
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: effectiveModelId, reason: "Failed to load vocabulary", underlying: error)
        }

        // Step 5: Load CoreML models (encoder, decoder, joint — no preprocessor)
        progressHandler?(0.80, "Loading CoreML models...")
        // On real devices and macOS: `cpuAndGPU` for the encoder. ANE
        // compilation fails on some devices (e.g. iPhone 17 Pro
        // "Unknown aneSubType") causing memory spike + fallback anyway.
        // In the iOS Simulator the GPU/MPSGraph backend is unavailable
        // ("Espresso compiled without MPSGraph engine"); the silent
        // fallback returns 0 tokens for the INT8-quantized encoder, so
        // pin to `cpuOnly` which uses the same kernels as the macOS test
        // path and produces correct results.
        #if targetEnvironment(simulator)
        let computeUnits: MLComputeUnits = .cpuOnly
        #else
        let computeUnits: MLComputeUnits = .cpuAndGPU
        #endif
        let encoder = try loadCoreMLModel(
            name: "encoder", from: resolvedCacheDir, computeUnits: computeUnits)
        progressHandler?(0.90, "Loading decoder...")
        let decoder = try loadCoreMLModel(
            name: "decoder", from: resolvedCacheDir, computeUnits: computeUnits)
        progressHandler?(0.95, "Loading joint network...")
        let joint = try loadCoreMLModel(
            name: "joint", from: resolvedCacheDir, computeUnits: computeUnits)

        let supportedMelLengths = discoverSupportedMelLengths(from: encoder)
        AudioLog.modelLoading.info("Parakeet encoder accepts mel frame counts: \(supportedMelLengths)")

        progressHandler?(1.0, "Model loaded")
        AudioLog.modelLoading.info("Parakeet model loaded successfully")

        return ParakeetASRModel(
            config: config,
            encoder: encoder,
            decoder: decoder,
            joint: joint,
            vocabulary: vocabulary,
            supportedMelLengths: supportedMelLengths
        )
    }

    /// Read the encoder's mel input shape constraint and return the supported
    /// frame counts (dim 2 of `[batch, mel_bins, frames]`), sorted ascending.
    ///
    /// Handles both export styles produced by `models/parakeet-asr/export/convert.py`:
    /// - `--single-shape` → fixed shape, returns one element (e.g. `[500]` for iOS-5s)
    /// - default → `EnumeratedShapes`, returns the enumerated frame counts
    ///
    /// Falls back to `defaultMelLengths` if introspection fails — that matches
    /// the historical hardcoded list, so behaviour on previously-working exports
    /// is unchanged.
    private static func discoverSupportedMelLengths(from encoder: MLModel) -> [Int] {
        guard let melDescription = encoder.modelDescription.inputDescriptionsByName["mel"],
              let arrayConstraint = melDescription.multiArrayConstraint else {
            return defaultMelLengths
        }

        let shapeConstraint = arrayConstraint.shapeConstraint
        switch shapeConstraint.type {
        case .enumerated:
            let frames = shapeConstraint.enumeratedShapes
                .compactMap { $0.count >= 3 ? $0[2].intValue : nil }
                .sorted()
            return frames.isEmpty ? defaultMelLengths : frames
        case .unspecified:
            // Single fixed shape — the canonical shape on the constraint is the
            // only one the model accepts.
            let canonical = arrayConstraint.shape
            return canonical.count >= 3 ? [canonical[2].intValue] : defaultMelLengths
        case .range:
            // Range-on-time-axis exports: pick supported lengths that fall
            // inside the model's allowed range. Snap to the historical list
            // (intersected with the range) so we don't explode into thousands
            // of candidate paddings for permissive ranges.
            let perDim = shapeConstraint.sizeRangeForDimension
            guard perDim.count >= 3 else { return defaultMelLengths }
            let timeRange = perDim[2].rangeValue
            let lower = max(1, timeRange.location)
            let upper = lower + timeRange.length
            let candidates = defaultMelLengths.filter { $0 >= lower && $0 <= upper }
            return candidates.isEmpty ? [upper] : candidates
        @unknown default:
            return defaultMelLengths
        }
    }

    /// Load a compiled CoreML model from a `.mlmodelc` directory.
    private static func loadCoreMLModel(
        name: String,
        from directory: URL,
        computeUnits: MLComputeUnits
    ) throws -> MLModel {
        let modelURL = directory.appendingPathComponent("\(name).mlmodelc", isDirectory: true)

        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw AudioModelError.modelLoadFailed(
                modelId: name, reason: "CoreML model not found at \(modelURL.path)")
        }

        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = computeUnits

        do {
            let model = try MLModel(contentsOf: modelURL, configuration: mlConfig)
            AudioLog.modelLoading.debug("Loaded CoreML model: \(name) (compute: \(String(describing: computeUnits)))")
            return model
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: name, reason: "Failed to compile/load CoreML model", underlying: error)
        }
    }
}
