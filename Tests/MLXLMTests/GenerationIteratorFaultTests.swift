// Copyright © 2026 Apple Inc.

import MLX
import MLXNN
import Testing

@_spi(Testing) @testable import MLXLMCommon

/// MTP and raw-stream faults that the qualified helper tests do not reach.
/// Fault graphs are the CPU float64 Hadamard. Cache length is not assumed.
struct GenerationIteratorFaultTests {
    private let prompt = LMInput(tokens: MLXArray([Int32(1), Int32(2), Int32(3)]))
    private let tokenID: Int32 = 5
    /// Last verify-row logit. It is inside the probe vocabulary and is not the draft token.
    private let verifierBonus: Int32 = 7
    private let parameters = GenerateParameters(maxTokens: 8, temperature: 0)

    @Test("MTP drafter eval failure is terminal and a healthy twin yields")
    func mtpDrafterFlatFaultAndHealthyTwin() throws {
        let model = MTPFaultProbeModel(tokenID: tokenID, verifierBonus: verifierBonus)
        let drafter = ProbeDrafter(drafts: .flatEvalFault)
        var iterator = try makeIterator(model: model, drafter: drafter)

        let bonus = iterator.next()
        #expect(bonus == Int(tokenID))
        #expect(iterator.tokenCount == 1)
        #expect(iterator.generationFault == nil)
        #expect(model.forwardLengths == [3])
        #expect(drafter.callCount == 0)

        // The yielded bonus is the next round's input and is not in the cache yet.
        // A failed round has to land back on this prompt prefix.
        let preRound = try capturePreRoundMTPBoundary(
            iterator, committedPrefix: prompt.text.tokens.size)

        #expect(iterator.next() == nil)
        let mainForwards = model.forwardCount
        let draftCalls = drafter.callCount
        #expect(mainForwards == 2)
        #expect(model.forwardRanks == [2, 2])
        #expect(model.forwardLengths == [3, 4])
        #expect(draftCalls == 1)
        #expect(drafter.requestedBlockSizes == [4])
        #expect(drafter.hiddenRanks == [3])
        #expect(drafter.sharedKVRanks == [4])
        #expect(iterator.tokenCount == 1)
        #expect(iterator.proposedCount == 0)
        #expect(iterator.acceptedCount == 0)
        let stored = try #require(iterator.generationFault as? MLXError)
        expectUnsupportedHadamard(stored)
        try expectRankCorrectMTPState(
            iterator, model: model, preRound: preRound, attemptedForwardPositions: 4)

        let offsetAtFault = iterator.mainCache.first?.offset
        #expect(offsetAtFault == preRound.offset as Int?)
        #expect(iterator.mainCacheStorage.processedTokenCount == preRound.timeline)
        #expect(iterator.mainCacheStorage.roundIsOpen == false)
        #expect(iterator.next() == nil)
        #expect(iterator.next() == nil)
        #expect(model.forwardCount == mainForwards)
        #expect(drafter.callCount == draftCalls)
        #expect(iterator.tokenCount == 1)
        #expect(iterator.mainCache.first?.offset == offsetAtFault)

        iterator.finalizeGeneration()
        #expect(iterator.next() == nil)
        #expect(model.forwardCount == mainForwards)
        #expect(drafter.callCount == draftCalls)
        #expect(iterator.tokenCount == 1)
        #expect(iterator.mainCache.first?.offset == preRound.offset as Int?)
        #expect(iterator.mainCacheStorage.processedTokenCount == preRound.timeline)
        #expect(iterator.mainCacheStorage.roundIsOpen == false)
        try expectRankCorrectMTPState(
            iterator, model: model, preRound: preRound, attemptedForwardPositions: 4)
        let storedAfterFinalize = try #require(iterator.generationFault as? MLXError)
        expectUnsupportedHadamard(storedAfterFinalize)

        let healthyModel = MTPFaultProbeModel(tokenID: tokenID, verifierBonus: verifierBonus)
        let healthyDrafter = ProbeDrafter(drafts: .fixed(tokenID))
        var healthy = try makeIterator(model: healthyModel, drafter: healthyDrafter)
        var yielded: [Int] = []
        for _ in 0 ..< 5 {
            if let token = healthy.next() {
                yielded.append(token)
            }
        }
        // Prefill bonus, three accepted drafts, then the planned verifier bonus.
        #expect(
            yielded == [Int(tokenID), Int(tokenID), Int(tokenID), Int(tokenID), Int(verifierBonus)]
        )
        #expect(healthy.generationFault == nil)
        #expect(healthy.tokenCount == 5)
        #expect(healthyDrafter.callCount == 1)
        #expect(healthyModel.forwardLengths == [3, 4])
    }

    @Test("A later MTP verifier fault drops pending tokens and keeps yielded output")
    func mtpLaterVerifierFaultClearsPending() throws {
        let model = MTPFaultProbeModel(tokenID: tokenID, verifierBonus: verifierBonus)
        let drafter = ProbeDrafter(drafts: .fixed(tokenID))
        var iterator = try makeIterator(model: model, drafter: drafter)
        let processor = RowFaultProcessor(faultAt: 1)

        let bonus = iterator.next()
        iterator._setProcessorForTesting(processor)
        #expect(bonus == Int(tokenID))
        #expect(iterator.tokenCount == 1)
        #expect(model.forwardCount == 1)
        #expect(drafter.callCount == 0)
        #expect(processor.processCount == 0)

        // The yielded bonus is the next round's input and is not in the cache yet.
        // A failed round has to land back on this prompt prefix.
        let preRound = try capturePreRoundMTPBoundary(
            iterator, committedPrefix: prompt.text.tokens.size)

        #expect(iterator.next() == nil)
        let mainForwards = model.forwardCount
        let draftCalls = drafter.callCount
        #expect(mainForwards == 2)
        #expect(model.forwardRanks == [2, 2])
        #expect(model.forwardLengths == [3, 4])
        #expect(draftCalls == 1)
        #expect(processor.processCount == 2)
        #expect(processor.sampledCount == 1)
        #expect(iterator.tokenCount == 1)
        #expect(iterator.proposedCount == 0)
        #expect(iterator.acceptedCount == 0)
        let stored = try #require(iterator.generationFault as? MLXError)
        expectUnsupportedHadamard(stored)
        try expectRankCorrectMTPState(
            iterator, model: model, preRound: preRound, attemptedForwardPositions: 4)

        let offsetAtFault = iterator.mainCache.first?.offset
        #expect(offsetAtFault == preRound.offset as Int?)
        #expect(iterator.mainCacheStorage.processedTokenCount == preRound.timeline)
        #expect(iterator.mainCacheStorage.roundIsOpen == false)
        #expect(iterator.next() == nil)
        #expect(iterator.next() == nil)
        #expect(model.forwardCount == mainForwards)
        #expect(drafter.callCount == draftCalls)
        #expect(processor.processCount == 2)
        #expect(iterator.tokenCount == 1)
        #expect(iterator.mainCache.first?.offset == offsetAtFault)

        iterator.finalizeGeneration()
        #expect(iterator.next() == nil)
        #expect(model.forwardCount == mainForwards)
        #expect(drafter.callCount == draftCalls)
        #expect(processor.processCount == 2)
        #expect(processor.sampledCount == 1)
        #expect(iterator.tokenCount == 1)
        #expect(iterator.mainCache.first?.offset == preRound.offset as Int?)
        #expect(iterator.mainCacheStorage.processedTokenCount == preRound.timeline)
        #expect(iterator.mainCacheStorage.roundIsOpen == false)
        try expectRankCorrectMTPState(
            iterator, model: model, preRound: preRound, attemptedForwardPositions: 4)
        let storedAfterFinalize = try #require(iterator.generationFault as? MLXError)
        expectUnsupportedHadamard(storedAfterFinalize)
    }

    @Test("generateTokenTask cancels a fault at the token budget")
    func generateTokenTaskFaultCancelsAtMatchingBudget() async throws {
        let faultModel = MTPFaultProbeModel(tokenID: tokenID, verifierBonus: verifierBonus)
        var faultIterator = try TokenIterator(
            input: prompt, model: faultModel,
            parameters: GenerateParameters(maxTokens: 1, temperature: 0))
        eval(faultIterator.y.tokens)
        #expect(faultIterator.y.tokens.shape == [1])
        #expect(faultIterator.y.tokens.item(Int.self) == Int(tokenID))
        #expect(faultIterator.generationFault == nil)
        let replacement = int32TokenShapedFault()
        #expect(replacement.shape == [1])
        #expect(replacement.dtype == .int32)
        faultIterator.y = .init(tokens: replacement)

        let promptCount = prompt.text.tokens.size
        let faultEvents = await tokenEvents(iterator: faultIterator, promptCount: promptCount)
        let faultInfo = try #require(faultEvents.last?.info)
        #expect(faultEvents.count == 1)
        #expect(faultEvents.dropLast().isEmpty)
        #expect(faultInfo.stopReason == .cancelled)
        #expect(faultInfo.generationTokenCount == 0)
        #expect(faultModel.forwardCount == 2)
        #expect(faultModel.forwardLengths == [3, 1])
        #expect(faultModel.forwardRanks == [2, 2])

        let healthyModel = MTPFaultProbeModel(tokenID: tokenID, verifierBonus: verifierBonus)
        let healthyIterator = try TokenIterator(
            input: prompt, model: healthyModel,
            parameters: GenerateParameters(maxTokens: 1, temperature: 0))
        let healthyEvents = await tokenEvents(iterator: healthyIterator, promptCount: promptCount)
        let healthyInfo = try #require(healthyEvents.last?.info)
        #expect(healthyEvents.compactMap(\.token) == [Int(tokenID)])
        #expect(healthyEvents.compactMap(\.info).count == 1)
        #expect(healthyEvents.dropLast().allSatisfy { $0.token != nil })
        #expect(healthyInfo.stopReason == .length)
        #expect(healthyInfo.generationTokenCount == 1)
        #expect(healthyModel.forwardCount == 2)
        #expect(healthyModel.forwardLengths == [3, 1])
    }

    private func makeIterator(
        model: MTPFaultProbeModel, drafter: ProbeDrafter
    ) throws -> MTPSpeculativeTokenIterator {
        try MTPSpeculativeTokenIterator(
            input: prompt, mainModel: model, drafter: drafter,
            parameters: parameters, blockSize: 4)
    }
}

/// Closed, aligned cache metadata from before a failing MTP round.
private struct MTPPreRoundBoundary {
    let offset: Int
    let timeline: Int
    let hiddenShape: [Int]
    let sharedKeyShape: [Int]
    let sharedValueShape: [Int]
    let sharedSources: [String: Int]
    let sharedOffsets: [String: Int]
    let cacheKeyShape: [Int]
    let cacheValueShape: [Int]
}

/// The yielded bonus is outside this prefix. Sequence axes match that prefix, and the round
/// is closed.
private func capturePreRoundMTPBoundary(
    _ iterator: MTPSpeculativeTokenIterator, committedPrefix: Int
) throws -> MTPPreRoundBoundary {
    #expect(iterator.mainCacheStorage.roundIsOpen == false)
    let offset = try #require(iterator.mainCache.first?.offset)
    #expect(offset == committedPrefix)
    #expect(iterator.mainCacheStorage.processedTokenCount == committedPrefix)
    #expect(iterator.mainCacheStorage.nativeAttentionOffsetsAreAligned)

    let hidden = try #require(iterator.mainState?[mtpLastHiddenStatesKey])
    #expect(hidden.ndim == 3)
    #expect(hidden.dim(0) == 1)
    #expect(hidden.dim(1) == committedPrefix)
    #expect(hidden.dim(-1) == MTPFaultProbeModel.headDim)

    let shared = try #require(iterator.mainState?[mtpSharedKVStatesKey]?["full_attention"])
    #expect(shared.0.ndim == 4)
    #expect(shared.1.ndim == 4)
    #expect(shared.0.dim(0) == 1)
    #expect(shared.0.dim(1) == 1)
    #expect(shared.0.dim(-2) == committedPrefix)
    #expect(shared.1.dim(-2) == committedPrefix)
    #expect(shared.0.dim(-1) == MTPFaultProbeModel.headDim)
    #expect(shared.1.dim(-1) == MTPFaultProbeModel.headDim)
    let sources = try #require(iterator.mainState?[mtpSharedKVSourceIndicesKey])
    #expect(sources["full_attention"] == 0)
    let sharedOffsets = try #require(iterator.mainState?[mtpSharedKVOffsetsKey])
    #expect(sharedOffsets["full_attention"] == committedPrefix)

    let cacheState = try #require(iterator.mainCache.first?.state)
    #expect(cacheState.count == 2)
    #expect(cacheState.allSatisfy { $0.ndim == 4 })
    #expect(cacheState[0].dim(-2) == committedPrefix)
    #expect(cacheState[1].dim(-2) == committedPrefix)
    #expect(shared.0.dim(-2) == cacheState[0].dim(-2))
    #expect(shared.1.dim(-2) == cacheState[1].dim(-2))

    return MTPPreRoundBoundary(
        offset: offset,
        timeline: iterator.mainCacheStorage.processedTokenCount,
        hiddenShape: hidden.shape,
        sharedKeyShape: shared.0.shape,
        sharedValueShape: shared.1.shape,
        sharedSources: sources,
        sharedOffsets: sharedOffsets,
        cacheKeyShape: cacheState[0].shape,
        cacheValueShape: cacheState[1].shape)
}

/// After an unsuccessful staged round, hidden state, shared K/V, and the cache match the
/// boundary captured before that round. Ranks stay absolute. `attemptedForwardPositions` is
/// the width the failed forward ran (4 for these probes); the restored hidden length is the
/// pre-round one.
private func expectRankCorrectMTPState(
    _ iterator: MTPSpeculativeTokenIterator,
    model: MTPFaultProbeModel,
    preRound: MTPPreRoundBoundary,
    attemptedForwardPositions: Int
) throws {
    let attempted = try #require(model.forwardLengths.last)
    #expect(attempted == attemptedForwardPositions)

    #expect(iterator.mainCacheStorage.roundIsOpen == false)
    #expect(iterator.mainCache.first?.offset == preRound.offset as Int?)
    #expect(iterator.mainCacheStorage.processedTokenCount == preRound.timeline)
    #expect(iterator.mainCacheStorage.nativeAttentionOffsetsAreAligned)

    let hidden = try #require(iterator.mainState?[mtpLastHiddenStatesKey])
    #expect(hidden.ndim == 3)
    #expect(hidden.dim(0) == 1)
    #expect(hidden.dim(1) == preRound.hiddenShape[1])
    #expect(hidden.dim(-1) == MTPFaultProbeModel.headDim)
    #expect(hidden.shape == preRound.hiddenShape)

    let shared = try #require(iterator.mainState?[mtpSharedKVStatesKey]?["full_attention"])
    #expect(shared.0.ndim == 4)
    #expect(shared.1.ndim == 4)
    #expect(shared.0.dim(0) == 1)
    #expect(shared.0.dim(1) == 1)
    #expect(shared.0.dim(-1) == MTPFaultProbeModel.headDim)
    #expect(shared.0.shape == preRound.sharedKeyShape)
    #expect(shared.1.shape == preRound.sharedValueShape)
    let sources = try #require(iterator.mainState?[mtpSharedKVSourceIndicesKey])
    #expect(sources["full_attention"] == 0)
    #expect(sources == preRound.sharedSources)
    let sharedOffsets = try #require(iterator.mainState?[mtpSharedKVOffsetsKey])
    #expect(sharedOffsets == preRound.sharedOffsets)

    let cacheState = try #require(iterator.mainCache.first?.state)
    #expect(cacheState.count == 2)
    #expect(cacheState.allSatisfy { $0.ndim == 4 })
    #expect(cacheState[0].shape == preRound.cacheKeyShape)
    #expect(cacheState[1].shape == preRound.cacheValueShape)
    #expect(shared.0.dim(-2) == cacheState[0].dim(-2))
    #expect(shared.1.dim(-2) == cacheState[1].dim(-2))
}

/// MLX appends a source location after the stable Hadamard marker.
private func expectUnsupportedHadamard(_ error: MLXError) {
    switch error {
    case .caught(let message):
        #expect(message.contains("[hadamard] Unsupported type."))
    }
}

private func tokenEvents(
    iterator: consuming TokenIterator, promptCount: Int
) async -> [TokenGeneration] {
    let (stream, task) = generateTokenTask(
        promptTokenCount: promptCount,
        modelConfiguration: .init(id: "iterator-fault"),
        tokenizer: TestTokenizer(),
        iterator: iterator)
    var events: [TokenGeneration] = []
    for await event in stream {
        events.append(event)
    }
    await task.value
    return events
}

/// CPU float64 Hadamard. The graph stays lazy until eval, which fails the dtype check.
private func cpuFloat64Hadamard() -> MLXArray {
    let source = MLXArray([Float(1), Float(2)])
    return hadamardTransform(source.asType(.float64, stream: .cpu), stream: .cpu)
}

/// Int32 slice whose shape matches a sampled token. Eval still reaches the Hadamard.
private func int32TokenShapedFault() -> MLXArray {
    cpuFloat64Hadamard().asType(.int32, stream: .cpu)[0 ..< 1, stream: .cpu]
}

/// Lazy float32 logits whose eval still reaches the unsupported Hadamard.
/// `count` is a power of two so the failure is the dtype, and no element is read.
private func unsupportedHadamardLogits(positions: Int, vocabulary: Int) -> MLXArray {
    let count = positions * vocabulary
    let source = MLXArray.zeros([count], dtype: .float32, stream: .cpu)
    let transformed = hadamardTransform(source.asType(.float64, stream: .cpu), stream: .cpu)
    return transformed.asType(.float32, stream: .cpu)
        .reshaped([1, positions, vocabulary], stream: .cpu)
}

/// One fault element, repeated to the drafter's `[batch, count]` token shape.
private func flatDraftFault(batch: Int, count: Int) -> MLXArray {
    let one = int32TokenShapedFault().reshaped([1, 1], stream: .cpu)
    return tiled(one, repetitions: [batch, count], stream: .cpu)
}

/// `process` call `faultAt` returns the CPU fault. Earlier calls return the row.
private final class RowFaultProcessor: LogitProcessor {
    let faultAt: Int
    private(set) var processCount = 0
    private(set) var sampledCount = 0

    init(faultAt: Int) {
        self.faultAt = faultAt
    }

    func prompt(_ prompt: MLXArray) {}

    func process(logits: MLXArray) -> MLXArray {
        let row = processCount
        processCount += 1
        guard row == faultAt else { return logits }
        // Eight logits keeps the Hadamard length a power of two. The probe vocabulary is wider.
        return unsupportedHadamardLogits(positions: 1, vocabulary: 8)
    }

    func didSample(token: MLXArray) {
        sampledCount += 1
    }

    func copy() -> Self {
        let copy = RowFaultProcessor(faultAt: faultAt)
        copy.processCount = processCount
        copy.sampledCount = sampledCount
        return copy as! Self
    }
}

/// Fixed drafts, or a lazy int32 grid whose eval fails in `flatDraftTokens`.
private final class ProbeDrafter: Module, MTPDrafterModel {
    enum Drafts {
        case fixed(Int32)
        case flatEvalFault
    }

    let drafts: Drafts
    private(set) var callCount = 0
    private(set) var requestedBlockSizes: [Int] = []
    private(set) var hiddenRanks: [Int] = []
    private(set) var sharedKVRanks: [Int] = []

    init(drafts: Drafts) {
        self.drafts = drafts
        super.init()
    }

    func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionDeltas: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        sampler: any LogitSampler
    ) -> MLXArray {
        callCount += 1
        requestedBlockSizes.append(blockSize)
        hiddenRanks.append(lastHidden.ndim)
        sharedKVRanks.append(sharedKV["full_attention"]?.0.ndim ?? -1)
        let batch = lastToken.dim(0)
        let count = blockSize - 1
        switch drafts {
        case .fixed(let value):
            let values = Array(repeating: value, count: batch * count)
            return MLXArray(values, [batch, count])
        case .flatEvalFault:
            return flatDraftFault(batch: batch, count: count)
        }
    }
}

/// Counting target. Forwards record rank and length only.
/// Constant logits make argmax deterministic. MTP state uses the live cache
/// offset as the shared-K/V span and does not read caller tokens.
private final class MTPFaultProbeModel: Module, LanguageModel {
    static let vocabulary = 20
    static let headDim = 4

    let tokenID: Int32
    private(set) var forwardCount = 0
    private(set) var forwardRanks: [Int] = []
    private(set) var forwardLengths: [Int] = []
    private var plan: [Int32]
    private var planIndex = 0

    init(tokenID: Int32, verifierBonus: Int32) {
        self.tokenID = tokenID
        self.plan = [0, 0, tokenID, tokenID, tokenID, tokenID, verifierBonus]
        super.init()
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        forwardCount += 1
        let tokens = input.tokens
        let rank = tokens.ndim
        let positions = tokens.dim(-1)
        let batch = rank >= 2 ? tokens.dim(0) : 1
        forwardRanks.append(rank)
        forwardLengths.append(positions)

        if let cache {
            for entry in cache {
                let keys = MLXArray.zeros([batch, 1, positions, Self.headDim], dtype: .float32)
                _ = entry.update(keys: keys, values: keys)
            }
        }

        let logits = constantLogits(batch: batch, positions: positions)
        guard state?[mtpEmitFlagKey] ?? false else {
            return LMOutput(logits: logits)
        }

        let span = cache?.first?.offset ?? positions
        var out = LMOutput.State()
        out[mtpLastHiddenStatesKey] = MLXArray.zeros([batch, positions, Self.headDim])
        out[mtpSharedKVSourceIndicesKey] = ["full_attention": 0]
        out[mtpSharedKVStatesKey] = [
            "full_attention": (
                MLXArray.zeros([batch, 1, span, Self.headDim]),
                MLXArray.zeros([batch, 1, span, Self.headDim])
            ),
        ]
        out[mtpSharedKVOffsetsKey] = ["full_attention": span]
        return LMOutput(logits: logits, state: out)
    }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        [KVCacheSimple()]
    }

    private func constantLogits(batch: Int, positions: Int) -> MLXArray {
        var data = [Float](
            repeating: 0, count: batch * positions * Self.vocabulary)
        for index in 0 ..< (batch * positions) {
            let planned =
                planIndex + index < plan.count ? Int(plan[planIndex + index]) : Int(tokenID)
            data[index * Self.vocabulary + planned] = 100
        }
        planIndex += positions
        return MLXArray(data, [batch, positions, Self.vocabulary])
    }
}
