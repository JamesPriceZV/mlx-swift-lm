// Copyright © 2026 Apple Inc.

import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

/// Eval faults stay typed as `MLXError` and end generation.
/// A fault is `.cancelled` even when the token count has reached the budget.
struct GenerationFaultTests {
    private struct SentinelFault: Error, Equatable {
        let marker: String
    }

    @Test("A successful step leaves no fault")
    func successfulStepLeavesNoFault() {
        var fault: (any Error)?
        let value = capturingGenerationFault(&fault) { 42 }
        #expect(value == 42)
        #expect(fault == nil)
    }

    @Test("A thrown compute fault is stored and ends the step")
    func thrownFaultIsStoredAndStops() {
        var fault: (any Error)?
        var calls = 0
        let failed = capturingGenerationFault(&fault) { () -> Int in
            calls += 1
            throw SentinelFault(marker: "metal")
        }
        let blocked = capturingGenerationFault(&fault) { () -> Int in
            calls += 1
            return 7
        }

        #expect(failed == nil)
        #expect(blocked == nil)
        #expect(calls == 1)
        #expect((fault as? SentinelFault) == SentinelFault(marker: "metal"))
    }

    @Test("A fault is not reported as a clean length stop")
    func faultIsNotALengthStop() {
        let fault = SentinelFault(marker: "metal")
        #expect(
            stopReasonForIteratorEnd(
                taskCancelled: false,
                generationFault: fault,
                tokenCount: 8,
                maxTokens: 8
            ) == .cancelled
        )
        #expect(
            stopReasonForIteratorEnd(
                taskCancelled: false,
                generationFault: nil,
                tokenCount: 8,
                maxTokens: 8
            ) == .length
        )
        #expect(
            stopReasonForIteratorEnd(
                taskCancelled: false,
                generationFault: nil,
                tokenCount: 3,
                maxTokens: 8
            ) == .cancelled
        )
        #expect(
            stopReasonForIteratorEnd(
                taskCancelled: true,
                generationFault: nil,
                tokenCount: 8,
                maxTokens: 8
            ) == .cancelled
        )
    }

    @Test("Settled reads return synthetic token ids")
    func settledReadsReturnSyntheticTokenIDs() throws {
        // Requires the MLX GPU runtime. Not executed by this generator.
        let scalar = MLXArray([Int32(42)])
        let scalarID = try readAfterSettledEval({ eval(scalar) }) {
            scalar.item(Int.self)
        }
        #expect(scalarID == 42)

        let vector = MLXArray([Int32(7), Int32(9)])
        let vectorIDs = try readAfterSettledEval({ eval(vector) }) {
            vector.asArray(Int.self)
        }
        #expect(vectorIDs == [7, 9])

        let first = MLXArray([Int32(1)])
        let second = MLXArray([Int32(1), Int32(2)])
        let pair = try readAfterSettledEval({ eval(first, second) }) {
            (first.asArray(Int.self), second.asArray(Int.self))
        }
        #expect(pair.0 == [1])
        #expect(pair.1 == [1, 2])
    }

    @Test("A failed CPU Hadamard eval throws and does not read")
    func failedCPUHadamardEvalDoesNotRead() throws {
        let bad = cpuFloat64Hadamard()
        var failedReads = 0
        let thrown = try #require(
            #expect(throws: MLXError.self) {
                try readAfterSettledEval({ eval(bad) }) {
                    failedReads += 1
                    return failedReads
                }
            }
        )
        #expect(failedReads == 0)
        expectUnsupportedHadamard(thrown)

        let healthy = hadamardTransform(MLXArray([Float(1), Float(2)]), stream: .cpu)
        var healthyReads = 0
        let read = try readAfterSettledEval({ eval(healthy) }) {
            healthyReads += 1
            return healthyReads
        }
        #expect(read == 1)
        #expect(healthyReads == 1)
    }

    @Test("TokenIterator stores a terminal MLX fault and a healthy twin yields its token")
    func tokenIteratorTerminalFaultAndHealthyTwin() throws {
        let prompt = LMInput(tokens: MLXArray([Int32(1), Int32(2), Int32(3)]))
        let parameters = GenerateParameters(maxTokens: 8, temperature: 0)
        let tokenID = 5

        let model = MetadataCountingLanguageModel(tokenID: tokenID)
        var iterator = try TokenIterator(input: prompt, model: model, parameters: parameters)
        eval(iterator.y.tokens)
        #expect(iterator.generationFault == nil)
        #expect(iterator.tokenCount == 0)
        #expect(iterator.y.tokens.shape == [1])
        #expect(iterator.y.tokens.item(Int.self) == tokenID)
        #expect(model.forwardCount == 1)

        let replacement = int32TokenShapedFault()
        #expect(replacement.shape == [1])
        #expect(replacement.dtype == .int32)
        iterator.y = .init(tokens: replacement)

        #expect(iterator.next() == nil)
        let forwardsAtFailure = model.forwardCount
        let tokensAtFailure = iterator.tokenCount
        #expect(forwardsAtFailure == 2)
        #expect(model.forwardRanks == [2, 2])
        #expect(model.forwardLengths == [3, 1])
        #expect(iterator.cache.first?.offset == 4)
        let stored = try #require(iterator.generationFault as? MLXError)
        expectUnsupportedHadamard(stored)

        #expect(iterator.next() == nil)
        #expect(iterator.next() == nil)
        #expect(model.forwardCount == forwardsAtFailure)
        #expect(iterator.tokenCount == tokensAtFailure)
        #expect(iterator.cache.first?.offset == 4)

        let healthyModel = MetadataCountingLanguageModel(tokenID: tokenID)
        var healthy = try TokenIterator(input: prompt, model: healthyModel, parameters: parameters)
        eval(healthy.y.tokens)
        #expect(healthy.y.tokens.item(Int.self) == tokenID)
        #expect(healthy.next() == tokenID)
        #expect(healthy.generationFault == nil)
        #expect(healthy.tokenCount == 1)
        #expect(healthyModel.forwardLengths == [3, 1])
        #expect(healthy.cache.first?.offset == 4)
    }

    @Test("SpeculativeTokenIterator faults on verifier eval and a healthy twin yields tokens")
    func speculativeVerifierFaultAndHealthyTwin() throws {
        // Prompt 3 plus one draft is a verify length of 4. With vocab 8 the
        // Hadamard input has 32 elements, a power of two, so eval fails the dtype check.
        let prompt = LMInput(tokens: MLXArray([Int32(1), Int32(2), Int32(3)]))
        let parameters = GenerateParameters(maxTokens: 8, temperature: 0)
        let tokenID = 5
        let numDraftTokens = 1

        let main = MetadataCountingLanguageModel(tokenID: tokenID, logits: .unsupportedHadamard)
        let draft = MetadataCountingLanguageModel(tokenID: tokenID)
        var iterator = try SpeculativeTokenIterator(
            input: prompt, mainModel: main, draftModel: draft,
            parameters: parameters, numDraftTokens: numDraftTokens)
        #expect(iterator.generationFault == nil)
        #expect(iterator.tokenCount == 0)
        #expect(main.forwardCount == 0)
        #expect(draft.forwardCount == 0)

        #expect(iterator.next() == nil)
        let mainForwardsAtFailure = main.forwardCount
        let draftForwardsAtFailure = draft.forwardCount
        #expect(mainForwardsAtFailure == 1)
        #expect(draftForwardsAtFailure == 1)
        #expect(main.forwardLengths == [4])
        #expect(draft.forwardLengths == [3])
        #expect(iterator.tokenCount == 0)
        let stored = try #require(iterator.generationFault as? MLXError)
        expectUnsupportedHadamard(stored)

        #expect(iterator.next() == nil)
        #expect(iterator.next() == nil)
        #expect(main.forwardCount == mainForwardsAtFailure)
        #expect(draft.forwardCount == draftForwardsAtFailure)
        #expect(iterator.tokenCount == 0)

        let healthyMain = MetadataCountingLanguageModel(tokenID: tokenID)
        let healthyDraft = MetadataCountingLanguageModel(tokenID: tokenID)
        var healthy = try SpeculativeTokenIterator(
            input: prompt, mainModel: healthyMain, draftModel: healthyDraft,
            parameters: parameters, numDraftTokens: numDraftTokens)
        #expect(healthy.next() == tokenID)
        #expect(healthy.next() == tokenID)
        #expect(healthy.generationFault == nil)
        #expect(healthy.tokenCount == 2)
    }
}

/// MLX appends a source location after the stable Hadamard marker.
private func expectUnsupportedHadamard(_ error: MLXError) {
    switch error {
    case .caught(let message):
        #expect(message.contains("[hadamard] Unsupported type."))
    }
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

/// Counting model. Forwards record rank and length only, then write a real
/// `[batch, heads, sequence, headDim]` cache. Constant logits make argmax
/// deterministic. The Hadamard mode returns a lazy logit graph and does not
/// read that graph or the input buffer.
private final class MetadataCountingLanguageModel: Module, LanguageModel {
    enum Logits {
        case constant
        case unsupportedHadamard
    }

    let tokenID: Int
    let vocabulary = 8
    let logits: Logits
    private(set) var forwardCount = 0
    private(set) var forwardRanks: [Int] = []
    private(set) var forwardLengths: [Int] = []

    init(tokenID: Int, logits: Logits = .constant) {
        self.tokenID = tokenID
        self.logits = logits
        super.init()
    }

    func prepare(
        _ input: LMInput, cache _: [KVCache], state _: LMOutput.State?, prefill _: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        forwardCount += 1
        let rank = inputs.ndim
        let positions = inputs.dim(-1)
        let batch = rank >= 2 ? inputs.dim(0) : 1
        forwardRanks.append(rank)
        forwardLengths.append(positions)

        if let cache {
            for entry in cache {
                let keys = MLXArray.zeros([batch, 1, positions, 4], dtype: .float32)
                _ = entry.update(keys: keys, values: keys)
            }
        }
        switch logits {
        case .constant:
            return constantLogits(batch: batch, positions: positions)
        case .unsupportedHadamard:
            return unsupportedHadamardLogits(positions: positions, vocabulary: vocabulary)
        }
    }

    func newCache(parameters _: GenerateParameters?) throws -> [KVCache] {
        [KVCacheSimple()]
    }

    private func constantLogits(batch: Int, positions: Int) -> MLXArray {
        var data = [Float](repeating: 0, count: batch * positions * vocabulary)
        for index in 0 ..< (batch * positions) {
            data[index * vocabulary + tokenID] = 100
        }
        return MLXArray(data, [batch, positions, vocabulary])
    }
}
