/**
 * @fileoverview Shared text generation utilities for different model types
 */

import type { Device, Tensor } from '@piston-ml/piston-web';

import { int32, tensor } from '@piston-ml/piston-web';

import type { GeneratableModel } from './types';

import { createEmptyDecoderKVCache, type DecoderKVCache } from './model/cache';
import { DecoderTransformer, EncoderDecoderTransformer } from './model/transformer';

export interface GenerationConfig {
	maxTokens?: number;
	stopTokens?: number | number[];
	device?: string | Device;
	startToken?: number;
	maxTargetLength?: number;
	temperature?: number;
	useKvCache?: boolean;
}

export interface GenerationResult {
	sequences: number[][];
	// Raw probabilities (post-softmax) for generated tokens
	probs?: Tensor;
	// Running average throughput since start of generation (tokens/second)
	tokensPerSecond?: number;
}

function normalizeStopTokens(stopTokens: number | number[] = []): Set<number> {
	return new Set(Array.isArray(stopTokens) ? stopTokens : [stopTokens]);
}

/**
 * Create tensor from token sequences with proper device and dtype
 */
function createInputTensor(sequences: number[][], device: Device): Tensor {
	return tensor(sequences, { device, dtype: int32 });
}

/**
 * Convert tensor output to array of token IDs
 */
async function tensorToTokens(tokenTensor: Tensor): Promise<Int32Array> {
	return (await (await tokenTensor.to('cpu')).toVec()) as Int32Array;
}

/**
 * Check if any sequence should continue generating (hasn't hit stop tokens)
 */
function shouldContinueGeneration(
	results: number[][],
	newTokens: Int32Array,
	stopTokenSet: Set<number>
): boolean {
	let shouldContinue = false;
	for (let i = 0; i < results.length; i++) {
		// If this sequence already ended with a stop token, do not append anything further
		const sequence = results[i];
		const lastToken = sequence.length > 0 ? sequence[sequence.length - 1] : undefined;
		const alreadyStopped = lastToken !== undefined && stopTokenSet.has(lastToken);

		if (alreadyStopped) {
			continue;
		}

		const token = newTokens[i];
		// Always append the newly generated token
		sequence.push(token);
		// Continue only if we did not just append a stop token
		if (!stopTokenSet.has(token)) {
			shouldContinue = true;
		}
	}
	return shouldContinue;
}

/**
 * Create a simple running tokens/sec tracker using the Performance API
 */
function startTokensPerSecondTracker(): (tokensProducedThisStep: number) => number {
	const now =
		typeof performance !== 'undefined' && typeof performance.now === 'function'
			? () => performance.now()
			: () => Date.now();
	const startMs = now();
	let totalTokens = 0;
	return (tokensProducedThisStep: number) => {
		if (tokensProducedThisStep > 0) totalTokens += tokensProducedThisStep;
		const elapsedMs = Math.max(1, now() - startMs);
		return totalTokens / (elapsedMs / 1000);
	};
}

/**
 * Generate tokens for GPT (decoder-only) model
 */
export async function* generateGPTStream(
	model: DecoderTransformer,
	input: number[] | number[][],
	config: GenerationConfig = {}
): AsyncGenerator<GenerationResult, void, unknown> {
	const stopTokenSet = normalizeStopTokens(config.stopTokens);
	const isBatch = Array.isArray(input[0]);
	const sequences = isBatch ? (input as number[][]) : [input as number[]];

	// Initialize results with copies of input sequences
	const results = sequences.map((seq) => [...seq]);
	const device = model.lmHead.weight.device;

	const kvCache: DecoderKVCache | null =
		(config.useKvCache ?? true) ? createEmptyDecoderKVCache(model.config.nLayers) : null;

	let seeded = false;
	let step = 0;
	const getTokensPerSecond = startTokensPerSecondTracker();
	while (true) {
		// Seed cache once with full sequences, then switch to single-token steps
		const prevLengths = results.map((seq) => seq.length);
		let inputTensor: Tensor;
		if (kvCache && seeded) {
			const latestTokens = results.map((seq) => [seq[seq.length - 1] ?? 0]);
			inputTensor = createInputTensor(latestTokens, device);
		} else {
			const { padded } = rightPadSequences(results);
			inputTensor = createInputTensor(padded, device);
		}

		// Forward pass to get logits
		const [logits, _] = model.forward(inputTensor, { kvCache });

		// Get logits for the last token in each sequence
		const [batchSize, seqLen, vocabSize] = logits.size();
		let lastTokenLogits = logits
			.slice([
				[0, batchSize],
				[seqLen - 1, seqLen],
				[0, vocabSize]
			])
			.view([batchSize, vocabSize]);

		if (config.temperature && config.temperature > 0) {
			lastTokenLogits = lastTokenLogits.div(config.temperature);
		}

		lastTokenLogits = lastTokenLogits.softmax(-1);

		// Choose next tokens: sample when temperature > 0, else greedy argmax
		const nextTokenTensor =
			config.temperature && config.temperature > 0
				? lastTokenLogits.multinomial(1, { replacement: false })
				: lastTokenLogits.argmax({ dim: -1 });
		const nextTokensArray = await tensorToTokens(nextTokenTensor);

		// Update sequences and check for stop conditions
		const shouldContinue = shouldContinueGeneration(results, nextTokensArray, stopTokenSet);
		// Compute tokens appended this step across active sequences
		let appendedThisStep = 0;
		for (let i = 0; i < results.length; i++) {
			appendedThisStep += results[i].length - prevLengths[i];
		}
		const tokensPerSecond = getTokensPerSecond(appendedThisStep);

		// Yield current state with sequences and logits
		yield {
			sequences: isBatch ? results.map((seq) => [...seq]) : [results[0].slice()],
			probs: lastTokenLogits, // Provide the softmax'd logits for the last token
			tokensPerSecond
		};

		// Mark cache as seeded after first forward with cache
		if (kvCache && !seeded) seeded = true;

		// If all sequences hit stop tokens, break
		if (!shouldContinue) {
			break;
		}

		step++;

		if (config.maxTokens !== undefined && step >= config.maxTokens) {
			break;
		}
	}
}

/**
 * Generate tokens for Transformer (encoder-decoder) model
 */
export async function* generateTransformerStream(
	model: EncoderDecoderTransformer,
	sourceInput: number[] | number[][],
	config: GenerationConfig & { startToken?: number; maxTargetLength?: number } = {}
): AsyncGenerator<GenerationResult, void, unknown> {
	const stopTokenSet = normalizeStopTokens(config.stopTokens);
	const isBatch = Array.isArray(sourceInput[0]);
	const sourceSequences = isBatch ? (sourceInput as number[][]) : [sourceInput as number[]];
	const startToken = config.startToken ?? 0; // Default start token
	const maxTargetLength = config.maxTargetLength ?? config.maxTokens ?? 50;

	// Create source tensor and encode once
	const device = model.lmHead.weight.device;
	const sourceTensor = createInputTensor(sourceSequences, device);
	const encoderHiddenStates = model.encode(sourceTensor);

	// Initialize target sequences with start token
	const targetResults = sourceSequences.map(() => [startToken]);

	let step = 0;
	const kvCache: DecoderKVCache | null =
		(config.useKvCache ?? true) ? createEmptyDecoderKVCache(model.config.nDecoderLayer) : null;
	let seeded = false;
	const getTokensPerSecond = startTokensPerSecondTracker();

	while (step < maxTargetLength) {
		// Seed cache once with full targets, then switch to single-token steps
		const prevLengths = targetResults.map((seq) => seq.length);
		let targetTensor: Tensor;
		let tgtPaddingMask: Tensor | null = null;
		if (kvCache && seeded) {
			const latestTargets = targetResults.map((seq) => [seq[seq.length - 1] ?? startToken]);
			targetTensor = createInputTensor(latestTargets, device);
			// No padding mask needed for single-token incremental step
			tgtPaddingMask = null;
		} else {
			const { padded: paddedTargets, paddingMask } = rightPadSequences(
				targetResults,
				Array.isArray(config.stopTokens) ? config.stopTokens[0] : config.stopTokens
			);
			targetTensor = createInputTensor(paddedTargets, device);
			tgtPaddingMask = tensor(paddingMask, { device, dtype: int32 });
		}

		// Use the model's forward method with pre-computed encoder hidden states
		const [logits, _] = model.forward(null, targetTensor, {
			tgtPaddingMask,
			encoderHiddenStates,
			kvCache
		});

		// Get logits for the last token in each sequence
		const [batchSize, seqLen, vocabSize] = logits.size();
		let lastTokenLogits = logits
			.slice([
				[0, batchSize],
				[seqLen - 1, seqLen],
				[0, vocabSize]
			])
			.view([batchSize, vocabSize]);

		if (config.temperature) {
			lastTokenLogits = lastTokenLogits.div(config.temperature);
		}

		lastTokenLogits = lastTokenLogits.softmax(-1);

		// Choose next tokens: sample when temperature > 0, else greedy argmax
		const nextTokenTensor =
			config.temperature && config.temperature > 0
				? lastTokenLogits.multinomial(1, { replacement: false })
				: lastTokenLogits.argmax({ dim: -1 });
		const nextTokensArray = await tensorToTokens(nextTokenTensor);

		// Update sequences and check for stop conditions
		const shouldContinue = shouldContinueGeneration(targetResults, nextTokensArray, stopTokenSet);

		// Yield current state with sequences and logits
		let appendedThisStep = 0;
		for (let i = 0; i < targetResults.length; i++) {
			appendedThisStep += targetResults[i].length - prevLengths[i];
		}
		const tokensPerSecond = getTokensPerSecond(appendedThisStep);
		yield {
			sequences: isBatch ? targetResults.map((seq) => [...seq]) : [targetResults[0].slice()],
			probs: lastTokenLogits, // Provide the logits for the last token
			tokensPerSecond
		};

		// Mark cache as seeded after first forward with cache
		if (kvCache && !seeded) seeded = true;

		// If all sequences hit stop tokens, break
		if (!shouldContinue) {
			break;
		}

		step++;

		if (config.maxTokens !== undefined && step >= config.maxTokens) {
			break;
		}
	}
}

/**
 * Unified generate function that works with any supported model
 */
export async function* generateStream(
	model: GeneratableModel,
	input: number[] | number[][],
	config: GenerationConfig = {}
): AsyncGenerator<GenerationResult, { tokensPerSecond?: number } | void, unknown> {
	// Check model type and delegate to appropriate generator, returning inner summary
	if (model instanceof EncoderDecoderTransformer) {
		// This is a Transformer (encoder-decoder) model
		return yield* generateTransformerStream(model, input, config);
	} else if (model instanceof DecoderTransformer) {
		// This is a GPT (decoder-only) model
		return yield* generateGPTStream(model as DecoderTransformer, input, config);
	} else {
		throw new Error('Unsupported model type for generation');
	}
}

/**
 * Standard generate function that collects all tokens (backward compatible)
 */
export async function generate(
	model: GeneratableModel,
	input: number[] | number[][],
	config: GenerationConfig = {}
): Promise<number[][]> {
	const results: number[][] = [];
	let tokenCount = 0;
	const maxTokens = config.maxTokens ?? 50;

	for await (const generationResult of generateStream(model, input, config)) {
		results.length = 0;
		results.push(...generationResult.sequences);
		tokenCount++;

		if (tokenCount >= maxTokens) {
			break;
		}
	}

	return results;
}

/**
 * Right-pad sequences to a uniform length for batched forward passes.
 * Returns padded sequences and a padding mask (1 for real tokens, 0 for padding).
 * Note: The original sequences array is NOT modified.
 */
export function rightPadSequences(
	sequences: number[][],
	padToken?: number
): { padded: number[][]; paddingMask: number[][] } {
	const maxLen = sequences.reduce((m, s) => Math.max(m, s.length), 0);
	const padded: number[][] = new Array(sequences.length);
	const paddingMask: number[][] = new Array(sequences.length);

	for (let i = 0; i < sequences.length; i++) {
		const seq = sequences[i];
		const realLen = seq.length;
		const rowMask: number[] = new Array(maxLen).fill(0);
		for (let j = 0; j < realLen; j++) rowMask[j] = 1;

		if (realLen === maxLen) {
			padded[i] = [...seq];
			paddingMask[i] = rowMask;
			continue;
		}

		let padVal: number;
		if (padToken !== undefined) {
			padVal = padToken;
		} else if (realLen > 0) {
			padVal = seq[realLen - 1];
		} else {
			padVal = 0; // degenerate case
		}

		const numPad = maxLen - realLen;
		padded[i] =
			realLen > 0 ? [...seq, ...new Array(numPad).fill(padVal)] : new Array(maxLen).fill(padVal);
		paddingMask[i] = rowMask;
	}

	return { padded, paddingMask };
}
