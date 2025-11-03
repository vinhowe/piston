import type { Tensor } from '@piston-ml/piston-web';

import { MersenneTwister19937, Random } from 'random-js';

import { type CollateWrapFunction, tensorWrap } from '../collate';
import {
	deriveToySampleSeed,
	type ToyAutoregressiveBatch,
	type ToyBidirectionalBatch,
	type ToyDatasetLike,
	type ToyEncoderDecoderBatch,
	type ToySequence
} from './dataset';

// Helper function to add special tokens to sequences for rawData
function withSpecials(
	body: number[],
	{ bosId, eosId }: { bosId: number | null; eosId: number | null }
): number[] {
	const seq = [...body];
	if (eosId !== null) seq.push(eosId);
	return bosId !== null ? [bosId, ...seq] : seq;
}

interface AutoregressiveCollateOptions<T> {
	ignorePrompt?: boolean;
	wrapFunction?: CollateWrapFunction<T> | null;
}

interface BidirectionalCollateOptions<T> {
	maskPrompt?: boolean;
	maskRatio?: number;
	generator?: Random;
	wrapFunction?: CollateWrapFunction<T> | null;
}

export interface EncoderDecoderCollateOptions<T> {
	wrapFunction?: CollateWrapFunction<T> | null;
}

export function toyDatasetAutoregressiveCollate<T>(
	batch: ToySequence[],
	dataset: ToyDatasetLike<unknown>,
	options: AutoregressiveCollateOptions<T> = {}
): ToyAutoregressiveBatch<T> {
	const { ignorePrompt = false, wrapFunction = tensorWrap } = options;

	const sequencesData = batch.map(({ prompt, target: completion, mask }) => {
		// Build full sequence, potentially adding BOS at start and EOS at end
		const fullSequence: number[] = [];

		// Add BOS if available
		if (dataset.bosId !== null) {
			fullSequence.push(dataset.bosId);
		}

		// Add prompt and completion
		fullSequence.push(...(prompt ?? []), ...completion);

		// Add EOS if available
		if (dataset.eosId !== null) {
			fullSequence.push(dataset.eosId);
		}

		const input = fullSequence.slice(0, -1);
		const target = fullSequence.slice(1);

		// Mask prompt tokens in target if ignorePrompt is true
		// Note: target has already been shifted by slice(1), so no bosOffset needed
		const promptLength = prompt?.length ?? 0;
		if (ignorePrompt && promptLength > 0) {
			target.fill(-100, 0, promptLength);
		}

		// Apply dataset-provided mask to target positions that correspond to completion tokens
		if (mask && mask.length > 0) {
			for (let i = 0; i < completion.length; i++) {
				if (mask[i] === false) {
					const targetIndex = promptLength + i;
					if (targetIndex >= 0 && targetIndex < target.length) {
						target[targetIndex] = -100;
					}
				}
			}
		}

		// Create visible completion with mask tokens for rawData
		const visibleCompletion = [...completion];
		if (mask && dataset.maskId !== null) {
			for (let i = 0; i < visibleCompletion.length; i++) {
				if (mask[i] === false) {
					visibleCompletion[i] = dataset.maskId;
				}
			}
		}

		return {
			input,
			target,
			rawData: {
				fullSequence,
				prompt: withSpecials(prompt ?? [], { bosId: dataset.bosId, eosId: null }),
				target: withSpecials(visibleCompletion, { bosId: null, eosId: dataset.eosId }),
				ignored: target.map((t) => t === -100)
			}
		};
	});

	const inputSequences = sequencesData.map(({ input }) => input);
	const targetSequences = sequencesData.map(({ target }) => target);

	const input = wrapFunction ? wrapFunction(inputSequences) : inputSequences;
	const target = wrapFunction ? wrapFunction(targetSequences) : targetSequences;

	return {
		tensors: [input as T, target as T],
		raw: sequencesData.map(({ rawData }) => rawData),
		samples: batch
	};
}

export function toyDatasetBidirectionalCollate<T = Tensor>(
	batch: ToySequence[],
	dataset: ToyDatasetLike<unknown>,
	options: BidirectionalCollateOptions<T> = {}
): ToyBidirectionalBatch<T> {
	const { maskPrompt = false, maskRatio = 0.15, generator, wrapFunction = tensorWrap } = options;

	if (dataset.maskId === null) {
		throw new Error(
			'Encoder-only collation requires a mask token, but none was configured in the dataset'
		);
	}

	const maskToken = dataset.maskId;

	const sequencesData = batch.map((seq) => {
		const { prompt, target, mask, absoluteIndex } = seq;
		const fullSequence: number[] = [...(prompt ?? []), ...target];
		// If a global generator is not provided, derive a per-sample RNG from baseSeed and absoluteIndex
		const rng = generator
			? generator
			: new Random(
					MersenneTwister19937.seed(
						deriveToySampleSeed(
							dataset.baseSeed,
							dataset.datasetName,
							absoluteIndex ?? 0,
							'toy-mask'
						)
					)
				);

		const maskedSequence = [...fullSequence];
		const labels = new Array(fullSequence.length).fill(-100);

		// Determine which tokens are eligible for masking
		const promptLength = prompt?.length ?? 0;

		const eligibleForMasking = fullSequence.map((_, index) => {
			if (maskPrompt) {
				// Allow masking across both prompt and target (no specials in bidirectional)
				return true;
			} else {
				// Only target tokens eligible (after prompt)
				const targetStart = promptLength;
				const targetEnd = fullSequence.length;
				if (index < targetStart || index >= targetEnd) return false;
				// If dataset provided a mask array, restrict eligible positions to those marked true
				if (mask && mask[index - targetStart] === false) return false;
				return true;
			}
		});

		// Apply random masking and track which positions got masked
		const maskedPositions: boolean[] = new Array(fullSequence.length).fill(false);
		eligibleForMasking.forEach((eligible, index) => {
			if (eligible && rng.real(0, 1) < maskRatio) {
				maskedSequence[index] = maskToken;
				maskedPositions[index] = true;
				labels[index] = fullSequence[index]; // compute loss against original token
			}
		});

		// Ensure at least one token is masked per example
		if (!maskedPositions.some((v) => v)) {
			// Prefer among eligible positions
			const candidates: number[] = eligibleForMasking
				.map((eligible, idx) => (eligible ? idx : -1))
				.filter((idx) => idx >= 0);
			if (candidates.length > 0) {
				const forcedIdx = candidates[rng.integer(0, candidates.length - 1)];
				maskedSequence[forcedIdx] = maskToken;
				maskedPositions[forcedIdx] = true;
				labels[forcedIdx] = fullSequence[forcedIdx];
			} else {
				throw new Error('No tokens eligible for masking');
			}
		}

		return {
			maskedSequence,
			labels,
			rawData: {
				fullSequence: maskedSequence,
				prompt: [...(prompt ?? [])],
				target: [...target],
				ignored: labels.map((l) => l === -100)
			}
		};
	});

	const inputSequences = sequencesData.map(({ maskedSequence }) => maskedSequence);
	const labelSequences = sequencesData.map(({ labels }) => labels);
	const attentionMaskSequences = sequencesData.map(({ maskedSequence }) =>
		maskedSequence.map(() => 1)
	);

	const input = wrapFunction ? wrapFunction(inputSequences) : inputSequences;
	const labels = wrapFunction ? wrapFunction(labelSequences) : labelSequences;
	const attentionMask = wrapFunction
		? wrapFunction(attentionMaskSequences)
		: attentionMaskSequences;

	return {
		tensors: [input as T, labels as T, attentionMask as T],
		raw: sequencesData.map(({ rawData }) => rawData),
		samples: batch
	};
}

export function toyDatasetEncoderDecoderCollate<T>(
	batch: ToySequence[],
	dataset: ToyDatasetLike<unknown>,
	options: EncoderDecoderCollateOptions<T> = {}
): ToyEncoderDecoderBatch<T> {
	if (dataset.bosId === null) {
		throw new Error(
			'Encoder-decoder collation requires a BOS token, but none was configured in the dataset'
		);
	}

	const encoderData = batch.map(({ prompt }) => {
		// Build encoder input WITHOUT BOS/EOS
		const encoderSequence: number[] = [...(prompt ?? [])];

		return {
			encoderSequence,
			rawData: {
				fullSequence: encoderSequence,
				prompt: encoderSequence, // encoder sequence is the prompt without specials
				target: [], // No target for encoder
				ignored: []
			}
		};
	});

	const decoderData = batch.map(({ target, mask }) => {
		// Apply mask to target array first
		let maskedTarget = [...target];
		if (mask) {
			maskedTarget = maskedTarget.map((token, i) => (mask[i] === false ? -100 : token));
		}

		// Build decoder target with EOS if available
		const decoderTarget: number[] = [...maskedTarget];
		if (dataset.eosId !== null) {
			decoderTarget.push(dataset.eosId);
		}

		// Decoder input: BOS + target (for teacher forcing)
		// Note: dataset.bosId is guaranteed to be non-null because we checked earlier
		// If EOS is enabled, we feed BOS + full target so model predicts target then EOS.
		// If EOS is disabled, feed BOS + target[:-1] so input/target lengths match and
		// the loss is computed only over generated tokens.
		const decoderInput: number[] =
			dataset.eosId !== null
				? [dataset.bosId!, ...target]
				: [dataset.bosId!, ...target.slice(0, -1)];

		// Create visible target with mask tokens for rawData
		const visibleTarget = [...target];
		if (mask && dataset.maskId !== null) {
			for (let i = 0; i < visibleTarget.length; i++) {
				if (mask[i] === false) {
					visibleTarget[i] = dataset.maskId;
				}
			}
		}

		return {
			decoderInput,
			decoderTarget,
			rawData: {
				fullSequence: decoderInput,
				prompt: [], // decoder has no prompt concept
				target: withSpecials(visibleTarget, { bosId: null, eosId: dataset.eosId }),
				ignored: decoderTarget.map((t) => t === -100)
			}
		};
	});

	const encoderSequences = encoderData.map(({ encoderSequence }) => encoderSequence);
	const decoderInputSequences = decoderData.map(({ decoderInput }) => decoderInput);
	const decoderTargetSequences = decoderData.map(({ decoderTarget }) => decoderTarget);

	const encoderInput = options.wrapFunction
		? options.wrapFunction(encoderSequences)
		: encoderSequences;
	const decoderInput = options.wrapFunction
		? options.wrapFunction(decoderInputSequences)
		: decoderInputSequences;
	const decoderTarget = options.wrapFunction
		? options.wrapFunction(decoderTargetSequences)
		: decoderTargetSequences;

	// Combine encoder and decoder raw data into single sequences
	const combinedRaw = batch.map((_, index) => {
		const encoderRaw = encoderData[index].rawData;
		const decoderRaw = decoderData[index].rawData;

		return {
			fullSequence: [...encoderRaw.fullSequence, ...decoderRaw.fullSequence],
			prompt: encoderRaw.prompt, // encoder provides the prompt
			target: decoderRaw.target, // decoder provides the target
			ignored: [...encoderRaw.ignored, ...decoderRaw.ignored]
		};
	});

	return {
		tensors: [encoderInput as T, decoderInput as T, decoderTarget as T],
		raw: combinedRaw,
		samples: batch
	};
}
