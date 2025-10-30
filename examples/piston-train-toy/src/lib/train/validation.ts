import type { Config, ValidationConfig } from '$lib/workspace/config';
import type { BaseStepData, TokenRollout } from '$lib/workspace/runs.svelte';

import { type Tensor, weak } from '@piston-ml/piston-web';
import { MersenneTwister19937, Random } from 'random-js';

import type { ToyValidationMetrics } from './data/toy/types';
import type {
	BidirectionalBatchType,
	EncoderDecoderBatchType,
	GeneratableModel,
	NaturalCollateFnType,
	PistonCollateFnType,
	PistonDatasetType,
	ToyCollateFnType
} from './types';

import { naturalLanguageBidirectionalCollate, NaturalLanguageDataset } from './data/natural';
import {
	toyDatasetAutoregressiveCollate,
	toyDatasetBidirectionalCollate,
	toyDatasetEncoderDecoderCollate
} from './data/toy/collate';
import { type ToyDatasetLike, type ToySequence } from './data/toy/dataset';
import {
	DecoderTransformer,
	EncoderDecoderTransformer,
	EncoderTransformer
} from './model/transformer';
import {
	generateDecoderCompletions,
	generateEncoderDecoderCompletions,
	predictEncoderOnlyCompletions
} from './validationHelpers';

export type ValidationStep = BaseStepData & {
	type: 'validation';
	completions: TokenRollout[];
	samplingParams: {
		temperature: number;
	};
	// Running average throughput across the generation in tokens/second (decoder/generative only)
	tokensPerSecond?: number;
	targets?: number[][]; // Only present in first step, what tokens should have been
	encoderInputs?: number[][]; // Encoder/source inputs used (for encoder-decoder or encoder-only)
	decoderPromptLengths?: number[]; // For decoder-only display: prompt token counts per example
	matches?: boolean[][]; // Per-example, per-token correctness flags
	metrics?: ToyValidationMetrics[]; // Per-example metrics computed by the dataset
};

export type ToyValidationExamples = {
	toySequences: ToySequence[];
	prompts: number[][]; // decoder-only: prompt; encoder-decoder: encoder input (with specials)
	targets: number[][];
	actualTargets: number[][];
};

export type NaturalValidationExamples = {
	naturalSequences: number[][];
};

export type ValidationExamples = ToyValidationExamples | NaturalValidationExamples;

export function buildValidationExamplesSubset(
	examples: ValidationExamples,
	subsetSize: number
): ValidationExamples {
	if ('toySequences' in examples) {
		return {
			toySequences: examples.toySequences.slice(0, subsetSize),
			prompts: examples.prompts.slice(0, subsetSize),
			targets: examples.targets.slice(0, subsetSize),
			actualTargets: examples.actualTargets.slice(0, subsetSize)
		};
	}

	if ('naturalSequences' in examples) {
		return {
			naturalSequences: examples.naturalSequences.slice(0, subsetSize)
		};
	}

	throw new Error('Unsupported validation examples');
}

export function prepareToyValidationExamples(
	config: Config,
	dataset: ToyDatasetLike<unknown>,
	options: { isDecoderOnly: boolean; isEncoderDecoder: boolean }
): ToyValidationExamples {
	const { isDecoderOnly, isEncoderDecoder } = options;

	const valSequences: ToySequence[] = [];
	const valPrompts: number[][] = [];
	const valTargets: number[][] = [];
	const valActualTargets: number[][] = [];
	const it = dataset[Symbol.iterator]();

	for (let i = 0; i < config.training.validation.batchSize; i++) {
		const sampleSequence = it.next().value as ToySequence;
		valSequences.push(sampleSequence);

		let samplePrompt: number[] | undefined;
		let sampleTarget: number[];
		let actualTarget: number[];

		if (isDecoderOnly) {
			const collateResult = toyDatasetAutoregressiveCollate<number[][]>([sampleSequence], dataset, {
				ignorePrompt: !config.data.trainOnPrompt,
				wrapFunction: null
			});
			samplePrompt = collateResult.raw[0].prompt;
			sampleTarget = collateResult.raw[0].target ?? [];
			actualTarget = collateResult.tensors[1][0];
		} else if (isEncoderDecoder) {
			const collateResult = toyDatasetEncoderDecoderCollate<number[][]>([sampleSequence], dataset, {
				wrapFunction: null
			});
			// Visualize the encoder input (with specials) as the prompt
			samplePrompt = collateResult.raw[0].prompt;
			sampleTarget = collateResult.raw[0].target ?? [];
			// For encoder-decoder, tensors are [encoderInput, decoderInput, decoderTarget].
			// Use decoderTarget so it mirrors the training loss target (respects EOS setting).
			actualTarget = collateResult.tensors[2][0];
		} else {
			// Encoder-only models: build masked input and labels for visualization
			const collateResult = toyDatasetBidirectionalCollate<number[][]>([sampleSequence], dataset, {
				maskPrompt: config.data.trainOnPrompt,
				maskRatio: config.data.maskRatio,
				generator: dataset.generator,
				wrapFunction: null
			});
			// Show masked sequence (what encoder saw) as the prompt
			samplePrompt = collateResult.raw[0].fullSequence;
			// Use labels (with -100 for unmasked) as targets so viewer can compare only masked positions
			sampleTarget = collateResult.tensors[1][0];
			// Actual training labels (with -100) for correctness comparison
			actualTarget = collateResult.tensors[1][0];
		}

		valPrompts.push(samplePrompt ?? []);
		valTargets.push(sampleTarget);
		valActualTargets.push(actualTarget);
	}

	if (valPrompts.length === 0) {
		return { toySequences: [], prompts: [], targets: [], actualTargets: [] };
	}

	return {
		toySequences: valSequences,
		prompts: valPrompts,
		targets: valTargets,
		actualTargets: valActualTargets
	};
}

export async function prepareNaturalValidationExamples(
	config: Config,
	dataset: NaturalLanguageDataset
): Promise<NaturalValidationExamples> {
	const naturalSequences: number[][] = [];

	let count = 0;
	for await (const sampleSequence of dataset) {
		naturalSequences.push(sampleSequence);
		count++;
		if (count >= config.training.validation.batchSize) break;
	}

	return { naturalSequences };
}

export async function prepareValidationExamples(
	config: Config,
	dataset: PistonDatasetType,
	options: { isDecoderOnly: boolean; isEncoderDecoder: boolean }
): Promise<ValidationExamples> {
	if (dataset instanceof NaturalLanguageDataset) {
		return await prepareNaturalValidationExamples(config, dataset);
	}
	return prepareToyValidationExamples(config, dataset, options);
}

export async function computeToyValidationMetrics(
	model: GeneratableModel,
	dataset: ToyDatasetLike<unknown>,
	valExamples: ToyValidationExamples,
	valConfig: ValidationConfig,
	options: { isDecoderOnly: boolean; isEncoderDecoder: boolean; includeTargets: boolean }
): Promise<Omit<ValidationStep, 'step'>> {
	const { isDecoderOnly, isEncoderDecoder, includeTargets } = options;

	let tokensPerSecond: number | undefined;
	let completions: TokenRollout[] = [];

	const maxTokens = Math.max(...valExamples.targets.map((t) => t.length));

	if (isDecoderOnly && model instanceof DecoderTransformer) {
		const prompts = valExamples.prompts.map((prompt) =>
			prompt.length > 0 ? prompt : [dataset.bosId!]
		);
		const result = await generateDecoderCompletions(model, prompts, {
			maxTokens,
			stopTokens: dataset.eosId !== null ? [dataset.eosId!] : [],
			temperature: valConfig.temperature,
			useKvCache: valConfig.useKvCache
		});
		completions = result.completions;
		tokensPerSecond = result.tokensPerSecond;
	} else if (isEncoderDecoder && model instanceof EncoderDecoderTransformer) {
		const sources = valExamples.prompts.map((prompt) =>
			prompt.length > 0 ? prompt : [dataset.bosId!]
		);
		const result = await generateEncoderDecoderCompletions(model, sources, {
			maxTokens,
			startToken: dataset.bosId ?? undefined,
			stopTokens: dataset.eosId !== null ? [dataset.eosId!] : [],
			temperature: valConfig.temperature,
			useKvCache: valConfig.useKvCache
		});
		completions = result.completions;
		tokensPerSecond = result.tokensPerSecond;
	} else {
		if (model instanceof EncoderTransformer) {
			const result = await predictEncoderOnlyCompletions(
				model,
				valExamples.prompts,
				valExamples.actualTargets,
				{ temperature: valConfig.temperature }
			);
			completions = result.completions;
		} else {
			throw new Error('Invalid model for encoder-only validation');
		}
	}

	const validationStepData: Omit<ValidationStep, 'step'> = {
		type: 'validation',
		completions,
		samplingParams: { temperature: valConfig.temperature },
		targets: includeTargets ? valExamples.actualTargets : undefined,
		// For display only: shave BOS from encoder inputs
		encoderInputs: isEncoderDecoder || !isDecoderOnly ? valExamples.prompts : undefined,
		// Only set for generative paths
		tokensPerSecond
	};

	// Compute per-example metrics
	try {
		const batchSize = valExamples.prompts.length;
		const metrics: Array<ToyValidationMetrics> = new Array(batchSize);
		const matches: boolean[][] = [];
		for (let bi = 0; bi < batchSize; bi++) {
			const rollout = completions[bi];
			const generatedIds = rollout?.tokenIds ?? [];

			// Build predicted/target slices
			let predictedSlice: number[] = [];
			let targetSlice: number[] = [];

			if (!isDecoderOnly && !isEncoderDecoder) {
				// Encoder-only: masked positions only
				const labelsRow = valExamples.actualTargets[bi] ?? [];
				const maskedIndices: number[] = [];
				for (let i = 0; i < labelsRow.length; i++) {
					if (labelsRow[i] !== -100) maskedIndices.push(i);
				}
				const filtered = maskedIndices.filter((i) => generatedIds[i] !== undefined);
				predictedSlice = filtered.map((i) => generatedIds[i]);
				targetSlice = filtered.map((i) => labelsRow[i]);
			} else {
				// Decoder-only or Encoder-decoder
				const targetTokens = valExamples.targets[bi] ?? [];
				const prefixLength = isDecoderOnly
					? (valExamples.prompts[bi]?.length ?? 0)
					: isEncoderDecoder
						? 1
						: 0;
				predictedSlice = generatedIds.slice(prefixLength, prefixLength + targetTokens.length);
				targetSlice = targetTokens;
			}

			// Single computeMetrics call and shared handling
			const m = dataset.computeMetrics(predictedSlice, targetSlice);
			metrics[bi] = m;
			if (Array.isArray(m.matches)) {
				matches[bi] = m.matches;
			}
		}
		validationStepData.metrics = metrics;
		validationStepData.matches = matches.length > 0 ? matches : undefined;
	} catch (e) {
		console.warn('Failed to compute validation metrics:', e);
	}

	return validationStepData;
}

export async function computeNaturalValidationMetrics(
	model: GeneratableModel,
	dataset: NaturalLanguageDataset,
	valExamples: NaturalValidationExamples,
	valConfig: ValidationConfig,
	options: { isDecoderOnly: boolean; includeTargets: boolean; maskRatio: number }
): Promise<Omit<ValidationStep, 'step'>> {
	const { isDecoderOnly, maskRatio } = options;

	let promptLen = 0;
	let encoderOnlyTargets: number[][] | null = null;
	let tokensPerSecond: number | undefined;
	let completions: TokenRollout[] = [];

	const contextSize = dataset.contextSize;

	if (isDecoderOnly && model instanceof DecoderTransformer) {
		// promptLen = Math.max(Math.floor(contextSize / 4), 1);
		promptLen = 8;
		const eosId = dataset.eosId as number;
		const starts = valExamples.naturalSequences.map((seq) => seq.slice(0, promptLen));
		const maxTokens = Math.max(0, contextSize - promptLen);
		const result = await generateDecoderCompletions(model, starts, {
			maxTokens,
			stopTokens: eosId !== null ? [eosId] : [],
			temperature: valConfig.temperature,
			useKvCache: valConfig.useKvCache
		});
		completions = result.completions;
		tokensPerSecond = result.tokensPerSecond;
	} else {
		// Encoder-only: predict masked tokens using MLM logits across the whole sequence
		const maskTokenId = dataset.maskId as number;
		const generator = new Random(MersenneTwister19937.autoSeed());

		const collated = naturalLanguageBidirectionalCollate<number[][]>(valExamples.naturalSequences, {
			maskRatio,
			generator,
			maskTokenId,
			wrapFunction: null
		});

		const inputs = collated.tensors[0];
		const labels = collated.tensors[1]; // -100 for unmasked
		encoderOnlyTargets = labels;

		if (model instanceof EncoderTransformer) {
			const attentionMask = collated.tensors[2];
			const result = await predictEncoderOnlyCompletions(model, inputs, labels, {
				attentionMask,
				temperature: valConfig.temperature
			});
			completions = result.completions;
		} else {
			throw new Error('Invalid model for encoder-only natural validation');
		}
	}

	const validationStepData: Omit<ValidationStep, 'step'> = {
		type: 'validation',
		completions,
		samplingParams: { temperature: valConfig.temperature },
		decoderPromptLengths: isDecoderOnly
			? new Array(valExamples.naturalSequences.length).fill(promptLen)
			: undefined,
		targets: !isDecoderOnly && encoderOnlyTargets ? encoderOnlyTargets : undefined,
		tokensPerSecond
	};

	// Compute metrics for natural encoder-only (MLM) similar to toy datasets
	if (!isDecoderOnly && encoderOnlyTargets) {
		try {
			const B = completions.length;
			const matches: boolean[][] = new Array(B);
			const numericMetrics: Array<ToyValidationMetrics> = new Array(B);
			for (let bi = 0; bi < B; bi++) {
				const predIds: number[] = completions[bi]?.tokenIds ?? [];
				const labelsRow: number[] = encoderOnlyTargets[bi] ?? [];
				let correct = 0;
				let total = 0;
				const matchRow: boolean[] = new Array(labelsRow.length).fill(false);
				for (let ti = 0; ti < labelsRow.length; ti++) {
					const label = labelsRow[ti];
					if (label === -100) continue; // skip unmasked positions
					total++;
					const ok = predIds[ti] === label;
					matchRow[ti] = ok;
					if (ok) correct++;
				}
				const accuracy = total > 0 ? correct / total : 0;
				numericMetrics[bi] = {
					accuracy,
					matches: matchRow
				};
				matches[bi] = matchRow;
			}
			validationStepData.metrics = numericMetrics;
			validationStepData.matches = matches;
		} catch (e) {
			console.warn('Failed to compute natural MLM validation metrics:', e);
		}
	}

	return validationStepData;
}

export function buildValidationLog(
	validationStepData: Omit<ValidationStep, 'step'>
): Record<string, number | Omit<ValidationStep, 'step'>> {
	// Aggregate numeric-like metrics from per-example metrics; average arrays per-example; skip 'matches'
	const aggregatedNumeric: Record<string, number> = {};
	const counts: Record<string, number> = {};
	const perExample = validationStepData.metrics;
	if (perExample && perExample.length > 0) {
		for (const m of perExample) {
			for (const entry of Object.entries(m)) {
				let [key] = entry;
				const [_, value] = entry;
				if (key === 'matches') {
					key = 'character_level_accuracy';
				}
				let numericValue: number | null = null;
				if (typeof value === 'number' && Number.isFinite(value)) {
					numericValue = value;
				} else if (typeof value === 'boolean') {
					numericValue = value ? 1 : 0;
				} else if (Array.isArray(value)) {
					// Average arrays of numbers or booleans
					const arr = value;
					if (arr.length > 0) {
						const sum = arr.reduce((s: number, v: number | boolean) => {
							const num = typeof v === 'boolean' ? (v ? 1 : 0) : Number.isFinite(v) ? v : 0;
							return s + num;
						}, 0);
						numericValue = sum / arr.length;
					}
				}
				if (numericValue !== null) {
					aggregatedNumeric[key] = (aggregatedNumeric[key] ?? 0) + numericValue;
					counts[key] = (counts[key] ?? 0) + 1;
				}
			}
		}
		for (const key of Object.keys(aggregatedNumeric)) {
			aggregatedNumeric[key] = aggregatedNumeric[key] / (counts[key] || 1);
		}
	}

	// Compute number of unique completions by hashing tokenIds
	const uniqueCompletionsCount = (() => {
		const seen = new Set<string>();
		for (const c of validationStepData.completions ?? []) {
			const ids = c?.tokenIds ?? [];
			seen.add(ids.join(','));
		}
		return seen.size;
	})();

	const validationLog: Record<string, number | typeof validationStepData> = {
		'validation/completions': validationStepData,
		'validation/unique_completions': uniqueCompletionsCount
	};
	if (typeof validationStepData.tokensPerSecond === 'number') {
		validationLog['validation/tokens_per_second'] = validationStepData.tokensPerSecond;
	}
	for (const [k, v] of Object.entries(aggregatedNumeric)) {
		validationLog[`validation/${k}`] = v;
	}
	return validationLog;
}

export async function computeLikelihoodMetrics(
	model: GeneratableModel,
	sequences: ValidationExamples,
	collateFn: PistonCollateFnType<Tensor>
): Promise<{ valLoss: number; perplexity: number }> {
	return await weak(async () => {
		model.eval();

		let valLoss: number | null = null;
		try {
			let collated;
			if ('toySequences' in sequences) {
				collated = (collateFn as ToyCollateFnType<Tensor>)(sequences.toySequences);
			} else {
				collated = (collateFn as NaturalCollateFnType<Tensor>)(sequences.naturalSequences);
			}

			let loss: Tensor | null = null;
			let modelName = '';
			if (model instanceof DecoderTransformer) {
				const [inputs, targets] = collated.tensors;
				[, loss] = model.forward(await inputs.to('gpu'), {
					targets: await targets.to('gpu')
				});
				modelName = 'decoder-only';
			} else if (model instanceof EncoderDecoderTransformer) {
				const [encoderInputs, decoderInputs, decoderTargets] = (
					collated as EncoderDecoderBatchType<Tensor>
				).tensors;
				[, loss] = model.forward(await encoderInputs.to('gpu'), await decoderInputs.to('gpu'), {
					targets: await decoderTargets.to('gpu')
				});
				modelName = 'encoder-decoder';
			} else if (model instanceof EncoderTransformer) {
				// Encoder-only: compute MLM loss over masked tokens
				const [inputs, labels, attentionMask] = (collated as BidirectionalBatchType<Tensor>)
					.tensors;
				modelName = 'encoder-only';
				[, , , loss] = model.forward(await inputs.to('gpu'), {
					attentionMask: await attentionMask.to('gpu'),
					targets: await labels.to('gpu')
				});
			} else {
				throw new Error('Unsupported model for validation');
			}

			if (!loss) {
				throw new Error(`No loss tensor returned from ${modelName} model during validation`);
			}
			valLoss = await (await loss.to('cpu')).item();
			if (valLoss === null) {
				throw new Error(`Validation loss item is null for ${modelName} model`);
			}
		} finally {
			model.train();
		}

		const perplexity = Math.exp(valLoss);
		return { valLoss, perplexity };
	});
}
