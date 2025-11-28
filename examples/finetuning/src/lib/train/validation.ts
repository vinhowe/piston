import type { Config, ValidationConfig } from '$lib/workspace/config';
import type { BaseStepData, TokenRollout } from '$lib/workspace/runs.svelte';

import { type Tensor, weak } from '@piston-ml/piston-web';

import type { GeneratableModel, NaturalCollateFnType } from './types';

import { NaturalLanguageDataset } from './data/natural';
import { generateDecoderCompletions } from './validationHelpers';

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
};

export type NaturalValidationExamples = {
	naturalSequences: number[][];
};

export function buildValidationExamplesSubset(
	examples: NaturalValidationExamples,
	subsetSize: number
): NaturalValidationExamples {
	return {
		naturalSequences: examples.naturalSequences.slice(0, subsetSize)
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

export async function computeNaturalValidationMetrics(
	model: GeneratableModel,
	dataset: NaturalLanguageDataset,
	valExamples: NaturalValidationExamples,
	valConfig: ValidationConfig
): Promise<Omit<ValidationStep, 'step'>> {
	let promptLen = 0;

	const contextSize = dataset.contextSize;

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
	const { completions, tokensPerSecond } = result;

	const validationStepData: Omit<ValidationStep, 'step'> = {
		type: 'validation',
		completions,
		samplingParams: { temperature: valConfig.temperature },
		decoderPromptLengths: new Array(valExamples.naturalSequences.length).fill(promptLen),
		tokensPerSecond
	};

	return validationStepData;
}

export function buildValidationLog(
	validationStepData: Omit<ValidationStep, 'step'>
): Record<string, number | Omit<ValidationStep, 'step'>> {
	// Aggregate numeric-like metrics from per-example metrics; average arrays per-example; skip 'matches'
	const aggregatedNumeric: Record<string, number> = {};

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
	sequences: NaturalValidationExamples,
	collateFn: NaturalCollateFnType<Tensor>
): Promise<{ valLoss: number; perplexity: number }> {
	return await weak(async () => {
		model.eval();

		let valLoss: number | null = null;
		try {
			const collated = collateFn(sequences.naturalSequences);

			let loss: Tensor | null = null;
			const [inputs, targets] = collated.tensors;
			[, loss] = model.forward(await inputs.to('gpu'), {
				targets: await targets.to('gpu')
			});

			if (!loss) {
				throw new Error(`No loss tensor returned from decoder-only model during validation`);
			}
			valLoss = await (await loss.to('cpu')).item();
			if (valLoss === null) {
				throw new Error(`Validation loss item is null for decoder-only model`);
			}
		} finally {
			model.train();
		}

		const perplexity = Math.exp(valLoss);
		return { valLoss, perplexity };
	});
}
