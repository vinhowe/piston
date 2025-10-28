import type { Config } from '$lib/workspace/config';

import { type Tensor } from '@piston-ml/piston-web';

import type {
	BidirectionalBatchType,
	EncoderDecoderBatchType,
	GeneratableModel,
	NaturalCollateFnType,
	PistonCollateFnType,
	PistonDatasetType,
	ToyCollateFnType
} from './types';

import { NaturalLanguageDataset } from './data/natural';
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

export async function computeLikelihoodMetrics(
	model: GeneratableModel,
	sequences: ValidationExamples,
	collateFn: PistonCollateFnType<Tensor>
): Promise<{ valLoss: number; perplexity: number }> {
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
			const [inputs, labels, attentionMask] = (collated as BidirectionalBatchType<Tensor>).tensors;
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
}
