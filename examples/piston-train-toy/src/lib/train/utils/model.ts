import type { Random } from 'random-js';

import {
	CaptureIndexMode,
	DataLoader,
	type IndexState,
	Module,
	Tensor,
	weak
} from '@piston-ml/piston-web';
import * as piston from '@piston-ml/piston-web';

import type { Config } from '../../workspace/config';
import type {
	NaturalBatchType,
	NaturalCollateFnType,
	NaturalDataloaderType,
	PistonCollateFnType,
	PistonDataloaderType,
	PistonDatasetType,
	ToyBatchType,
	ToyCollateFnType,
	ToyDataloaderType
} from '../types';

import { buildDataset, type CollateWrapFunction, tensorWrap } from '../data';
import {
	naturalLanguageAutoregressiveCollate,
	naturalLanguageBidirectionalCollate,
	NaturalLanguageDataset
} from '../data/natural';
import {
	toyDatasetAutoregressiveCollate,
	toyDatasetBidirectionalCollate,
	toyDatasetEncoderDecoderCollate
} from '../data/toy/collate';
import { type ToyDatasetLike, type ToySequence } from '../data/toy/dataset';
import { RNNDecoder, RNNEncoder, RNNEncoderDecoder } from '../model/rnn';
import {
	DecoderTransformer,
	EncoderDecoderTransformer,
	EncoderTransformer
} from '../model/transformer';
import { initRNNParameters, initTransformerParameters } from './init';
import { parseSeed, seededRandom } from './random';

type EncoderDecoderBlockSize = { source: number; target: number };

/**
 * Calculate the vocabulary size based on the dataset and configuration
 */
export function calculateVocabSize(config: Config, dataset: PistonDatasetType): number {
	let vocabSize =
		'vocabSize' in dataset
			? (dataset.vocabSize as number)
			: Object.keys(dataset.tokenizer.vocab).length;

	// Round vocab size up to nearest multiple if configured
	if (config.model.roundVocabSizeToNearestMultiple.present) {
		vocabSize =
			Math.ceil(vocabSize / config.model.roundVocabSizeToNearestMultiple.value) *
			config.model.roundVocabSizeToNearestMultiple.value;
	}

	return vocabSize;
}

export function calculateBlockSize<T>(
	config: Config,
	dataloader: PistonDataloaderType<T>
): number | EncoderDecoderBlockSize {
	if (dataloader.dataset instanceof NaturalLanguageDataset) {
		return config.data.natural.contextSize;
	}

	// For toy datasets, infer lengths without advancing iteration using deterministic index-based generation
	const toy = dataloader.dataset as ToyDatasetLike;

	const sample = toy.generateSequenceAt(0);

	const promptLen = sample.prompt?.length ?? 0;
	const targetLen = sample.target.length;
	const eosPlus = toy.eosId !== null ? 1 : 0;
	const bosPlus = toy.bosId !== null ? 1 : 0;

	if (config.model.topology === 'encoder-decoder') {
		// Encoder input uses prompt (no BOS/EOS). Decoder target includes EOS if enabled.
		return { source: promptLen, target: targetLen + eosPlus };
	}

	if (config.model.topology === 'encoder') {
		// Encoder-only sees prompt+target (no specials)
		return promptLen + targetLen;
	}

	// Decoder-only sees BOS + prompt + target + EOS in full sequence
	return bosPlus + promptLen + targetLen + eosPlus;
}

// Overloads for strong typing based on dataset kind
export function createCollateFn<W = Tensor>(
	config: Config,
	dataset: NaturalLanguageDataset,
	maskGenerator: Random | null,
	wrapFunction?: CollateWrapFunction<W> | null
): NaturalCollateFnType<W>;
export function createCollateFn<W = Tensor>(
	config: Config,
	dataset: ToyDatasetLike,
	maskGenerator: Random | null,
	wrapFunction?: CollateWrapFunction<W> | null
): ToyCollateFnType<W>;
export function createCollateFn<W = Tensor>(
	config: Config,
	dataset: PistonDatasetType,
	maskGenerator: Random | null,
	wrapFunction?: CollateWrapFunction<W> | null
): PistonCollateFnType<W>;
export function createCollateFn<W = Tensor>(
	config: Config,
	dataset: PistonDatasetType,
	maskGenerator: Random | null,
	wrapFunction?: CollateWrapFunction<W> | null
): PistonCollateFnType<W> {
	const isEncoderOnly = config.model.topology === 'encoder';
	const isEncoderDecoder = config.model.topology === 'encoder-decoder';
	const collateOptions = wrapFunction !== undefined ? { wrapFunction } : {};
	if (dataset instanceof NaturalLanguageDataset) {
		if (isEncoderDecoder) {
			throw new Error('Encoder-decoder is not supported for natural language datasets.');
		}
		if (isEncoderOnly) {
			return (batch: number[][]) =>
				naturalLanguageBidirectionalCollate(batch as number[][], {
					maskRatio: config.data.maskRatio,
					generator: maskGenerator!,
					maskTokenId: dataset.maskId! as number,
					...collateOptions
				});
		} else {
			return (batch: number[][]) =>
				naturalLanguageAutoregressiveCollate(batch as number[][], {
					...collateOptions
				});
		}
	} else if (isEncoderOnly) {
		return (batch: ToySequence[]) =>
			toyDatasetBidirectionalCollate(batch as ToySequence[], dataset, {
				maskPrompt: config.data.trainOnPrompt,
				maskRatio: config.data.maskRatio,
				// Derive per-sample RNG from dataset/index internally (stateless)
				...collateOptions
			});
	} else if (isEncoderDecoder) {
		return (batch: ToySequence[]) =>
			toyDatasetEncoderDecoderCollate(batch as ToySequence[], dataset, {
				...collateOptions
			});
	} else {
		return (batch: ToySequence[]) =>
			toyDatasetAutoregressiveCollate(batch as ToySequence[], dataset, {
				ignorePrompt: !config.data.trainOnPrompt,
				...collateOptions
			});
	}
}

export function createDataloader<W = Tensor>(
	config: Config,
	dataset: NaturalLanguageDataset,
	generator: Random,
	wrapFunction?: CollateWrapFunction<W> | null
): [NaturalDataloaderType<W>, NaturalCollateFnType<W>];
export function createDataloader<W = Tensor>(
	config: Config,
	dataset: ToyDatasetLike,
	generator: Random,
	wrapFunction?: CollateWrapFunction<W> | null
): [ToyDataloaderType<W>, ToyCollateFnType<W>];
export function createDataloader<W = Tensor>(
	config: Config,
	dataset: PistonDatasetType,
	generator: Random,
	wrapFunction?: CollateWrapFunction<W> | null
):
	| [NaturalDataloaderType<W>, NaturalCollateFnType<W>]
	| [ToyDataloaderType<W>, ToyCollateFnType<W>];
export function createDataloader<W = Tensor>(
	config: Config,
	dataset: PistonDatasetType,
	generator: Random,
	wrapFunction?: CollateWrapFunction<W> | null
):
	| [NaturalDataloaderType<W>, NaturalCollateFnType<W>]
	| [ToyDataloaderType<W>, ToyCollateFnType<W>] {
	if (dataset instanceof NaturalLanguageDataset) {
		const collateFn = createCollateFn(config, dataset, generator, wrapFunction);
		return [new DataLoader<number[], NaturalBatchType<W>>(dataset, { collateFn }), collateFn];
	} else {
		const toyDataset = dataset as ToyDatasetLike;
		const collateFn = createCollateFn(config, toyDataset, generator, wrapFunction);
		return [new DataLoader<ToySequence, ToyBatchType<W>>(toyDataset, { collateFn }), collateFn];
	}
}

/**
 * Create a model instance based on the configuration
 */
export function createModel(
	config: Config,
	vocabSize: number,
	blockSize: number | { source: number; target: number }
):
	| DecoderTransformer
	| EncoderTransformer
	| EncoderDecoderTransformer
	| RNNDecoder
	| RNNEncoder
	| RNNEncoderDecoder {
	const isEncoderOnly = config.model.topology === 'encoder';
	const isDecoderOnly = config.model.topology === 'decoder';
	const isEncoderDecoder = config.model.topology === 'encoder-decoder';

	if (!isEncoderOnly && !isDecoderOnly && !isEncoderDecoder) {
		throw new Error(
			`Unsupported model type: ${config.model.topology}. Only 'encoder', 'decoder', and 'encoder-decoder' are currently supported.`
		);
	}

	if (isEncoderOnly) {
		if (config.model.family === 'rnn') {
			return new RNNEncoder(config, vocabSize);
		} else {
			return new EncoderTransformer(config, vocabSize, blockSize as number);
		}
	} else if (isEncoderDecoder) {
		const { source, target } = blockSize as { source: number; target: number };
		if (config.model.family === 'rnn') {
			return new RNNEncoderDecoder(config, vocabSize);
		} else {
			return new EncoderDecoderTransformer(config, vocabSize, source, target);
		}
	} else {
		if (config.model.family === 'rnn') {
			return new RNNDecoder(config, vocabSize);
		} else {
			return new DecoderTransformer(config, vocabSize, blockSize as number);
		}
	}
}

export function initializeModel(
	config: Config,
	model:
		| DecoderTransformer
		| EncoderTransformer
		| EncoderDecoderTransformer
		| RNNDecoder
		| RNNEncoder
		| RNNEncoderDecoder
) {
	if (config.model.family === 'rnn') {
		initRNNParameters(model, config);
	} else {
		initTransformerParameters(model, config);
	}
}

export function calculateParameterSum(model: Module): Tensor {
	const sums = model.parameters().map((param) => param.sum());
	return piston.stack(sums).sum();
}

export function countParameters(
	model:
		| DecoderTransformer
		| EncoderTransformer
		| EncoderDecoderTransformer
		| RNNDecoder
		| RNNEncoder
		| RNNEncoderDecoder
): number {
	let totalParams = 0;

	// Walk through all named parameters
	for (const [_, param] of model.namedParameters()) {
		if (param && param.shape) {
			const paramCount = (param.shape as number[]).reduce(
				(acc: number, dim: number) => acc * dim,
				1
			);
			totalParams += paramCount;
		}
	}

	return totalParams;
}

/**
 * Inspect model for a given configuration: count the number of parameters and capture an "index"
 * of the model.
 */
export function inspectModel(config: Config): {
	parameterCount: number;
	modelIndex: IndexState;
	vocabSize: number;
	blockSize: number;
} {
	return weak(
		() => {
			const seed = seedPiston(config);
			const generator = seededRandom(seed);
			const dataset = buildDataset(config, generator, 'train');
			const [dataloader] = createDataloader(config, dataset, generator, tensorWrap);
			const isEncoderDecoder = config.model.topology === 'encoder-decoder';
			const blockSizeOrSizes = calculateBlockSize(config, dataloader);
			const vocabSize = calculateVocabSize(config, dataset);
			const model = createModel(config, vocabSize, blockSizeOrSizes);
			const parameterCount = countParameters(model);

			let indexMode: CaptureIndexMode | null = null;
			try {
				indexMode = new CaptureIndexMode(model);

				// Run the model forward with an input from the dataloader
				if (model instanceof DecoderTransformer || model instanceof RNNDecoder) {
					model.forward(piston.zeros([1, blockSizeOrSizes as number], { dtype: piston.int32 }));
				} else if (model instanceof EncoderTransformer || model instanceof RNNEncoder) {
					model.forward(piston.zeros([1, blockSizeOrSizes as number], { dtype: piston.int32 }));
				} else if (
					model instanceof EncoderDecoderTransformer ||
					model instanceof RNNEncoderDecoder
				) {
					model.forward(
						piston.zeros([1, (blockSizeOrSizes as EncoderDecoderBlockSize).source], {
							dtype: piston.int32
						}),
						piston.zeros([1, (blockSizeOrSizes as EncoderDecoderBlockSize).target], {
							dtype: piston.int32
						})
					);
				}

				console.debug(`Model has ${parameterCount} parameters with vocab size ${vocabSize}`);

				const blockSize = isEncoderDecoder
					? Math.max(
							(blockSizeOrSizes as { source: number; target: number }).source,
							(blockSizeOrSizes as { source: number; target: number }).target
						)
					: (blockSizeOrSizes as number);
				return { parameterCount, vocabSize, blockSize, modelIndex: indexMode!.index };
			} finally {
				indexMode![Symbol.dispose]();
			}
		},
		{
			label: 'inspectModel'
		}
	);
}

export function seedPiston(config: Config) {
	// Set up random number generator
	const seed = parseSeed(
		config.training.randomSeed.present ? config.training.randomSeed.value : undefined
	);

	if (seed !== undefined) {
		piston.seed(seed);
	}

	return seed;
}
