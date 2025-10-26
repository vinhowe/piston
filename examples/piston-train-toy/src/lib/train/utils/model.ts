import type { Random } from 'random-js';

import { DataLoader, Tensor } from '@piston-ml/piston-web';

import type { Config } from '../../workspace/config';
import type { ToyBatchType, ToyCollateFnType, ToyDataloaderType } from '../types';

import { type CollateWrapFunction } from '../data';
import {
	toyDatasetAutoregressiveCollate,
	toyDatasetBidirectionalCollate,
	toyDatasetEncoderDecoderCollate
} from '../data/toy/collate';
import { type ToyDatasetLike, type ToySequence } from '../data/toy/dataset';
import {
	DecoderTransformer,
	EncoderDecoderTransformer,
	EncoderTransformer
} from '../model/transformer';

type EncoderDecoderBlockSize = { source: number; target: number };

/**
 * Calculate the vocabulary size based on the dataset and configuration
 */
export function calculateVocabSize(dataset: ToyDatasetLike): number {
	return 'vocabSize' in dataset
		? (dataset.vocabSize as number)
		: Object.keys(dataset.tokenizer.vocab).length;
}

export function calculateBlockSize<T>(
	config: Config,
	dataloader: ToyDataloaderType<T>
): number | EncoderDecoderBlockSize {
	// Infer lengths without advancing iteration using deterministic index-based generation
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
	dataset: ToyDatasetLike,
	maskGenerator: Random | null,
	wrapFunction?: CollateWrapFunction<W> | null
): ToyCollateFnType<W> {
	const isEncoderOnly = config.model.topology === 'encoder';
	const isEncoderDecoder = config.model.topology === 'encoder-decoder';
	const collateOptions = wrapFunction !== undefined ? { wrapFunction } : {};
	if (isEncoderOnly) {
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
	dataset: ToyDatasetLike,
	generator: Random,
	wrapFunction?: CollateWrapFunction<W> | null
): [ToyDataloaderType<W>, ToyCollateFnType<W>] {
	const toyDataset = dataset as ToyDatasetLike;
	const collateFn = createCollateFn(config, toyDataset, generator, wrapFunction);
	return [new DataLoader<ToySequence, ToyBatchType<W>>(toyDataset, { collateFn }), collateFn];
}

/**
 * Create a model instance based on the configuration
 */
export function createModel(
	config: Config,
	vocabSize: number,
	blockSize: number | { source: number; target: number }
): DecoderTransformer | EncoderTransformer | EncoderDecoderTransformer {
	const isEncoderOnly = config.model.topology === 'encoder';
	const isDecoderOnly = config.model.topology === 'decoder';
	const isEncoderDecoder = config.model.topology === 'encoder-decoder';

	if (!isEncoderOnly && !isDecoderOnly && !isEncoderDecoder) {
		throw new Error(
			`Unsupported model type: ${config.model.topology}. Only 'encoder', 'decoder', and 'encoder-decoder' are currently supported.`
		);
	}

	if (isEncoderOnly) {
		return new EncoderTransformer(config, vocabSize, blockSize as number);
	} else if (isEncoderDecoder) {
		const { source, target } = blockSize as { source: number; target: number };
		return new EncoderDecoderTransformer(config, vocabSize, source, target);
	} else {
		return new DecoderTransformer(config, vocabSize, blockSize as number);
	}
}
