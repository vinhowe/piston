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
import type { NaturalBatchType, NaturalCollateFnType, NaturalDataloaderType } from '../types';

import { type CollateWrapFunction } from '../data';
import { naturalLanguageAutoregressiveCollate, NaturalLanguageDataset } from '../data/natural';
import { buildGPT2Config } from '../model/config';
import { GPT, GPT2_BLOCK_SIZE, GPT2_VOCAB_SIZE } from '../model/gpt';
import { parseSeed } from './random';

// Overloads for strong typing based on dataset kind
export function createCollateFn<W = Tensor>(
	wrapFunction?: CollateWrapFunction<W> | null
): NaturalCollateFnType<W> {
	const collateOptions = wrapFunction !== undefined ? { wrapFunction } : {};
	return (batch: number[][]) =>
		naturalLanguageAutoregressiveCollate(batch as number[][], {
			...collateOptions
		});
}

export function createDataloader<W = Tensor>(
	config: Config,
	dataset: NaturalLanguageDataset,
	wrapFunction?: CollateWrapFunction<W> | null
): [NaturalDataloaderType<W>, NaturalCollateFnType<W>] {
	const collateFn = createCollateFn(wrapFunction);
	return [
		new DataLoader<number[], NaturalBatchType<W>>(dataset, {
			collateFn,
			batchSize: config.training.batchSize
		}),
		collateFn
	];
}

/**
 * Create a model instance based on the configuration
 */
export function createModel(config: Config): GPT {
	return new GPT(buildGPT2Config(config.model.type), config);
}

export function calculateParameterSum(model: Module): Tensor {
	const sums = model.parameters().map((param) => param.sum());
	return piston.stack(sums).sum();
}

export function countParameters(model: GPT): number {
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
	hiddenSize: number;
	mlpIntermediateSize: number;
	modelIndex: IndexState;
	vocabSize: number;
	blockSize: number;
} {
	return weak(
		() => {
			const blockSize = GPT2_BLOCK_SIZE;
			const vocabSize = GPT2_VOCAB_SIZE;
			const model = createModel(config);
			const parameterCount = countParameters(model);
			const hiddenSize = model.config.nEmbd;
			const mlpIntermediateSize = hiddenSize * 4;

			let indexMode: CaptureIndexMode | null = null;
			try {
				indexMode = new CaptureIndexMode(model);

				// Run the model forward with an input from the dataloader
				model.forward(piston.zeros([1, blockSize], { dtype: piston.int32 }));

				console.debug(`Model has ${parameterCount} parameters with vocab size ${vocabSize}`);

				return {
					parameterCount,
					hiddenSize,
					mlpIntermediateSize,
					vocabSize,
					blockSize,
					modelIndex: indexMode!.index
				};
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
