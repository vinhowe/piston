import type {
	NaturalCollateFnType,
	PistonCollateFnType,
	PistonDatasetType,
	ToyCollateFnType
} from '$lib/train/types';

import { gpu, int32, tensor, Tensor } from '@piston-ml/piston-web';

import type { CollatedRawSequence, ToySequence } from './toy/dataset';

import { NaturalLanguageDataset } from './natural/dataset';
import ToyDataset from './toy/dataset';

export type CollateWrapFunction<T> = (sequences: number[][]) => T;

export function tensorWrap(sequences: number[][]): Tensor {
	return tensor(sequences, { dtype: int32, device: gpu });
}

// Utility functions for accessing raw format data
export async function getCollatedSampleData<T>(
	dataset: PistonDatasetType,
	collateFn: PistonCollateFnType<T>,
	sampleCount: number = 4
): Promise<{
	samples: ToySequence[] | number[][];
	collated: CollatedRawSequence[];
}> {
	let collated: CollatedRawSequence[];
	let samples: ToySequence[] | number[][];

	if (dataset instanceof ToyDataset) {
		const iterator = dataset[Symbol.iterator]();
		samples = Array.from({ length: sampleCount }, () => iterator.next().value);
		collated = (collateFn as ToyCollateFnType<T>)(samples as ToySequence[]).raw;
	} else if (dataset instanceof NaturalLanguageDataset) {
		const iterator = dataset[Symbol.asyncIterator]();
		samples = [];
		for (let i = 0; i < sampleCount; i++) {
			const sample = await iterator.next();
			samples.push(sample.value);
		}
		collated = (collateFn as NaturalCollateFnType<T>)(samples as number[][]).samples.map((s) => ({
			fullSequence: s
		}));
	} else {
		throw new Error('Unsupported dataset type');
	}

	return { samples, collated };
}
