import type { ToyCollateFnType } from '$lib/train/types';

import { gpu, int32, tensor, Tensor } from '@piston-ml/piston-web';

import type { CollatedRawSequence, ToyDatasetLike, ToySequence } from './toy/dataset';

export type CollateWrapFunction<T> = (sequences: number[][]) => T;

export function tensorWrap(sequences: number[][]): Tensor {
	return tensor(sequences, { dtype: int32, device: gpu });
}

// Utility functions for accessing raw format data
export function getCollatedSampleData<T>(
	dataset: ToyDatasetLike,
	collateFn: ToyCollateFnType<T>,
	sampleCount: number = 4
): {
	samples: ToySequence[] | number[][];
	collated: CollatedRawSequence[];
} {
	// Generate sample sequences
	const iterator = dataset[Symbol.iterator]();
	const samples = Array.from({ length: sampleCount }, () => iterator.next().value);
	const collated = (collateFn as ToyCollateFnType<T>)(samples as ToySequence[]).raw;

	return { samples, collated };
}
