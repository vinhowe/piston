import type { Config } from '$lib/workspace/config';
import type { Tensor } from '@piston-ml/piston-web';
import type { Random } from 'random-js';

import type { ToyDatasetLike } from './toy/dataset';

import { tensorWrap } from '.';
import { createDataloader } from '../utils/model';
import { buildToyDataset } from './toy';

export type BuiltData = {
	dataset: ToyDatasetLike;
	iterator: AsyncIterator<{
		readonly tensors: Tensor[];
		readonly samples?: unknown[];
	}>;
};

export async function buildDataPipeline(
	config: Config,
	generator: Random,
	trainDatasetOverride?: ToyDatasetLike
): Promise<BuiltData> {
	const trainDataset = trainDatasetOverride ?? buildToyDataset(config, generator);
	const [trainDataloader] = createDataloader(config, trainDataset, generator, tensorWrap);

	return {
		dataset: trainDataset,
		iterator: trainDataloader[Symbol.asyncIterator]()
	};
}
