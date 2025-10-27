import type { Config } from '$lib/workspace/config';
import type { Tensor } from '@piston-ml/piston-web';
import type { Random } from 'random-js';

import type { PistonDatasetType } from '../types';

import { buildDataset, tensorWrap } from '.';
import { createDataloader } from '../utils/model';

export type BuiltData = {
	dataset: PistonDatasetType;
	iterator: AsyncIterator<{
		readonly tensors: Tensor[];
		readonly samples?: unknown[];
	}>;
};

export async function buildDataPipeline(
	config: Config,
	generator: Random,
	trainDatasetOverride?: PistonDatasetType
): Promise<BuiltData> {
	const trainDataset = trainDatasetOverride ?? buildDataset(config, generator, 'train');
	const [trainDataloader] = createDataloader(config, trainDataset, generator, tensorWrap);

	return {
		dataset: trainDataset,
		iterator: trainDataloader[Symbol.asyncIterator]()
	};
}
