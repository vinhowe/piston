import type { Config } from '$lib/workspace/config';
import type { Tensor } from '@piston-ml/piston-web';
import type { Random } from 'random-js';

import type { PistonCollateFnType, PistonDatasetType } from '../types';

import { buildDataset, tensorWrap } from '.';
import { createCollateFn, createDataloader } from '../utils/model';
import { forkRandom } from '../utils/random';

export type BuiltData = {
	train: {
		dataset: PistonDatasetType;
		iterator: AsyncIterator<{
			readonly tensors: Tensor[];
			readonly samples?: unknown[];
		}>;
	};
	validation?: {
		dataset: PistonDatasetType;
		collateFn: PistonCollateFnType<Tensor>;
	};
};

export async function buildDataPipeline(
	config: Config,
	generator: Random,
	maskGenerator: Random | null,
	trainDatasetOverride?: PistonDatasetType
): Promise<BuiltData> {
	const trainDataset = trainDatasetOverride ?? buildDataset(config, generator, 'train');
	const [trainDataloader] = createDataloader(config, trainDataset, generator, tensorWrap);

	let validation: BuiltData['validation'] | undefined;
	if (config.training.validation.present) {
		const validationGenerator = forkRandom(generator);
		const validationDataset = buildDataset(config, validationGenerator, 'val');
		const collateFn = createCollateFn(config, validationDataset, maskGenerator, tensorWrap);
		validation = { dataset: validationDataset, collateFn };
	}

	return {
		train: {
			dataset: trainDataset,
			iterator: trainDataloader[Symbol.asyncIterator]()
		},
		validation
	};
}
