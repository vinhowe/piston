import type { Config } from '$lib/workspace/config';
import type { Random } from 'random-js';

import type { PistonDatasetType } from '../types';

import {
	buildNaturalLanguageDataset,
	NATURAL_DATASET_META,
	type NaturalDatasetName
} from './natural';
import { buildToyDataset } from './toy';
import { TOY_DATASET_CONFIG_DEFAULTS, TOY_DATASET_CONFIG_METADATA } from './toy/config';

export const DATASET_CONFIG_METADATA = {
	...Object.fromEntries(
		Object.entries(NATURAL_DATASET_META).map(([name, meta]) => [
			name,
			{
				...meta,
				supportsModelTypes: ['encoder', 'decoder'],
				type: 'natural'
			}
		])
	),
	...Object.fromEntries(
		Object.entries(TOY_DATASET_CONFIG_METADATA).map(([name, meta]) => [
			name,
			{ ...meta, type: 'toy' }
		])
	)
} as const;

export const DATASET_CONFIG_DEFAULTS = {
	...TOY_DATASET_CONFIG_DEFAULTS,
	...(Object.fromEntries(
		Object.entries(NATURAL_DATASET_META).map(([name]) => [name, {}])
	) as Record<NaturalDatasetName, unknown>)
} as const;

export function buildDataset(
	config: Config,
	generator: Random,
	split: 'train' | 'val'
): PistonDatasetType {
	if (Object.keys(NATURAL_DATASET_META).includes(config.data.dataset)) {
		return buildNaturalLanguageDataset(split, config);
	}
	return buildToyDataset(config, generator);
}

export * from './collate';
