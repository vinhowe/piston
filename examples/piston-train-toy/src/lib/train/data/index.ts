import { TOY_DATASET_CONFIG_DEFAULTS, TOY_DATASET_CONFIG_METADATA } from './toy/config';

export const DATASET_CONFIG_METADATA = {
	...Object.fromEntries(
		Object.entries(TOY_DATASET_CONFIG_METADATA).map(([name, meta]) => [
			name,
			{ ...meta, type: 'toy' }
		])
	)
} as const;

export const DATASET_CONFIG_DEFAULTS = TOY_DATASET_CONFIG_DEFAULTS;

export * from './collate';
