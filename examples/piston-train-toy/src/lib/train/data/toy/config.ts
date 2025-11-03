// Shared configuration parameter interface
export interface ConfigParameter {
	name: string;
	description?: string;
	type: 'number' | 'boolean';
	min?: number;
	max?: number;
	step?: number;
	default: number | boolean;
}

// Import all dataset configurations
import { ADDITION_CONFIG_DEFAULTS, ADDITION_CONFIG_METADATA } from './addition';
import { COPY_MEMORY_CONFIG_DEFAULTS, COPY_MEMORY_CONFIG_METADATA } from './copyMemory';
import { DYCK_CONFIG_DEFAULTS, DYCK_CONFIG_METADATA } from './dyck';
import { ELMAN_CONFIG_DEFAULTS, ELMAN_CONFIG_METADATA } from './elman';
import { MARKED_ADDITION_CONFIG_DEFAULTS, MARKED_ADDITION_CONFIG_METADATA } from './markedAddition';
import {
	MODULAR_ADDITION_CONFIG_DEFAULTS,
	MODULAR_ADDITION_CONFIG_METADATA
} from './modularAddition';
import { PARITY_CONFIG_DEFAULTS, PARITY_CONFIG_METADATA } from './parity';
import { RANDOM_CONFIG_DEFAULTS, RANDOM_CONFIG_METADATA } from './random';
import { REPEAT_CONFIG_DEFAULTS, REPEAT_CONFIG_METADATA } from './repeat';
import { REVERSE_CONFIG_DEFAULTS, REVERSE_CONFIG_METADATA } from './reverse';
import { SLAPJACK_CONFIG_DEFAULTS, SLAPJACK_CONFIG_METADATA } from './slapjack';
import { SORT_CONFIG_DEFAULTS, SORT_CONFIG_METADATA } from './sort';
import { TEMPORAL_ORDER_CONFIG_DEFAULTS, TEMPORAL_ORDER_CONFIG_METADATA } from './temporalOrder';
import { TWO_SUM_CONFIG_DEFAULTS, TWO_SUM_CONFIG_METADATA } from './twoSum';
import { ZEROS_CONFIG_DEFAULTS, ZEROS_CONFIG_METADATA } from './zeros';

// Re-export types and constants
export type { AdditionConfig } from './addition';
export { ADDITION_CONFIG_DEFAULTS, ADDITION_CONFIG_METADATA } from './addition';

export type { CopyMemoryConfig } from './copyMemory';
export { COPY_MEMORY_CONFIG_DEFAULTS, COPY_MEMORY_CONFIG_METADATA } from './copyMemory';

export type { DyckConfig } from './dyck';
export { DYCK_CONFIG_DEFAULTS, DYCK_CONFIG_METADATA } from './dyck';

export { ELMAN_CONFIG_DEFAULTS, ELMAN_CONFIG_METADATA } from './elman';

export type { MarkedAdditionConfig } from './markedAddition';
export { MARKED_ADDITION_CONFIG_DEFAULTS, MARKED_ADDITION_CONFIG_METADATA } from './markedAddition';

export type { ModularAdditionConfig } from './modularAddition';
export {
	MODULAR_ADDITION_CONFIG_DEFAULTS,
	MODULAR_ADDITION_CONFIG_METADATA
} from './modularAddition';

export type { ParityConfig } from './parity';
export { PARITY_CONFIG_DEFAULTS, PARITY_CONFIG_METADATA } from './parity';

export type { RandomConfig } from './random';
export { RANDOM_CONFIG_DEFAULTS, RANDOM_CONFIG_METADATA } from './random';

export type { RepeatConfig } from './repeat';
export { REPEAT_CONFIG_DEFAULTS, REPEAT_CONFIG_METADATA } from './repeat';

export type { ReverseConfig } from './reverse';
export { REVERSE_CONFIG_DEFAULTS, REVERSE_CONFIG_METADATA } from './reverse';

export type { SlapjackConfig } from './slapjack';
export { SLAPJACK_CONFIG_DEFAULTS, SLAPJACK_CONFIG_METADATA } from './slapjack';

export type { SortConfig } from './sort';
export { SORT_CONFIG_DEFAULTS, SORT_CONFIG_METADATA } from './sort';

export type { TemporalOrderConfig } from './temporalOrder';
export { TEMPORAL_ORDER_CONFIG_DEFAULTS, TEMPORAL_ORDER_CONFIG_METADATA } from './temporalOrder';

export type { TwoSumConfig } from './twoSum';
export { TWO_SUM_CONFIG_DEFAULTS, TWO_SUM_CONFIG_METADATA } from './twoSum';

export type { ZerosConfig } from './zeros';
export { ZEROS_CONFIG_DEFAULTS, ZEROS_CONFIG_METADATA } from './zeros';

// Aggregate metadata for all datasets
export const TOY_DATASET_CONFIG_METADATA = {
	addition: ADDITION_CONFIG_METADATA,
	'copy-memory': COPY_MEMORY_CONFIG_METADATA,
	dyck: DYCK_CONFIG_METADATA,
	elman: ELMAN_CONFIG_METADATA,
	'marked-addition': MARKED_ADDITION_CONFIG_METADATA,
	'modular-addition': MODULAR_ADDITION_CONFIG_METADATA,
	parity: PARITY_CONFIG_METADATA,
	random: RANDOM_CONFIG_METADATA,
	repeat: REPEAT_CONFIG_METADATA,
	reverse: REVERSE_CONFIG_METADATA,
	slapjack: SLAPJACK_CONFIG_METADATA,
	sort: SORT_CONFIG_METADATA,
	'temporal-order': TEMPORAL_ORDER_CONFIG_METADATA,
	'two-sum': TWO_SUM_CONFIG_METADATA,
	zeros: ZEROS_CONFIG_METADATA
} as const;

// Aggregate defaults for all datasets
export const TOY_DATASET_CONFIG_DEFAULTS = {
	addition: ADDITION_CONFIG_DEFAULTS,
	'copy-memory': COPY_MEMORY_CONFIG_DEFAULTS,
	dyck: DYCK_CONFIG_DEFAULTS,
	elman: ELMAN_CONFIG_DEFAULTS,
	'marked-addition': MARKED_ADDITION_CONFIG_DEFAULTS,
	'modular-addition': MODULAR_ADDITION_CONFIG_DEFAULTS,
	parity: PARITY_CONFIG_DEFAULTS,
	random: RANDOM_CONFIG_DEFAULTS,
	repeat: REPEAT_CONFIG_DEFAULTS,
	reverse: REVERSE_CONFIG_DEFAULTS,
	slapjack: SLAPJACK_CONFIG_DEFAULTS,
	sort: SORT_CONFIG_DEFAULTS,
	'temporal-order': TEMPORAL_ORDER_CONFIG_DEFAULTS,
	'two-sum': TWO_SUM_CONFIG_DEFAULTS,
	zeros: ZEROS_CONFIG_DEFAULTS
} as const;
