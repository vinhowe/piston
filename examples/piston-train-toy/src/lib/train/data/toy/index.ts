import type { Config } from '$lib/workspace/config';
import type { Random } from 'random-js';

import type ToyDataset from './dataset';
import type { SpecialTokensConfig } from './dataset';

import { type AdditionConfig, AdditionDataset } from './addition';
import { type CopyMemoryConfig, CopyMemoryDataset } from './copyMemory';
import { type DyckConfig, DyckDataset } from './dyck';
import { type ElmanConfig, ElmanDataset } from './elman';
import { type MarkedAdditionConfig, MarkedAdditionDataset } from './markedAddition';
import { type ModularAdditionConfig, ModularAdditionDataset } from './modularAddition';
import { type ParityConfig, ParityDataset } from './parity';
import { type RandomConfig, RandomDataset } from './random';
import { type RepeatConfig, RepeatDataset } from './repeat';
import { type ReverseConfig, ReverseDataset } from './reverse';
import { type SlapjackConfig, SlapjackDataset } from './slapjack';
import { type SortConfig, SortDataset } from './sort';
import { type TemporalOrderConfig, TemporalOrderDataset } from './temporalOrder';
import { type TwoSumConfig, TwoSumDataset } from './twoSum';
import { type ZerosConfig, ZerosDataset } from './zeros';
export { type AdditionConfig, AdditionDataset } from './addition';
export { type CopyMemoryConfig, CopyMemoryDataset } from './copyMemory';
export { default as ToyDataset } from './dataset';
export { type DyckConfig, DyckDataset } from './dyck';
export { type ElmanConfig, ElmanDataset } from './elman';
export { type MarkedAdditionConfig, MarkedAdditionDataset } from './markedAddition';
export { type ModularAdditionConfig, ModularAdditionDataset } from './modularAddition';
export { type RepeatConfig, RepeatDataset } from './repeat';
export { type ReverseConfig, ReverseDataset } from './reverse';
export { type SlapjackConfig, SlapjackDataset } from './slapjack';
export { type SortConfig, SortDataset } from './sort';
export { type TemporalOrderConfig, TemporalOrderDataset } from './temporalOrder';
export { type TwoSumConfig, TwoSumDataset } from './twoSum';

export interface DatasetConfigs {
	'two-sum': TwoSumConfig;
	sort: SortConfig;
	repeat: RepeatConfig;
	reverse: ReverseConfig;
	addition: AdditionConfig;
	'modular-addition': ModularAdditionConfig;
	zeros: ZerosConfig;
	slapjack: SlapjackConfig;
	dyck: DyckConfig;
	parity: ParityConfig;
	random: RandomConfig;
	'marked-addition': MarkedAdditionConfig;
	'copy-memory': CopyMemoryConfig;
	'temporal-order': TemporalOrderConfig;
	elman: ElmanConfig;
}

export function buildToyDataset(config: Config, generator: Random): ToyDataset<unknown> {
	const datasetName = config.data.dataset;
	// Derive a baseSeed for deterministic per-sample generation; if the run is seeded,
	// the provided generator will be seeded deterministically. If not, we still persist
	// baseSeed in checkpoints for resumption.
	const baseSeed = generator.int32() >>> 0;

	const isEncoderOnly = config.model.topology === 'encoder';
	const specialTokensConfig: SpecialTokensConfig = {
		includeBos: !isEncoderOnly,
		includeEos: !isEncoderOnly && config.data.specialTokens.includeEos,
		includeMask: isEncoderOnly
	};

	if (datasetName === 'sort') {
		const sortConfig = config.data.datasets.sort;
		return new SortDataset(sortConfig, generator, specialTokensConfig, datasetName, baseSeed);
	} else if (datasetName === 'repeat') {
		const repeatConfig = config.data.datasets.repeat;
		return new RepeatDataset(repeatConfig, generator, specialTokensConfig, datasetName, baseSeed);
	} else if (datasetName === 'reverse') {
		const reverseConfig = config.data.datasets.reverse;
		return new ReverseDataset(reverseConfig, generator, specialTokensConfig, datasetName, baseSeed);
	} else if (datasetName === 'addition') {
		const additionConfig = config.data.datasets.addition;
		return new AdditionDataset(
			additionConfig,
			generator,
			specialTokensConfig,
			datasetName,
			baseSeed
		);
	} else if (datasetName === 'modular-addition') {
		const modularAdditionConfig = config.data.datasets['modular-addition'];
		return new ModularAdditionDataset(
			modularAdditionConfig,
			generator,
			specialTokensConfig,
			datasetName,
			baseSeed
		);
	} else if (datasetName === 'two-sum') {
		const twoSumConfig = config.data.datasets['two-sum'];
		return new TwoSumDataset(twoSumConfig, generator, specialTokensConfig, datasetName, baseSeed);
	} else if (datasetName === 'zeros') {
		const zerosConfig = config.data.datasets.zeros;
		return new ZerosDataset(zerosConfig, generator, specialTokensConfig, datasetName, baseSeed);
	} else if (datasetName === 'slapjack') {
		const slapjackConfig = config.data.datasets.slapjack;
		return new SlapjackDataset(
			slapjackConfig,
			generator,
			specialTokensConfig,
			datasetName,
			baseSeed
		);
	} else if (datasetName === 'dyck') {
		const dyckConfig = config.data.datasets.dyck;
		return new DyckDataset(dyckConfig, generator, specialTokensConfig, datasetName, baseSeed);
	} else if (datasetName === 'parity') {
		const parityConfig = config.data.datasets.parity;
		return new ParityDataset(parityConfig, generator, specialTokensConfig, datasetName, baseSeed);
	} else if (datasetName === 'random') {
		const randomConfig = config.data.datasets.random;
		return new RandomDataset(randomConfig, generator, specialTokensConfig, datasetName, baseSeed);
	} else if (datasetName === 'marked-addition') {
		const markedAdditionConfig = config.data.datasets['marked-addition'];
		return new MarkedAdditionDataset(
			markedAdditionConfig,
			generator,
			specialTokensConfig,
			datasetName,
			baseSeed
		);
	} else if (datasetName === 'temporal-order') {
		const cfg = config.data.datasets['temporal-order'];
		return new TemporalOrderDataset(cfg, generator, specialTokensConfig, datasetName, baseSeed);
	} else if (datasetName === 'copy-memory') {
		const cfg = config.data.datasets['copy-memory'];
		return new CopyMemoryDataset(cfg, generator, specialTokensConfig, datasetName, baseSeed);
	} else if (datasetName === 'elman') {
		return new ElmanDataset(null, generator, specialTokensConfig, datasetName, baseSeed);
	} else {
		throw new Error(`Unknown dataset: ${datasetName}`);
	}
}
