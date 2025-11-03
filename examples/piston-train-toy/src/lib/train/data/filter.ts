import { AsyncIterableDataset, IterableDataset } from '@piston-ml/piston-web';

import type { PistonDatasetType } from '../types';
import type { ToyDatasetLike, ToySequence } from './toy/dataset';

import { NaturalLanguageDataset } from './natural/dataset';

// Custom error class for low-diversity dataset so we can catch it and display a custom message
export class LowDiversityDatasetError extends Error {
	constructor(consecutiveSkips: number) {
		const message =
			`FilteredPistonDataset: skipped ${consecutiveSkips} consecutive examples because they ` +
			'collide with validation set; not enough diversity in the training dataset. Consider ' +
			'changing seed or dataset parameters.';
		super(message);
		this.name = 'LowDiversityDatasetError';
	}
}

export function buildToySequenceKeyer(datasetName: string): (seq: ToySequence) => string {
	return (seq) => {
		const prompt = seq.prompt ?? [];
		const mask = seq.mask ?? [];
		return [
			`dataset=${datasetName}`,
			`promptLength=${prompt.length}`,
			`targetLength=${seq.target.length}`,
			`maskLength=${mask.length}`,
			`prompt=${prompt.join(',')}`,
			`target=${seq.target.join(',')}`,
			`mask=${mask.join(',')}`
		].join('|');
	};
}

export function buildNaturalSequenceKeyer(datasetName: string): (seq: number[]) => string {
	return (seq) => [`dataset=${datasetName}`, `len=${seq.length}`, `seq=${seq.join(',')}`].join('|');
}

export interface FilterOptions<T> {
	sequenceKeyer: (sample: T) => string;
	maxConsecutiveSkips?: number;
}

export function makeFilteredIterableDataset<T, B extends IterableDataset<T>>(
	base: B,
	disallowed: Set<string>,
	options: FilterOptions<T>
): B {
	const maxSkips = options.maxConsecutiveSkips ?? 10;
	const sequenceKeyer = options.sequenceKeyer;

	const handler: ProxyHandler<B> = {
		get(target, prop, receiver) {
			if (prop === Symbol.iterator) {
				return function (this: unknown) {
					const iter = target[Symbol.iterator]();
					let consecutiveSkips = 0;
					return {
						next(): IteratorResult<T> {
							for (;;) {
								const n = iter.next();
								if (n.done) return n;
								const sample = n.value;
								const key = sequenceKeyer(sample);
								if (!disallowed.has(key)) {
									consecutiveSkips = 0;
									return { value: sample, done: false };
								}
								consecutiveSkips++;
								if (consecutiveSkips >= maxSkips) {
									throw new LowDiversityDatasetError(consecutiveSkips);
								}
							}
						}
					};
				};
			}
			return Reflect.get(target, prop, receiver);
		}
	};

	return new Proxy(base, handler);
}

export function makeFilteredAsyncIterableDataset<T, B extends AsyncIterableDataset<T>>(
	base: B,
	disallowed: Set<string>,
	options: FilterOptions<T>
): B {
	const maxSkips = options.maxConsecutiveSkips ?? 10;
	const sequenceKeyer = options.sequenceKeyer;

	const handler: ProxyHandler<B> = {
		get(target, prop, receiver) {
			if (prop === Symbol.asyncIterator) {
				return function (this: unknown) {
					const iter = target[Symbol.asyncIterator]();
					let consecutiveSkips = 0;
					return {
						async next(): Promise<IteratorResult<T>> {
							for (;;) {
								const n = await iter.next();
								if (n.done) return n;
								const sample = n.value;
								const key = sequenceKeyer(sample);
								if (!disallowed.has(key)) {
									consecutiveSkips = 0;
									return { value: sample, done: false };
								}
								consecutiveSkips++;
								if (consecutiveSkips >= maxSkips) {
									throw new LowDiversityDatasetError(consecutiveSkips);
								}
							}
						}
					};
				};
			}
			return Reflect.get(target, prop, receiver);
		}
	};

	return new Proxy(base, handler);
}

export function filterDatasetByHeldoutSamples(
	dataset: PistonDatasetType,
	datasetName: string,
	samples: ReadonlyArray<ToySequence> | ReadonlyArray<number[]>
): PistonDatasetType {
	if (dataset instanceof NaturalLanguageDataset) {
		const sequenceKeyer = buildNaturalSequenceKeyer(datasetName);
		const disallowed = new Set((samples as ReadonlyArray<number[]>).map(sequenceKeyer));

		return makeFilteredAsyncIterableDataset(dataset, disallowed, { sequenceKeyer });
	} else {
		const sequenceKeyer = buildToySequenceKeyer(datasetName);
		const disallowed = new Set((samples as ReadonlyArray<ToySequence>).map(sequenceKeyer));

		return makeFilteredIterableDataset<ToySequence, ToyDatasetLike>(dataset, disallowed, {
			sequenceKeyer
		});
	}
}
