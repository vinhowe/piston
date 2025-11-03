import { IterableDataset } from '@piston-ml/piston-web';
import { MersenneTwister19937, Random } from 'random-js';

import type { ToyTokenizer, ToyValidationMetrics } from './types';

export const BOS = '<bos>';
export const EOS = '<eos>';
export const MASK = '<mask>';

export interface SpecialTokensConfig {
	includeBos: boolean;
	includeEos: boolean;
	includeMask: boolean;
}

export type SpecialTokenSet = {
	bos: string;
	eos?: string;
	mask?: string;
};

export interface ToySequence {
	prompt?: number[];
	target: number[];
	mask?: boolean[];
	// Absolute sample index within this run (for deterministic per-sample behavior)
	absoluteIndex?: number;
}

// Raw format interfaces that preserve prompt/target distinction and show mask tokens
export interface CollatedRawSequence {
	// The full sequence with all tokens visible (including mask tokens, BOS, EOS)
	fullSequence: number[];
	// Original prompt tokens (if any)
	prompt?: number[];
	// Original target tokens
	target?: number[];
	// Which tokens are masked out (-100 in labels)
	ignored?: boolean[];
}

export interface ToyAutoregressiveBatch<T> {
	// Tensor outputs for training
	tensors: [T, T]; // [input, target]
	// Raw format for visualization/debugging
	raw: CollatedRawSequence[];
	// Original uncollated samples
	samples: ToySequence[];
}

export interface ToyBidirectionalBatch<T> {
	tensors: [T, T, T]; // [input, labels, attentionMask]
	raw: CollatedRawSequence[];
	samples: ToySequence[];
}

export interface ToyEncoderDecoderBatch<T> {
	tensors: [T, T, T]; // [encoderInput, decoderInput, decoderTarget]
	raw: CollatedRawSequence[];
	samples: ToySequence[];
}

export interface ToyDatasetLike<DatasetConfig = unknown> extends IterableDataset<ToySequence> {
	readonly config: DatasetConfig;
	readonly tokenizer: ToyTokenizer;
	readonly generator: Random;
	readonly bosId: number | null;
	readonly eosId: number | null;
	readonly maskId: number | null;
	readonly hasCanonicalTargets?: boolean;
	readonly disableValidation?: boolean;
	baseSeed: number;
	readonly datasetName: string;
	generateSequence(): ToySequence;
	generateSequenceAt(index: number): ToySequence;
	computeMetrics(
		completion: number[],
		target: number[] | [number[], boolean[]]
	): ToyValidationMetrics;
}

function hashString32(input: string): number {
	// Simple 32-bit FNV-1a hash
	let hash = 0x811c9dc5;
	for (let i = 0; i < input.length; i++) {
		hash ^= input.charCodeAt(i);
		hash = Math.imul(hash, 0x01000193);
	}
	return hash >>> 0;
}

function mix32(x: number): number {
	x = Math.imul(x ^ (x >>> 16), 0x7feb352d);
	x = Math.imul(x ^ (x >>> 15), 0x846ca68b);
	x = x ^ (x >>> 16);
	return x >>> 0;
}

export function deriveToySampleSeed(
	baseSeed: number,
	datasetName: string,
	index: number,
	scope: string = 'sample'
): number {
	const nameHash = hashString32(datasetName);
	const scopeHash = hashString32(scope);
	const mixed = mix32(baseSeed ^ mix32(nameHash) ^ mix32(index) ^ mix32(scopeHash));
	// Ensure non-zero seed for MT engine
	return mixed >>> 0 || 0x1;
}

abstract class ToyDataset<DatasetConfig>
	extends IterableDataset<ToySequence>
	implements ToyDatasetLike<DatasetConfig>
{
	readonly config: DatasetConfig;
	readonly tokenizer: ToyTokenizer;
	generator: Random;
	public readonly bosId: number | null;
	public readonly eosId: number | null;
	public readonly maskId: number | null;
	public readonly hasCanonicalTargets: boolean = true;
	baseSeed: number;
	public readonly datasetName: string;
	public cursor: number = 0;

	private readonly _originalGenerateSequence: () => ToySequence;

	constructor(
		config: DatasetConfig,
		generator: Random,
		specialTokensConfig: SpecialTokensConfig,
		datasetName: string,
		baseSeed: number
	) {
		super();
		this.config = config;

		// Build vocabulary with special tokens
		const coreVocab = this.buildVocab();
		const specialTokens: string[] = [];

		if (specialTokensConfig.includeBos) specialTokens.push(BOS);
		if (specialTokensConfig.includeEos) specialTokens.push(EOS);
		if (specialTokensConfig.includeMask) specialTokens.push(MASK);

		const fullVocab = [...coreVocab, ...specialTokens];

		this.tokenizer = this.tokenizerFromVocab(fullVocab);
		this.bosId = specialTokensConfig.includeBos ? this.tokenizer.vocab[BOS] : null;
		this.eosId = specialTokensConfig.includeEos ? this.tokenizer.vocab[EOS] : null;
		this.maskId = specialTokensConfig.includeMask ? this.tokenizer.vocab[MASK] : null;
		this.generator = generator;
		this.datasetName = datasetName;
		this.baseSeed = baseSeed;

		// Wrap subclass-defined generateSequence with index-based deterministic generation
		this._originalGenerateSequence = this.generateSequence.bind(this);
		// Replace generateSequence to advance cursor and delegate to generateSequenceAt
		this.generateSequence = () => this.generateSequenceAt(this.cursor++);
	}

	protected abstract buildVocab(): string[];
	public abstract generateSequence(): ToySequence;
	public abstract computeMetrics(
		completion: number[],
		target: number[] | [number[], boolean[]]
	): ToyValidationMetrics;

	private tokenizerFromVocab(vocab: string[]): ToyTokenizer {
		const vocabMap = vocab.reduce(
			(acc, token, index) => {
				acc[token] = index;
				return acc;
			},
			{} as Record<string, number>
		);
		const ids = vocab.reduce(
			(acc, token, index) => {
				acc[index] = token;
				return acc;
			},
			{} as Record<number, string>
		);
		const tokenizer: ToyTokenizer = {
			vocab: vocabMap,
			ids,
			lastToken: vocab.length - 1,
			decode: (tokens) => tokens.map((token) => ids[token] ?? '<unk>').join(' ')
		};

		return tokenizer;
	}

	public getItem(_index: number): ToySequence {
		return this.generateSequence();
	}

	public [Symbol.iterator](): Iterator<ToySequence> {
		return {
			next: () => {
				const sequence = this.generateSequence();
				return { value: sequence, done: false };
			}
		};
	}

	/**
	 * Generate a deterministic sequence at an absolute index without affecting cursor or RNG state elsewhere.
	 */
	public generateSequenceAt(index: number): ToySequence {
		const seed = deriveToySampleSeed(this.baseSeed, this.datasetName, index, 'toy-sample');
		const rng = new Random(MersenneTwister19937.seed(seed));
		const previous = this.generator;
		try {
			this.generator = rng;
			const seq = this._originalGenerateSequence();
			return { ...seq, absoluteIndex: index };
		} finally {
			this.generator = previous;
		}
	}
}

export function tokenMatches(completion: number[], target: number[]): boolean[] {
	if (completion.length !== target.length) return completion.map(() => false);
	return completion.map((t, i) => t === target[i]);
}

export function mustMatchAccuracy(completion: number[], target: number[]): number {
	return completion.every((t, i) => t === target[i]) ? 1 : 0;
}

export default ToyDataset;
