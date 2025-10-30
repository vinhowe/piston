import type { ToyValidationMetrics } from './types';

import ToyDataset, { mustMatchAccuracy, tokenMatches, type ToySequence } from './dataset';

export interface ZerosConfig {
	sequenceLength: number;
}

export const ZEROS_CONFIG_METADATA = {
	name: 'Zeros',
	description:
		'Output only zeros (useful test case; if a model fails to learn this, something is wrong)',
	supportsModelTypes: ['encoder', 'encoder-decoder', 'decoder'],
	disableValidation: true,
	parameters: {
		sequenceLength: {
			name: 'Sequence Length',
			description: 'Length of the sequence',
			type: 'number' as const,
			min: 2,
			max: 10,
			default: 5
		}
	}
} as const;

export const ZEROS_CONFIG_DEFAULTS: ZerosConfig = {
	sequenceLength: ZEROS_CONFIG_METADATA.parameters.sequenceLength.default
};

export class ZerosDataset extends ToyDataset<ZerosConfig> {
	/**
	 * For zeros dataset (baseline/debug task):
	 */
	protected buildVocab(): string[] {
		// We have at least two tokens so this is actually technically a classification task
		return '01'.split('');
	}

	public readonly disableValidation = true;

	/**
	 * Zeros: (Baseline) returns a sequence of zeros.
	 * prompt: single '0'
	 * target: sequence of zeros (sequenceLength - 1)
	 * Example: 0:0000
	 */
	public generateSequence(): ToySequence {
		const { sequenceLength } = this.config;
		const zeroToken = this.tokenizer.vocab['0'];

		// Target is sequenceLength zeros
		const target = Array(sequenceLength).fill(zeroToken);

		return { target };
	}

	public computeMetrics(completion: number[], target: number[]): ToyValidationMetrics {
		return {
			matches: tokenMatches(completion, target),
			accuracy: mustMatchAccuracy(completion, target)
		};
	}
}
