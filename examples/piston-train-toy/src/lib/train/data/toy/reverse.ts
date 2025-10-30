import type { ToyValidationMetrics } from './types';

import ToyDataset, { mustMatchAccuracy, tokenMatches, type ToySequence } from './dataset';

export interface ReverseConfig {
	sequenceLength: number;
	maxNumber: number;
	includeCommas: boolean;
	includeColon: boolean;
}

export const REVERSE_CONFIG_METADATA = {
	name: 'Reverse',
	description: 'Reverse a sequence of characters',
	supportsModelTypes: ['encoder', 'encoder-decoder', 'decoder'],
	parameters: {
		sequenceLength: {
			name: 'Sequence Length',
			description: 'Number of items to reverse',
			type: 'number' as const,
			min: 1,
			max: 10,
			default: 3
		},
		maxNumber: {
			name: 'Max Alphabet Character',
			description: 'Maximum alphabet character',
			type: 'number' as const,
			min: 5,
			max: 26,
			default: 5
		},
		includeCommas: {
			name: 'Include Commas',
			type: 'boolean' as const,
			default: false
		},
		includeColon: {
			name: 'Include Colon',
			type: 'boolean' as const,
			default: true
		}
	}
} as const;

export const REVERSE_CONFIG_DEFAULTS: ReverseConfig = {
	sequenceLength: REVERSE_CONFIG_METADATA.parameters.sequenceLength.default,
	maxNumber: REVERSE_CONFIG_METADATA.parameters.maxNumber.default,
	includeCommas: REVERSE_CONFIG_METADATA.parameters.includeCommas.default,
	includeColon: REVERSE_CONFIG_METADATA.parameters.includeColon.default
};

export class ReverseDataset extends ToyDataset<ReverseConfig> {
	/**
	 * For reverse dataset:
	 * - tokens 0 .. maxNum represent the alphabet characters corresponding to the numbers 0 .. maxNum.
	 * - token maxNum+1 -> ':'
	 * - token maxNum+2 -> ','
	 */
	protected buildVocab(): string[] {
		const { maxNumber, includeColon, includeCommas } = this.config;
		const vocab: string[] = [];
		const aCharCode = 'A'.charCodeAt(0);
		for (let n = 0; n < maxNumber; n++) {
			const token = String.fromCharCode(aCharCode + n);
			vocab.push(token);
		}
		if (includeColon) {
			vocab.push(':');
		}
		if (includeCommas) {
			vocab.push(',');
		}
		return vocab;
	}

	/**
	 * Reverse: generate a sequence of characters and return:
	 * - prompt: the sequence with commas (token 1) between numbers and a colon (token 0) at the end.
	 * - target: the reversed sequence with commas between numbers.
	 * Example: BBCA:ACBB
	 */
	public generateSequence(): ToySequence {
		const { sequenceLength, maxNumber, includeCommas, includeColon } = this.config;
		const nums = Array.from({ length: sequenceLength }, () =>
			this.generator.integer(0, maxNumber - 1)
		);
		const prompt: number[] = [];

		const comma = this.tokenizer.vocab[','];
		const colon = this.tokenizer.vocab[':'];

		for (let i = 0; i < nums.length; i++) {
			prompt.push(nums[i]);
			if (includeCommas && i < nums.length - 1) {
				prompt.push(comma);
			}
		}
		if (includeColon) {
			prompt.push(colon);
		}

		const target: number[] = [];
		const reversedNums = nums.reverse();
		for (let i = 0; i < reversedNums.length; i++) {
			target.push(reversedNums[i]);
			if (includeCommas && i < reversedNums.length - 1) {
				target.push(comma);
			}
		}
		return { prompt, target };
	}

	public computeMetrics(completion: number[], target: number[]): ToyValidationMetrics {
		return {
			matches: tokenMatches(completion, target),
			accuracy: mustMatchAccuracy(completion, target)
		};
	}
}
