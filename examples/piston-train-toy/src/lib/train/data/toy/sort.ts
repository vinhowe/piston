import ToyDataset, { type ToySequence } from './dataset';

export interface SortConfig {
	sequenceLength: number;
	maxNumber: number;
	includeCommas: boolean;
	includeColon: boolean;
}

export const SORT_CONFIG_METADATA = {
	name: 'Alphabetical Sorting',
	description: 'Sort a sequence of alphabet characters',
	supportsModelTypes: ['encoder', 'encoder-decoder', 'decoder'],
	parameters: {
		sequenceLength: {
			name: 'Number of Items',
			type: 'number' as const,
			min: 2,
			max: 10,
			default: 5
		},
		maxNumber: {
			name: 'Max Alphabet Character',
			type: 'number' as const,
			min: 3,
			max: 26,
			default: 26
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

export const SORT_CONFIG_DEFAULTS: SortConfig = {
	sequenceLength: SORT_CONFIG_METADATA.parameters.sequenceLength.default,
	maxNumber: SORT_CONFIG_METADATA.parameters.maxNumber.default,
	includeCommas: SORT_CONFIG_METADATA.parameters.includeCommas.default,
	includeColon: SORT_CONFIG_METADATA.parameters.includeColon.default
};

export class SortDataset extends ToyDataset<SortConfig> {
	/**
	 * For sort dataset:
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
	 * Sorting: generate a sequence of characters and return:
	 * - prompt: the unsorted sequence with commas (token 1) between numbers and a colon (token 0) at the end.
	 * - target: the sorted sequence with commas between numbers.
	 * Example: BBCA:ABCBB
	 */
	public generateSequence(): ToySequence {
		const { sequenceLength, maxNumber, includeCommas, includeColon } = this.config;
		const nums = Array.from({ length: sequenceLength }, () =>
			this.generator.integer(0, maxNumber - 1)
		);
		const sorted = [...nums].sort((a, b) => a - b);
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
		for (let i = 0; i < sorted.length; i++) {
			target.push(sorted[i]);
			if (includeCommas && i < sorted.length - 1) {
				target.push(comma);
			}
		}
		return { prompt, target };
	}
}
