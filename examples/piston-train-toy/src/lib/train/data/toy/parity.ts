import type { ToyValidationMetrics } from './types';

import ToyDataset, { mustMatchAccuracy, tokenMatches, type ToySequence } from './dataset';

export interface ParityConfig {
	sequenceLength: number;
	includeColon: boolean;
}

export const PARITY_CONFIG_METADATA = {
	name: 'Parity',
	description: 'Determine if a bit string has even or odd number of 1s (parity)',
	citations: {
		entries: [
			{ name: 'Minsky & Papert, 1969', url: 'https://psycnet.apa.org/record/1969-35017-000' }
		]
	},
	supportsModelTypes: ['encoder', 'encoder-decoder', 'decoder'],
	parameters: {
		sequenceLength: {
			name: 'Sequence Length',
			description: 'Number of bits in the string',
			type: 'number' as const,
			min: 1,
			max: 20,
			default: 4
		},
		includeColon: {
			name: 'Include Colon',
			type: 'boolean' as const,
			default: true
		}
	}
} as const;

export const PARITY_CONFIG_DEFAULTS: ParityConfig = {
	sequenceLength: PARITY_CONFIG_METADATA.parameters.sequenceLength.default,
	includeColon: PARITY_CONFIG_METADATA.parameters.includeColon.default
};

export const PARITY_SHORT_DESCRIPTIONS = {
	sequenceLength: 'seq len',
	includeColon: 'include :'
};

export class ParityDataset extends ToyDataset<ParityConfig> {
	/**
	 * For parity dataset:
	 * - token 0 -> '0'
	 * - token 1 -> '1'
	 * - token 2 -> ':' (if includeColon is true)
	 * - token 3 -> 'even'
	 * - token 4 -> 'odd'
	 */
	protected buildVocab(): string[] {
		const { includeColon } = this.config;
		const vocab: string[] = ['0', '1'];

		if (includeColon) {
			vocab.push(':');
		}

		vocab.push('even', 'odd');
		return vocab;
	}

	/**
	 * Parity: generate a bit string and determine its parity:
	 * - prompt: the bit string with optional colon at the end
	 * - target: "even" if even number of 1s, "odd" if odd number of 1s
	 * Example: 10110:odd (3 ones = odd parity)
	 */
	public generateSequence(): ToySequence {
		const { sequenceLength, includeColon } = this.config;
		const bits = Array.from({ length: sequenceLength }, () => this.generator.integer(0, 1));

		const prompt: number[] = [];

		// Add the bit string to the prompt
		for (const bit of bits) {
			prompt.push(bit);
		}

		// Add colon if enabled
		if (includeColon) {
			const colon = this.tokenizer.vocab[':'];
			prompt.push(colon);
		}

		// Calculate parity (count number of 1s)
		const onesCount = bits.filter((bit) => bit === 1).length;
		const isEven = onesCount % 2 === 0;

		const target: number[] = [];
		if (isEven) {
			target.push(this.tokenizer.vocab['even']);
		} else {
			target.push(this.tokenizer.vocab['odd']);
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
