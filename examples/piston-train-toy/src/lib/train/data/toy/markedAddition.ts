import ToyDataset, { type ToySequence } from './dataset';

export interface MarkedAdditionConfig {
	sequenceLength: number; // number of (value, marker) pairs
	maxInputValue: number; // largest value sampled per element in the sequence
	maxSumValue: number; // largest representable sum (numeric vocab range)
	includeEqualsToken: boolean; // include '=' delimiter at end of prompt
}

export const MARKED_ADDITION_CONFIG_METADATA = {
	name: 'Marked Addition',
	description:
		'Sum values whose associated marker equals 1. Not real-valued, unlike the original task.',
	citations: {
		entries: [
			{
				name: 'Hochreiter & Schmidhuber, 1997',
				url: 'https://ieeexplore.ieee.org/abstract/document/6795963'
			}
		]
	},
	supportsModelTypes: ['encoder', 'encoder-decoder', 'decoder'],
	parameters: {
		sequenceLength: {
			name: 'Sequence Length',
			description: 'Number of (value, marker) pairs',
			type: 'number' as const,
			min: 2,
			max: 64,
			step: 1,
			default: 8
		},
		maxSumValue: {
			name: 'Max Sum Value',
			description: 'Largest representable sum (controls numeric vocab range)',
			type: 'number' as const,
			min: 1,
			max: 10000,
			step: 1,
			default: 100
		},
		maxInputValue: {
			name: 'Max Input Value',
			description: 'Largest value sampled per element in the sequence',
			type: 'number' as const,
			min: 1,
			max: 1000,
			step: 1,
			default: 10
		},
		includeEqualsToken: {
			name: 'Include Equals Token',
			description: 'Append an = token after the sequence',
			type: 'boolean' as const,
			default: true
		}
	}
} as const;

export const MARKED_ADDITION_CONFIG_DEFAULTS: MarkedAdditionConfig = {
	sequenceLength: MARKED_ADDITION_CONFIG_METADATA.parameters.sequenceLength.default,
	maxInputValue: MARKED_ADDITION_CONFIG_METADATA.parameters.maxInputValue.default,
	maxSumValue: MARKED_ADDITION_CONFIG_METADATA.parameters.maxSumValue.default,
	includeEqualsToken: MARKED_ADDITION_CONFIG_METADATA.parameters.includeEqualsToken.default
};

export class MarkedAdditionDataset extends ToyDataset<MarkedAdditionConfig> {
	/**
	 * Vocab design:
	 * - values are discretized based on precision: for precision p, there are (10^p + 1) buckets
	 * 0.0..1.0.
	 *   We emit tokens V0..V{numBuckets-1} representing these buckets.
	 * - markers are tokens 'M0' and 'M1'.
	 * - optionally BOS/EOS/mask are appended by ToyDataset constructor based on specialTokensConfig.
	 */
	protected buildVocab(): string[] {
		const { maxSumValue, includeEqualsToken } = this.config;
		const padWidth = maxSumValue.toString().length;
		const vocab = Array.from({ length: maxSumValue + 1 }, (_, i) =>
			i.toString().padStart(padWidth, '0')
		);
		vocab.push('M0', 'M1');
		if (includeEqualsToken) {
			vocab.push('=');
		}
		return vocab;
	}

	/**
	 * Generate a sequence of (value, marker) pairs and target the sum of values with marker==1.
	 * Representation:
	 * - Discretize each value in [0,1] to bucket index b in [0, buckets-1]. Token is Vb.
	 * - Marker token is M0 or M1.
	 * - Prompt layout: Vb0 Mx0 Vb1 Mx1 ... Vb{n-1} Mx{n-1} '='
	 * - Target: a single token VbSum where bSum is the discretized sum clamped to [0, 1].
	 */
	public generateSequence(): ToySequence {
		const { sequenceLength, maxInputValue, maxSumValue, includeEqualsToken } = this.config;

		const prompt: number[] = [];
		let sum = 0;
		const padWidth = maxSumValue.toString().length;

		for (let i = 0; i < sequenceLength; i++) {
			const value = this.generator.integer(0, maxInputValue);
			const marker = this.generator.integer(0, 1);

			prompt.push(value);
			prompt.push(this.tokenizer.vocab[`M${marker}`]);

			if (marker === 1) sum += value;
		}

		if (includeEqualsToken) {
			prompt.push(this.tokenizer.vocab['=']);
		}

		if (sum > maxSumValue) {
			sum = maxSumValue; // clamp to representable range
		}

		const targetTokenStr = sum.toString().padStart(padWidth, '0');
		const target = [this.tokenizer.vocab[targetTokenStr]];

		return { prompt, target };
	}
}
