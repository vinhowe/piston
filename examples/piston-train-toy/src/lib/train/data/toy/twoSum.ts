import ToyDataset, { type ToySequence } from './dataset';

export interface TwoSumConfig {
	sequenceLength: number;
	maxNumber: number;
	includeCommas: boolean;
	includeExpressionTokens: boolean;
}

export const TWO_SUM_CONFIG_METADATA = {
	name: 'Two Sum',
	description: 'Find two numbers in a sequence that sum to a target',
	supportsModelTypes: ['encoder', 'encoder-decoder', 'decoder'],
	parameters: {
		sequenceLength: {
			name: 'Sequence Length',
			description: 'Length of the sequence',
			type: 'number' as const,
			min: 2,
			max: 10,
			default: 4
		},
		maxNumber: {
			name: 'Max Number',
			description: 'Maximum value in the sequence',
			type: 'number' as const,
			min: 10,
			max: 100,
			default: 15
		},
		includeCommas: {
			name: 'Include Commas',
			type: 'boolean' as const,
			default: false
		},
		includeExpressionTokens: {
			name: 'Include Expression Tokens',
			type: 'boolean' as const,
			default: true
		}
	}
} as const;

export const TWO_SUM_CONFIG_DEFAULTS: TwoSumConfig = {
	sequenceLength: TWO_SUM_CONFIG_METADATA.parameters.sequenceLength.default,
	maxNumber: TWO_SUM_CONFIG_METADATA.parameters.maxNumber.default,
	includeCommas: TWO_SUM_CONFIG_METADATA.parameters.includeCommas.default,
	includeExpressionTokens: TWO_SUM_CONFIG_METADATA.parameters.includeExpressionTokens.default
};

export class TwoSumDataset extends ToyDataset<TwoSumConfig> {
	/**
	 * For Two-Sum dataset:
	 * - tokens 0 .. maxNum represent the numbers 0 .. maxNum.
	 * - token maxNum+1 -> ':'
	 * - token maxNum+2 -> '='
	 * - token maxNum+3 -> ','
	 */
	protected buildVocab(): string[] {
		const { maxNumber, includeCommas, includeExpressionTokens } = this.config;
		const vocab: string[] = [];
		const padWidth = maxNumber.toString().length;
		// 1. Add tokens for numbers
		vocab.push(
			...Array.from({ length: maxNumber + 1 }, (_, i) => i.toString().padStart(padWidth, '0'))
		);
		// 2. Add tokens for operators
		if (includeExpressionTokens) {
			vocab.push(':');
			vocab.push('=');
		}
		if (includeCommas) {
			vocab.push(',');
		}
		return vocab;
	}

	/**
	 * Two-Sum: generate a sequence of numbers (each in [0, maxNum]) and two random indices.
	 * The prompt consists of:
	 *   - the sequence (with commas if enabled)
	 *   - a colon, the sum and an equals sign.
	 * The target is the two numbers (from the chosen indices) separated by a comma.
	 * Example: ABC:D=AB or ABC:D=A,B
	 */
	public generateSequence(): ToySequence {
		const { sequenceLength, maxNumber, includeCommas, includeExpressionTokens } = this.config;
		// Generate a two-sum instance with a unique solution using a target-first constructive method.
		const MAX_RETRIES = 10;
		const maxElement = Math.floor(maxNumber / 2);
		for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
			// 1) Choose two distinct indices for the special pair
			const i = this.generator.integer(0, sequenceLength - 1);
			let j = this.generator.integer(0, sequenceLength - 1);
			while (j === i) {
				j = this.generator.integer(0, sequenceLength - 1);
			}
			// 2) Choose their values from the usable range so t stays in-vocab
			const a = this.generator.integer(0, maxElement);
			const b = this.generator.integer(0, maxElement);
			const t = a + b; // target sum

			// 3) Fill the rest while avoiding any other pair summing to t
			const nums = new Array<number>(sequenceLength);
			nums[i] = a;
			nums[j] = b;

			const seenNonSpecial: number[] = [];
			let feasible = true;

			for (let k = 0; k < sequenceLength; k++) {
				if (k === i || k === j) continue;

				// Build the set of disallowed values for this position
				const disallowed = new Set<number>();
				disallowed.add(a);
				disallowed.add(b);
				for (const y of seenNonSpecial) {
					const complement = t - y;
					if (complement >= 0 && complement <= maxElement) {
						disallowed.add(complement);
					}
				}

				// Enumerate allowed candidates in the usable range
				const allowed: number[] = [];
				for (let val = 0; val <= maxElement; val++) {
					if (!disallowed.has(val)) {
						allowed.push(val);
					}
				}

				if (allowed.length === 0) {
					feasible = false;
					break;
				}

				const choice = allowed[this.generator.integer(0, allowed.length - 1)];
				nums[k] = choice;
				seenNonSpecial.push(choice);
			}

			if (!feasible) {
				continue;
			}

			// 4) Build prompt and target
			const prompt: number[] = [];
			const comma = this.tokenizer.vocab[','];

			for (let k = 0; k < nums.length; k++) {
				prompt.push(nums[k]);
				if (includeCommas && k < nums.length - 1) {
					prompt.push(comma);
				}
			}

			if (includeExpressionTokens) {
				prompt.push(this.tokenizer.vocab[':']);
			}

			prompt.push(t);

			if (includeExpressionTokens) {
				prompt.push(this.tokenizer.vocab['=']);
			}

			const target: number[] = [nums[i]];
			if (includeCommas) {
				target.push(comma);
			}
			target.push(nums[j]);

			return { prompt, target };
		}
		throw new Error('Failed to generate a unique-solution two-sum instance after 10 attempts');
	}
}
