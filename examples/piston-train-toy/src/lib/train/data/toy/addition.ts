import type { ToyValidationMetrics } from './types';

import ToyDataset, { mustMatchAccuracy, tokenMatches, type ToySequence } from './dataset';

export interface AdditionConfig {
	maxNumber: number;
	includeExpressionTokens: boolean;
}

export const ADDITION_CONFIG_METADATA = {
	name: 'Addition',
	description: 'Add two numbers.',
	supportsModelTypes: ['encoder', 'encoder-decoder', 'decoder'],
	parameters: {
		maxNumber: {
			name: 'Max Number',
			description: 'Maximum value for each addend',
			type: 'number' as const,
			min: 5,
			max: 100,
			default: 20
		},
		includeExpressionTokens: {
			name: 'Include Expression Tokens (+, =)',
			type: 'boolean' as const,
			default: true
		}
	}
} as const;

export const ADDITION_CONFIG_DEFAULTS: AdditionConfig = {
	maxNumber: ADDITION_CONFIG_METADATA.parameters.maxNumber.default,
	includeExpressionTokens: ADDITION_CONFIG_METADATA.parameters.includeExpressionTokens.default
};

export class AdditionDataset extends ToyDataset<AdditionConfig> {
	/**
	 * Tokenizer for addition.
	 * - tokens 0 .. maxNum represent the numbers 0 … maxNum
	 *   (When converting to a string, token id i <= maxNum becomes `<${i}>` with zero-padding.)
	 * - token maxNum+1 -> "+"
	 * - token maxNum+2 -> "="
	 */
	protected buildVocab(): string[] {
		const { maxNumber, includeExpressionTokens } = this.config;
		const vocab: string[] = [];
		const padWidth = maxNumber.toString().length;
		// 1. Add tokens for numbers
		vocab.push(
			...Array.from({ length: maxNumber + 1 }, (_, i) => i.toString().padStart(padWidth, '0'))
		);
		// 2. Add tokens for + and =
		if (includeExpressionTokens) {
			vocab.push('+', '=');
		}
		return vocab;
	}

	/**
	 * Addition: generate two addends (in [0, maxNum]) chosen so that their sum <= maxNum.
	 * Example: <15>+<03>=<18>
	 */
	public generateSequence(): ToySequence {
		const { maxNumber, includeExpressionTokens } = this.config;

		// 1. Pick numbers and compute sum. Include maxNumber in the range.
		const sum = this.generator.integer(0, maxNumber);
		const num1 = sum > 0 ? this.generator.integer(0, sum - 1) : 0;
		const num2 = sum - num1;

		// 2. Convert to token IDs:
		// - 0..maxNum are numbers
		// - maxNum+1 is '+'
		// - maxNum+2 is '='
		let prompt;
		if (includeExpressionTokens) {
			prompt = [num1, maxNumber + 1, num2, maxNumber + 2];
		} else {
			prompt = [num1, num2];
		}

		// 3. Target is a single token corresponding to the sum
		const target = [sum];

		return { prompt, target };
	}

	public computeMetrics(completion: number[], target: number[]): ToyValidationMetrics {
		return {
			matches: tokenMatches(completion, target),
			accuracy: mustMatchAccuracy(completion, target)
		};
	}
}
