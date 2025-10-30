import type { ToyValidationMetrics } from './types';

import ToyDataset, { mustMatchAccuracy, tokenMatches, type ToySequence } from './dataset';

export interface ModularAdditionConfig {
	maxNumber: number;
	modulo: number;
	includeExpressionTokens: boolean;
}

export const MODULAR_ADDITION_CONFIG_METADATA = {
	name: 'Modular Addition',
	description: 'Add two numbers with modulo',
	supportsModelTypes: ['encoder', 'encoder-decoder', 'decoder'],
	parameters: {
		maxNumber: {
			name: 'Max Number',
			type: 'number' as const,
			min: 5,
			max: 1000,
			default: 500
		},
		modulo: {
			name: 'Modulo',
			type: 'number' as const,
			min: 2,
			max: 113,
			default: 113
		},
		includeExpressionTokens: {
			name: 'Include Expression Tokens (+, =)',
			type: 'boolean' as const,
			default: true
		}
	}
} as const;

export const MODULAR_ADDITION_CONFIG_DEFAULTS: ModularAdditionConfig = {
	maxNumber: MODULAR_ADDITION_CONFIG_METADATA.parameters.maxNumber.default,
	modulo: MODULAR_ADDITION_CONFIG_METADATA.parameters.modulo.default,
	includeExpressionTokens:
		MODULAR_ADDITION_CONFIG_METADATA.parameters.includeExpressionTokens.default
};

export class ModularAdditionDataset extends ToyDataset<ModularAdditionConfig> {
	/**
	 * Tokenizer for modular addition.
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
	 * Modular Addition: generate two numbers (in [0, maxNum)) and compute (num1+num2)%maxNum.
	 * Example: <15>+<03>=<05> (if maxNum is 113 and 15+3=18, 18%113=18, but if 15+100=115, 115%113=2)
	 */
	public generateSequence(): ToySequence {
		const { maxNumber, modulo, includeExpressionTokens } = this.config;

		// 1. Pick numbers and compute sum. Include maxNumber in the range.
		const sum = this.generator.integer(0, maxNumber);
		const num1 = sum > 0 ? this.generator.integer(0, sum - 1) : 0;
		const num2 = sum - num1;
		const sumModulo = sum % modulo;

		// 2. Convert to token IDs:
		// - 0..maxNum-1 are numbers
		// - maxNum is '+'
		// - maxNum+1 is '='
		let prompt;
		if (includeExpressionTokens) {
			prompt = [num1, maxNumber + 1, num2, maxNumber + 2];
		} else {
			prompt = [num1, num2];
		}

		// 3. Target is a single token corresponding to the modular sum
		const target = [sumModulo];

		return { prompt, target };
	}

	public computeMetrics(completion: number[], target: number[]): ToyValidationMetrics {
		return {
			matches: tokenMatches(completion, target),
			accuracy: mustMatchAccuracy(completion, target)
		};
	}
}
