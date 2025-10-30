import type { ToyValidationMetrics } from './types';

import ToyDataset, { mustMatchAccuracy, tokenMatches, type ToySequence } from './dataset';

export interface SlapjackConfig {
	sequenceLength: number;
	slapOnDoubles: boolean;
	slapOnSandwiches: boolean;
	includeColon: boolean;
	// onlyTrainOnSlaps: boolean;
}

export const SLAPJACK_CONFIG_METADATA = {
	name: 'Slapjack',
	description:
		'"Slap" (🖐️) when the card is a Jack, a double (same card twice in a row), or a sandwich (same card with one card in between). Otherwise, take no action (❌).',
	supportsModelTypes: ['encoder', 'encoder-decoder', 'decoder'],
	parameters: {
		sequenceLength: {
			name: 'Sequence Length',
			description: 'Total length of the sequence',
			type: 'number' as const,
			min: 5,
			max: 200,
			default: 5
		},
		slapOnDoubles: {
			name: 'Slap on Doubles',
			description: 'Slap on doubles (same card twice in a row)',
			type: 'boolean' as const,
			default: true
		},
		slapOnSandwiches: {
			name: 'Slap on Sandwiches',
			description: 'Slap on sandwiches (same card with one card in between)',
			type: 'boolean' as const,
			default: true
		},
		includeColon: {
			name: 'Include Colon',
			type: 'boolean' as const,
			default: true
		}
		// onlyTrainOnSlaps: {
		// 	name: 'Only Train on Slaps',
		// 	description: 'Only train on slaps (not on cards)',
		// 	type: 'boolean' as const,
		// 	default: true
		// }
	}
} as const;

export const SLAPJACK_CONFIG_DEFAULTS: SlapjackConfig = {
	sequenceLength: SLAPJACK_CONFIG_METADATA.parameters.sequenceLength.default,
	slapOnDoubles: SLAPJACK_CONFIG_METADATA.parameters.slapOnDoubles.default,
	slapOnSandwiches: SLAPJACK_CONFIG_METADATA.parameters.slapOnSandwiches.default,
	includeColon: SLAPJACK_CONFIG_METADATA.parameters.includeColon.default
	// onlyTrainOnSlaps: SLAPJACK_CONFIG_METADATA.parameters.onlyTrainOnSlaps.default,
};

const CARDS = 'A23456789JQK';

export class SlapjackDataset extends ToyDataset<SlapjackConfig> {
	/**
	 * Vocabulary contains card tokens and action tokens (🖐️ slap, ❌ no slap).
	 */
	protected buildVocab(): string[] {
		const { includeColon } = this.config;
		const vocab = [...CARDS.split('')];
		if (includeColon) vocab.push(':');
		vocab.push('🖐️', '❌');
		return vocab;
	}

	/**
	 * Slapjack:
	 * - Input (prompt): sequence of cards of length sequenceLength
	 * - Target: action at each position (🖐️ if slap, ❌ otherwise)
	 * Slap rules:
	 *   - Slap on Jack (J)
	 *   - Slap on doubles (same card twice in a row) if enabled
	 *   - Slap on sandwiches (same card with one card in between) if enabled
	 */
	public generateSequence(): ToySequence {
		const { sequenceLength, slapOnDoubles, slapOnSandwiches, includeColon } = this.config;

		// Build card sequence
		const cardSeq = Array.from(
			{ length: sequenceLength },
			() => CARDS[this.generator.integer(0, CARDS.length - 1)]
		);

		// Map to token ids for prompt
		const prompt: number[] = cardSeq.map((c) => this.tokenizer.vocab[c]);
		if (includeColon) {
			const colon = this.tokenizer.vocab[':'];
			prompt.push(colon);
		}

		// Compute action at each position
		const slapId = this.tokenizer.vocab['🖐️'];
		const noId = this.tokenizer.vocab['❌'];
		const target: number[] = [];
		for (let i = 0; i < cardSeq.length; i++) {
			const c = cardSeq[i];
			const isJack = c === 'J';
			const isDouble = slapOnDoubles && i > 0 && c === cardSeq[i - 1];
			const isSandwich = slapOnSandwiches && i > 1 && c === cardSeq[i - 2];
			const shouldSlap = isJack || isDouble || isSandwich;
			target.push(shouldSlap ? slapId : noId);
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
