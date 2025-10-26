import ToyDataset, { type ToySequence } from './dataset';

export interface TemporalOrderConfig {
	sequenceLength: number; // total length including distractors and special symbols X/Y
	vocabSize: number; // number of distractor letters (A..)
	includeSeparators: boolean; // whether to include comma separators between tokens
}

export const TEMPORAL_ORDER_CONFIG_METADATA = {
	name: 'Temporal Order Recognition',
	description: 'Classify whether X appears before Y (XY) or after Y (YX).',
	supportsModelTypes: ['encoder', 'encoder-decoder', 'decoder'],
	citations: {
		entries: [
			{
				name: 'Hochreiter & Schmidhuber, 1997',
				url: 'https://ieeexplore.ieee.org/abstract/document/6795963'
			}
		]
	},
	parameters: {
		sequenceLength: {
			name: 'Sequence Length',
			description: 'Total number of tokens (including X and Y)',
			type: 'number' as const,
			min: 5,
			max: 256,
			step: 1,
			default: 13
		},
		vocabSize: {
			name: 'Distractor Vocab Size',
			description: 'Number of distractor letters (A..)',
			type: 'number' as const,
			min: 2,
			max: 100,
			step: 1,
			default: 10
		},
		includeSeparators: {
			name: 'Include Separators',
			description: 'Insert commas between tokens',
			type: 'boolean' as const,
			default: false
		}
	}
} as const;

export const TEMPORAL_ORDER_CONFIG_DEFAULTS: TemporalOrderConfig = {
	sequenceLength: TEMPORAL_ORDER_CONFIG_METADATA.parameters.sequenceLength.default,
	vocabSize: TEMPORAL_ORDER_CONFIG_METADATA.parameters.vocabSize.default,
	includeSeparators: TEMPORAL_ORDER_CONFIG_METADATA.parameters.includeSeparators.default
};

export class TemporalOrderDataset extends ToyDataset<TemporalOrderConfig> {
	/**
	 * Vocab:
	 * - distractor letters D0..D{vocabSize-1}
	 * - special symbols 'X', 'Y'
	 * - classes 'XY', 'YX'
	 * - optional ',' separator
	 */
	protected buildVocab(): string[] {
		const { vocabSize, includeSeparators } = this.config;
		const padWidth = vocabSize.toString().length;
		// 1. Add tokens for numbers
		const vocab = Array.from(
			{ length: vocabSize + 1 },
			(_, i) => `D${i.toString().padStart(padWidth, '0')}`
		);
		vocab.push('X', 'Y', 'XY', 'YX');
		if (includeSeparators) vocab.push(',');
		return vocab;
	}

	/**
	 * Build a sequence with exactly one X and one Y inserted among distractors.
	 * Prompt: sequence of tokens (with optional commas). Target: single class token.
	 */
	public generateSequence(): ToySequence {
		const { sequenceLength, vocabSize, includeSeparators } = this.config;
		if (sequenceLength < 2) {
			throw new Error('sequenceLength must be at least 2 to place X and Y');
		}
		const prompt: number[] = [];

		// Choose distinct positions for X and Y
		const posX = this.generator.integer(0, sequenceLength - 1);
		let posY = this.generator.integer(0, sequenceLength - 1);
		while (posY === posX) posY = this.generator.integer(0, sequenceLength - 1);

		const comma = this.tokenizer.vocab[','];
		const x = this.tokenizer.vocab['X'];
		const y = this.tokenizer.vocab['Y'];

		for (let i = 0; i < sequenceLength; i++) {
			let tokenId: number;
			if (i === posX) {
				tokenId = x;
			} else if (i === posY) {
				tokenId = y;
			} else {
				tokenId = this.generator.integer(0, vocabSize - 1);
			}
			prompt.push(tokenId);
			if (includeSeparators && i < sequenceLength - 1) {
				prompt.push(comma);
			}
		}

		const classToken = posX < posY ? 'XY' : 'YX';
		const target = [this.tokenizer.vocab[classToken]];
		return { prompt, target };
	}
}
