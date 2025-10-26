import ToyDataset, { type ToySequence } from './dataset';

export interface CopyMemoryConfig {
	prefixLength: number; // length of the subsequence to memorize (A,B,C)
	distractorLength: number; // length of distractor stream between prefix and recall signal
	vocabSize: number; // vocabulary for prefix and distractors (A..)
	recallTokenEnabled: boolean; // include '?' token to signal recall time
	includeSeparators: boolean; // optional commas between tokens
}

export const COPY_MEMORY_CONFIG_METADATA = {
	name: 'Copy Memory with Distractors',
	description: 'Memorize a subsequence and reproduce it at the end after distractors.',
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
		prefixLength: {
			name: 'Prefix Length',
			description: 'Length of subsequence to memorize',
			type: 'number' as const,
			min: 1,
			max: 32,
			step: 1,
			default: 3
		},
		distractorLength: {
			name: 'Distractor Length',
			description: 'Number of distractor tokens before recall',
			type: 'number' as const,
			min: 0,
			max: 512,
			step: 1,
			default: 12
		},
		vocabSize: {
			name: 'Vocab Size',
			description: 'Number of token types used for prefix/distractors',
			type: 'number' as const,
			min: 2,
			max: 128,
			step: 1,
			default: 10
		},
		recallTokenEnabled: {
			name: 'Include Recall Token (?)',
			description: 'Append a ? token to signal recall time',
			type: 'boolean' as const,
			default: true
		},
		includeSeparators: {
			name: 'Include Separators',
			description: 'Insert commas between tokens',
			type: 'boolean' as const,
			default: false
		}
	}
} as const;

export const COPY_MEMORY_CONFIG_DEFAULTS: CopyMemoryConfig = {
	prefixLength: COPY_MEMORY_CONFIG_METADATA.parameters.prefixLength.default,
	distractorLength: COPY_MEMORY_CONFIG_METADATA.parameters.distractorLength.default,
	vocabSize: COPY_MEMORY_CONFIG_METADATA.parameters.vocabSize.default,
	recallTokenEnabled: COPY_MEMORY_CONFIG_METADATA.parameters.recallTokenEnabled.default,
	includeSeparators: COPY_MEMORY_CONFIG_METADATA.parameters.includeSeparators.default
};

export class CopyMemoryDataset extends ToyDataset<CopyMemoryConfig> {
	/**
	 * Vocab:
	 * - letters L0..L{vocabSize-1}
	 * - optional comma separator
	 * - optional recall token '?'
	 */
	protected buildVocab(): string[] {
		const { vocabSize, recallTokenEnabled, includeSeparators } = this.config;
		const padWidth = vocabSize.toString().length;
		const vocab = Array.from(
			{ length: vocabSize + 1 },
			(_, i) => `L${i.toString().padStart(padWidth, '0')}`
		);
		if (includeSeparators) vocab.push(',');
		if (recallTokenEnabled) vocab.push('?');
		return vocab;
	}

	/**
	 * Prompt layout: P0 P1 ... P{prefix-1} [distractors...] [?]
	 * Target: P0 P1 ... P{prefix-1}
	 */
	public generateSequence(): ToySequence {
		const { prefixLength, distractorLength, vocabSize, recallTokenEnabled, includeSeparators } =
			this.config;
		const prompt: number[] = [];

		// Build prefix
		const prefix: number[] = [];
		for (let i = 0; i < prefixLength; i++) {
			prefix.push(this.generator.integer(0, vocabSize - 1));
		}

		const comma = this.tokenizer.vocab[','];

		// Push prefix into prompt
		for (let i = 0; i < prefix.length; i++) {
			prompt.push(prefix[i]);
			if (includeSeparators) {
				if (i < prefix.length - 1 || distractorLength > 0 || recallTokenEnabled) {
					prompt.push(comma);
				}
			}
		}

		// Add distractors
		for (let i = 0; i < distractorLength; i++) {
			prompt.push(this.generator.integer(0, vocabSize - 1));
			if (includeSeparators && (i < distractorLength - 1 || recallTokenEnabled)) {
				prompt.push(comma);
			}
		}

		// Add recall token
		if (recallTokenEnabled) {
			prompt.push(this.tokenizer.vocab['?']);
		}

		// Target is the original prefix
		const target = [...prefix];
		return { prompt, target };
	}
}
