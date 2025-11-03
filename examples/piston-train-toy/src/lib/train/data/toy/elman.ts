import type { ToyValidationMetrics } from './types';

import ToyDataset, { type ToySequence } from './dataset';

export const ELMAN_CONFIG_METADATA = {
	name: 'Elman Grammar',
	description: 'Generate simplified sentences based on Elman (1990) grammar.',
	citations: {
		entries: [
			{
				name: 'Elman, 1990',
				url: 'https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog1402_1'
			}
		]
	},
	supportsModelTypes: ['encoder', 'encoder-decoder', 'decoder'],
	parameters: {}
} as const;

export const ELMAN_CONFIG_DEFAULTS = {} as const;
export type ElmanConfig = object;

export class ElmanDataset extends ToyDataset<null> {
	public readonly hasCanonicalTargets = false;
	// Compact placeholder dictionary
	private static categories: Record<string, string[]> = {
		NOUN_HUM: ['man', 'woman', 'boy', 'girl'],
		NOUN_ANIM: ['cat', 'mouse', 'dog'],
		NOUN_INANIM: ['book', 'rock', 'pencil'],
		NOUN_AGRESS: ['dragon', 'monster'],
		NOUN_FRAG: ['glass', 'plate'],
		NOUN_FOOD: ['apple', 'banana', 'orange', 'pie', 'pizza', 'sandwich'],
		VERB_INTRAN: ['think', 'sleep'],
		VERB_TRAN: ['see', 'chase'],
		VERB_AGPAT: ['move', 'break'],
		VERB_PERCEPT: ['smell', 'see'],
		VERB_DESTROY: ['break', 'smash'],
		VERB_EAT: ['eat', 'consume']
	};

	// Precompiled templates with exactly 3 positions; null means PAD
	private static templates: (string[] | null)[][] = [
		['NOUN_HUM', 'VERB_EAT', 'NOUN_FOOD'],
		['NOUN_HUM', 'VERB_PERCEPT', 'NOUN_INANIM'],
		['NOUN_HUM', 'VERB_DESTROY', 'NOUN_FRAG'],
		['NOUN_HUM', 'VERB_INTRAN', 'PAD'],
		['NOUN_HUM', 'VERB_TRAN', 'NOUN_HUM'],
		['NOUN_HUM', 'VERB_AGPAT', 'NOUN_INANIM'],
		['NOUN_HUM', 'VERB_AGPAT', 'PAD'],
		['NOUN_ANIM', 'VERB_EAT', 'NOUN_FOOD'],
		['NOUN_ANIM', 'VERB_TRAN', 'NOUN_ANIM'],
		['NOUN_ANIM', 'VERB_AGPAT', 'NOUN_INANIM'],
		['NOUN_ANIM', 'VERB_AGPAT', 'PAD'],
		['NOUN_INANIM', 'VERB_AGPAT', 'PAD'],
		['NOUN_AGRESS', 'VERB_DESTROY', 'NOUN_FRAG'],
		['NOUN_AGRESS', 'VERB_EAT', 'NOUN_HUM'],
		['NOUN_AGRESS', 'VERB_EAT', 'NOUN_ANIM'],
		['NOUN_AGRESS', 'VERB_EAT', 'NOUN_FOOD']
	].map((tpl) => tpl.map((slot) => (slot === 'PAD' ? null : ElmanDataset.categories[slot])));

	protected buildVocab(): string[] {
		// Union of all words used in generation, plus 'who'
		const vocabSet = new Set<string>();
		// console.log(ElmanDataset.categories);
		for (const list of Object.values(ElmanDataset.categories)) {
			for (const w of list) {
				vocabSet.add(w);
			}
		}
		vocabSet.add('PAD');
		return Array.from(vocabSet);
	}

	public generateSequence(): ToySequence {
		// Choose a precompiled template
		const template =
			ElmanDataset.templates[this.generator.integer(0, ElmanDataset.templates.length - 1)];

		// Replace placeholders with dictionary lookup (null => PAD)
		const sentenceWords: string[] = template.map((slot) => {
			if (slot === null) return 'PAD';
			const idx = this.generator.integer(0, slot.length - 1);
			return slot[idx] as string;
		});

		// Map words to token IDs
		const target = sentenceWords.map((w) => this.tokenizer.vocab[w]);
		return { target };
	}

	public computeMetrics(completion: number[], _target: number[]): ToyValidationMetrics {
		// Remove EOS if present at the end
		let seq = completion.slice();
		if (this.eosId !== null && seq.length > 0 && seq[seq.length - 1] === this.eosId) {
			seq = seq.slice(0, -1);
		}

		// Decode to words for template matching
		const words = seq.map((id) => this.tokenizer.ids[id] ?? '<unk>');

		// Best-prefix match across templates (only first n tokens, in order)
		let bestPrefix = 0;
		for (const tpl of ElmanDataset.templates) {
			let prefix = 0;
			for (let i = 0; i < Math.min(3, words.length); i++) {
				const slot = tpl[i];
				const w = words[i];
				const ok = slot === null ? w === 'PAD' : slot.includes(w);
				if (!ok) break;
				prefix++;
				if (prefix === 3) break; // early exit for full match
			}
			if (prefix > bestPrefix) {
				bestPrefix = prefix;
				if (bestPrefix === 3) break; // early exit if any template fully matches
			}
		}

		// Per-token matches for up to 3 tokens
		const matches: boolean[] = [];
		for (let i = 0; i < Math.min(3, words.length); i++) {
			matches.push(i < bestPrefix);
		}

		const length_correct = seq.length === 3;
		const matched_template = bestPrefix === 3;
		const valid_statement = matched_template;

		return {
			length_correct,
			matched_template,
			valid_statement,
			matches
		};
	}
}
