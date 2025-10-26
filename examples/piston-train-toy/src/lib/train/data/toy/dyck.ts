import ToyDataset, { type ToySequence } from './dataset';

export interface DyckConfig {
	sequenceLength: number;
	order: number;
	onlyTrainOnClosingBrackets: boolean;
}

export const DYCK_CONFIG_METADATA = {
	name: 'Dyck Language',
	description: 'Generate balanced bracket sequences (Dyck words)',
	supportsModelTypes: ['encoder'],
	parameters: {
		sequenceLength: {
			name: 'Sequence Length',
			description: 'Length of the sequence (must be even)',
			type: 'number' as const,
			min: 2,
			max: 100,
			step: 2,
			default: 10
		},
		order: {
			name: 'Order (Dyck-n)',
			description: 'Number of bracket types (Dyck-n)',
			type: 'number' as const,
			min: 1,
			max: 10,
			default: 2
		},
		onlyTrainOnClosingBrackets: {
			name: 'Only Train on Closing Brackets',
			description: 'Only train on closing brackets',
			type: 'boolean' as const,
			default: false
		}
	}
} as const;

export const DYCK_CONFIG_DEFAULTS: DyckConfig = {
	sequenceLength: DYCK_CONFIG_METADATA.parameters.sequenceLength.default,
	order: DYCK_CONFIG_METADATA.parameters.order.default,
	onlyTrainOnClosingBrackets: DYCK_CONFIG_METADATA.parameters.onlyTrainOnClosingBrackets.default
};

export class DyckDataset extends ToyDataset<DyckConfig> {
	/**
	 * For Dyck dataset:
	 * - Generate bracket pairs based on order
	 * - For order > 3, use regular number notation (e.g., (2, [2, {2)
	 */
	protected buildVocab(): string[] {
		const { order } = this.config;
		const vocab: string[] = [];

		// Base bracket types
		const baseBrackets = ['()', '[]', '{}'];

		// Generate bracket pairs based on order
		for (let i = 0; i < order; i++) {
			const baseIndex = i % 3;
			const level = Math.floor(i / 3);

			let openBracket = baseBrackets[baseIndex][0];
			let closeBracket = baseBrackets[baseIndex][1];

			// Add number notation for levels > 0
			if (order > 3) {
				const numberSuffix = (level + 1).toString();
				openBracket += numberSuffix;
				closeBracket += numberSuffix;
			}

			vocab.push(openBracket);
			vocab.push(closeBracket);
		}

		return vocab;
	}

	/**
	 * Generate a random Dyck word (balanced brackets) of exact length.
	 * The algorithm uses a stack to track unclosed openings and ensures
	 * the result is properly balanced.
	 */
	public generateSequence(): ToySequence {
		const { sequenceLength, order, onlyTrainOnClosingBrackets } = this.config;

		// Ensure length is even
		const length = sequenceLength % 2 === 0 ? sequenceLength : sequenceLength - 1;

		// Generate bracket pairs
		const pairs: [string, string][] = [];
		const vocab = this.buildVocab();

		for (let i = 0; i < order; i++) {
			const openBracket = vocab[i * 2];
			const closeBracket = vocab[i * 2 + 1];
			pairs.push([openBracket, closeBracket]);
		}

		let openLeft = length / 2; // how many openings remain
		let closeLeft = length / 2; // how many closings remain
		const stack: [string, string][] = []; // track unclosed openings
		const result: string[] = []; // output characters

		while (openLeft > 0 || closeLeft > 0) {
			if (openLeft === 0) {
				// must close
				const pair = stack.pop()!;
				result.push(pair[1]);
				closeLeft -= 1;
			} else if (closeLeft === 0) {
				// must open
				const pair = pairs[this.generator.integer(0, pairs.length - 1)];
				stack.push(pair);
				result.push(pair[0]);
				openLeft -= 1;
			} else {
				// we can choose to open or close (if stack not empty)
				if (stack.length > 0 && this.generator.real(0, 1) < 0.5) {
					// close
					const pair = stack.pop()!;
					result.push(pair[1]);
					closeLeft -= 1;
				} else {
					// open
					const pair = pairs[this.generator.integer(0, pairs.length - 1)];
					stack.push(pair);
					result.push(pair[0]);
					openLeft -= 1;
				}
			}
		}

		// Convert to tokens
		const tokens = result.map((char) => this.tokenizer.vocab[char]);

		// Only apply mask if onlyTrainOnClosingBrackets is true
		if (onlyTrainOnClosingBrackets) {
			// Create a set of closing bracket characters for easy lookup
			const closingBrackets = new Set<string>();
			for (let i = 0; i < order; i++) {
				const closeBracket = vocab[i * 2 + 1]; // Closing brackets are at odd indices
				closingBrackets.add(closeBracket);
			}

			// Create boolean mask - true for closing brackets, false for opening brackets
			const mask = result.map((char) => closingBrackets.has(char));

			return { target: tokens, mask };
		} else {
			// Return tokens without masking
			return { target: tokens };
		}
	}
}
