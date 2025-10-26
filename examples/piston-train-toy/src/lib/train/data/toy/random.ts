import ToyDataset, { type ToySequence } from './dataset';

export interface RandomConfig {
	sequenceLength: number; // length of the prompt
	responseLength: number; // length of the response
	vocabSize: number; // number of non-special tokens
}

export const RANDOM_CONFIG_METADATA = {
	name: 'Random Token Prediction',
	description:
		'Prompt is random tokens and target is an independent random token. This is a good test case; there is no learnable mapping, so any model should struggle to learn this task.',
	supportsModelTypes: ['encoder', 'encoder-decoder', 'decoder'],
	parameters: {
		sequenceLength: {
			name: 'Prompt Length',
			description: 'Number of tokens in the prompt',
			type: 'number' as const,
			min: 0,
			max: 512,
			step: 1,
			default: 5
		},
		responseLength: {
			name: 'Response Length',
			description: 'Number of tokens in the response',
			type: 'number' as const,
			min: 0,
			max: 512,
			step: 1,
			default: 1
		},
		vocabSize: {
			name: 'Vocab Size',
			description: 'Number of distinct tokens (excluding special tokens)',
			type: 'number' as const,
			min: 2,
			max: 1000,
			step: 1,
			default: 10
		}
	}
} as const;

export const RANDOM_CONFIG_DEFAULTS: RandomConfig = {
	sequenceLength: RANDOM_CONFIG_METADATA.parameters.sequenceLength.default,
	responseLength: RANDOM_CONFIG_METADATA.parameters.responseLength.default,
	vocabSize: RANDOM_CONFIG_METADATA.parameters.vocabSize.default
};

export class RandomDataset extends ToyDataset<RandomConfig> {
	/**
	 * For the random dataset:
	 * - Create token strings V0..V{vocabSize-1}
	 */
	protected buildVocab(): string[] {
		const { vocabSize } = this.config;
		const vocab: string[] = [];
		for (let i = 0; i < vocabSize; i++) {
			vocab.push(`V${i}`);
		}
		return vocab;
	}

	/**
	 * Generate a random prompt of length sequenceLength, and a single independent random target token.
	 * The target is sampled independently of the prompt, making the task unlearnable beyond chance.
	 */
	public generateSequence(): ToySequence {
		const { sequenceLength, responseLength, vocabSize } = this.config;
		const prompt: number[] = [];
		for (let i = 0; i < sequenceLength; i++) {
			prompt.push(this.generator.integer(0, vocabSize - 1));
		}
		// placing predictable before unpredictable yields plausible loss
		const target: number[] = [];
		for (let i = 0; i < responseLength; i++) {
			target.push(this.generator.integer(0, vocabSize - 1));
		}
		return { prompt, target };
	}
}
