// Task-specific configuration types
export interface TrainBatchConfig {
	batchSize: number;
}

export interface NumberSequenceConfig {
	seqLen: number;
	maxNum: number;
}

export interface AdditionConfig {
	maxNum: number;
}

export interface ModAdditionConfig {
	maxNum: number;
}

export interface FixedLengthConfig {
	seqLen: number;
}

// Task metadata for UI configuration
export interface TaskParameter {
	name: string;
	description: string;
	min: number;
	max: number;
	default: number;
}

export interface TaskMetadata {
	name: string;
	description: string;
	parameters: Record<string, TaskParameter>;
	vocab: string;
}

export const taskMetadata: Record<string, TaskMetadata> = {
	sort: {
		name: 'Sorting',
		description: 'Learn to sort a sequence of numbers',
		parameters: {
			seqLen: {
				name: 'Sequence Length',
				description: 'Length of the sequence to sort',
				min: 2,
				max: 10,
				default: 2
			},
			maxNum: {
				name: 'Max Number',
				description: 'Maximum value in the sequence',
				min: 10,
				max: 100,
				default: 100
			}
		},
		vocab: '0123456789,:'
	},
	add: {
		name: 'Addition',
		description: 'Learn to add two numbers',
		parameters: {
			maxNum: {
				name: 'Max Number',
				description: 'Maximum value for each addend',
				min: 10,
				max: 100,
				default: 100
			}
		},
		vocab: '0123456789+='
	},
	mod_add: {
		name: 'Modular Addition',
		description: 'Learn to add two numbers with modulo',
		parameters: {
			maxNum: {
				name: 'Modulo',
				description: 'The modulo to use',
				min: 10,
				max: 113,
				default: 113
			}
		},
		vocab: '0123456789+='
	},
	// count: {
	// 	name: 'Counting',
	// 	description: 'Learn to count the occurrences of a character in a sequence',
	// 	parameters: {
	// 		seqLen: {
	// 			name: 'Sequence Length',
	// 			description: 'Length of the counting sequence',
	// 			min: 2,
	// 			max: 10,
	// 			default: 5
	// 		},
	// 		maxNum: {
	// 			name: 'Max Number',
	// 			description: 'Maximum starting number',
	// 			min: 10,
	// 			max: 100,
	// 			default: 100
	// 		}
	// 	}
	// },
	// slapjack: {
	// 	name: 'Slapjack',
	// 	description: 'Learn to find a specific number in a sequence',
	// 	parameters: {
	// 		seqLen: {
	// 			name: 'Sequence Length',
	// 			description: 'Length of the sequence',
	// 			min: 2,
	// 			max: 10,
	// 			default: 5
	// 		}
	// 	}
	// },
	zeros: {
		name: 'Zeros',
		description: 'Learn to output zeros (baseline/debug task)',
		parameters: {
			seqLen: {
				name: 'Sequence Length',
				description: 'Length of the sequence',
				min: 2,
				max: 10,
				default: 5
			}
		},
		vocab: '0123456789'
	},
	two_sum: {
		name: 'Two Sum',
		description: 'Find two numbers in a sequence that sum to a target',
		parameters: {
			seqLen: {
				name: 'Sequence Length',
				description: 'Length of the sequence',
				min: 2,
				max: 10,
				default: 5
			},
			maxNum: {
				name: 'Max Number',
				description: 'Maximum value in the sequence',
				min: 10,
				max: 100,
				default: 100
			}
		},
		vocab: '0123456789:=,'
	}
};

// Type for single sequence generation
type SequenceGenerator<K extends keyof TaskConfigMap> = (
	config: TaskConfigMap[K]
) => [string, string];

type SimpleTokenizer = {
	vocab: Record<string, number>;
	ids: Record<number, string>;
	endToken: number;
};

function vocabToSimpleTokenizer(vocab: string): SimpleTokenizer {
	return {
		vocab: vocab.split('').reduce(
			(acc, c, i) => {
				acc[c] = i;
				return acc;
			},
			{} as Record<string, number>
		),
		ids: vocab.split('').reduce(
			(acc, c, i) => {
				acc[i] = c;
				return acc;
			},
			{} as Record<number, string>
		),
		endToken: vocab.length
	};
}

// Helper function to handle batch processing and tokenization
function generateTrainBatch<K extends keyof TaskConfigMap>(
	generator: SequenceGenerator<K>,
	config: TrainBatchConfig & TaskConfigMap[K],
	tokenizer: SimpleTokenizer
): [number[][], number[][]] {
	const input: number[][] = [];
	const target: number[][] = [];

	for (let b = 0; b < config.batchSize; b++) {
		const [prompt, completion] = generator(config);
		const sequence = prompt + completion;
		const tokens = sequenceToTokens(sequence, tokenizer);
		const promptTokens = sequenceToTokens(prompt, tokenizer);

		input.push(tokens.slice(0, -1));
		// Create target with -100 for prompt tokens and actual tokens for completion
		const targetTokens = tokens
			.slice(1)
			.map((token, i) => (i < promptTokens.length - 1 ? -100 : token));
		target.push(targetTokens);
		// const sequence = generator(config).join('');
		// const tokens = sequenceToTokens(sequence, tokenizer);
		// input.push(tokens.slice(0, -1));
		// target.push(tokens.slice(1));
	}

	return [input, target];
}

function generateEvalExample<K extends keyof TaskConfigMap>(
	generator: SequenceGenerator<K>,
	config: TaskConfigMap[K],
	tokenizer: SimpleTokenizer
): [number[], number[]] {
	const [sequence, completion] = generator(config);
	const sequenceTokens = sequenceToTokens(sequence, tokenizer);
	const completionTokens = sequenceToTokens(completion, tokenizer);
	return [sequenceTokens, completionTokens];
}

// Helper function to convert a sequence to tokens
export function sequenceToTokens(sequence: string, tokenizer: SimpleTokenizer): number[] {
	return sequence.split('').map((c) => tokenizer.vocab[c]);
}

export function tokensToString(tokens: number[], tokenizer: SimpleTokenizer): string {
	return tokens.map((t) => tokenizer.ids[t]).join('');
}

// A helper function to pad a number to the given number of digits
function pad(num: number, digits: number): string {
	return num.toString().padStart(digits, '0');
}

// Task-specific sequence generators
function sortSequence(config: NumberSequenceConfig): [string, string] {
	const { maxNum, seqLen } = config;
	const width = Math.floor(Math.log10(maxNum)) + 1;
	const nums = Array.from({ length: seqLen }, () => Math.floor(Math.random() * maxNum));
	const sorted = [...nums].sort((a, b) => a - b);
	return [
		`${nums.map((n) => pad(n, width)).join(',')}:`,
		`${sorted.map((n) => pad(n, width)).join(',')}`
	];
}

function twoSumSequence(config: NumberSequenceConfig): [string, string] {
	const { maxNum, seqLen } = config;
	const width = Math.floor(Math.log10(maxNum)) + 1;
	const nums = Array.from({ length: seqLen }, () => Math.floor(Math.random() * maxNum));
	const i = Math.floor(Math.random() * seqLen);
	const j = Math.floor(Math.random() * seqLen);
	const sum = nums[i] + nums[j];
	return [
		`${nums.map((n) => pad(n, width)).join(',')}:${pad(sum, width)}=`,
		`${pad(nums[i], width)},${pad(nums[j], width)}`
	];
}

function addSequence(config: AdditionConfig): [string, string] {
	const { maxNum } = config;
	const width = Math.floor(Math.log10(maxNum)) + 1;
	const num1 = Math.floor(Math.random() * maxNum);
	const num2 = Math.floor(Math.random() * maxNum);
	const sum = num1 + num2;
	return [`${pad(num1, width)}+${pad(num2, width)}=`, `${pad(sum, width)}`];
}

function modAddSequence(config: ModAdditionConfig): [string, string] {
	const { maxNum } = config;
	const width = Math.floor(Math.log10(maxNum)) + 1;
	const num1 = Math.floor(Math.random() * maxNum);
	const num2 = Math.floor(Math.random() * maxNum);
	const sum = (num1 + num2) % maxNum;
	return [`${pad(num1, width)}+${pad(num2, width)}=`, `${pad(sum, width)}`];
}

// function countSequence(config: NumberSequenceConfig): [string, string] {
// 	const { maxNum, seqLen } = config;
// 	const start = Math.floor(Math.random() * (maxNum - seqLen));
// 	const sequence = Array.from({ length: seqLen - 1 }, (_, i) => pad2(start + i)).join(',');
// 	return [sequence, `,${pad2(start + seqLen - 1)}`];
// }

// function slapjackSequence(config: FixedLengthConfig): [string, string] {
// 	const { seqLen } = config;
// 	const maxNum = 100; // Fixed max number for consistency
// 	const nums = Array.from({ length: seqLen }, () => Math.floor(Math.random() * maxNum));
// 	const jackPosition = Math.floor(Math.random() * seqLen);
// 	nums[jackPosition] = maxNum - 1; // Use max number as "jack"
// 	return [
// 		`${nums.map((n) => pad2(n)).join(',')}`,
// 		`:${pad2(jackPosition)}`
// 	];
// }

function zerosSequence(config: FixedLengthConfig): [string, string] {
	const { seqLen } = config;
	const sequence = Array(seqLen - 1)
		.fill('0')
		.join('');
	return ['', sequence];
}

export type TaskConfigMap = {
	two_sum: NumberSequenceConfig;
	sort: NumberSequenceConfig;
	add: AdditionConfig;
	mod_add: ModAdditionConfig;
	zeros: FixedLengthConfig;
};

export type TrainBatchGenerator<K extends keyof TaskConfigMap> = (
	config: TrainBatchConfig & TaskConfigMap[K]
) => [number[][], number[][]];

// Map of task names to their generator functions
export const trainBatchGenerators: {
	[K in keyof TaskConfigMap]: TrainBatchGenerator<K>;
} = {
	two_sum: (config: TrainBatchConfig & NumberSequenceConfig) => {
		const { seqLen, maxNum } = config;
		const vocab = taskMetadata.two_sum.vocab;
		const tokenizer = vocabToSimpleTokenizer(vocab);
		return generateTrainBatch((c) => twoSumSequence({ ...c, seqLen, maxNum }), config, tokenizer);
	},
	sort: (config: TrainBatchConfig & NumberSequenceConfig) => {
		const { seqLen, maxNum } = config;
		const vocab = taskMetadata.sort.vocab;
		const tokenizer = vocabToSimpleTokenizer(vocab);
		return generateTrainBatch((c) => sortSequence({ ...c, seqLen, maxNum }), config, tokenizer);
	},
	add: (config: TrainBatchConfig & AdditionConfig) => {
		const { maxNum } = config;
		const vocab = taskMetadata.add.vocab;
		const tokenizer = vocabToSimpleTokenizer(vocab);
		return generateTrainBatch((c) => addSequence({ ...c, maxNum }), config, tokenizer);
	},
	mod_add: (config: TrainBatchConfig & ModAdditionConfig) => {
		const { maxNum } = config;
		const vocab = taskMetadata.mod_add.vocab;
		const tokenizer = vocabToSimpleTokenizer(vocab);
		return generateTrainBatch((c) => modAddSequence({ ...c, maxNum }), config, tokenizer);
	},
	// count: (config: BaseTaskConfig) => {
	// 	const { seqLen, maxNum } = config as NumberSequenceConfig;
	// 	return generateBatch((c) => countSequence({ ...c, seqLen, maxNum }), config);
	// },
	// slapjack: (config: BaseTaskConfig) => {
	// 	const { seqLen } = config as FixedLengthConfig;
	// 	return generateBatch((c) => slapjackSequence({ ...c, seqLen }), config);
	// },
	zeros: (config: TrainBatchConfig & FixedLengthConfig) => {
		const { seqLen } = config;
		const vocab = taskMetadata.zeros.vocab;
		const tokenizer = vocabToSimpleTokenizer(vocab);
		return generateTrainBatch((c) => zerosSequence({ ...c, seqLen }), config, tokenizer);
	}
};

export type EvalExampleGenerator = () => [number[], number[]];

export type EvalMetric = (
	completion: number[],
	target: number[],
	sequence: number[]
) => Array<boolean | null>;

export type EvalConfig = {
	metric: EvalMetric;
	generator: EvalExampleGenerator;
	tokenizer: SimpleTokenizer;
};

export type EvalConfigGenerator<K extends keyof TaskConfigMap> = (
	config: TaskConfigMap[K]
) => EvalConfig;

export const evalExampleGenerators: {
	[K in keyof TaskConfigMap]: EvalConfigGenerator<K>;
} = {
	two_sum: (config: NumberSequenceConfig) => {
		const tokenizer = vocabToSimpleTokenizer(taskMetadata.two_sum.vocab);
		return {
			tokenizer,
			generator: () => {
				const { seqLen, maxNum } = config as NumberSequenceConfig;
				return generateEvalExample(
					(c) => twoSumSequence({ ...c, seqLen, maxNum }),
					config,
					tokenizer
				);
			},
			metric: (completion, _, _sequence) => {
				// Passing is binary in this case, and depends on the whole completion
				const shouldPass = (() => {
					const completionString = tokensToString(completion, tokenizer);
					// Count the number of commas
					const numCommas = (completionString.match(/,/g) || []).length;
					if (numCommas !== 1) {
						return false;
					}

					// Use log to find the number length
					const numberLength = Math.floor(Math.log10(config.maxNum));
					// Check that there are two numbers of the correct length, separated
					// by a comma
					// Match pattern of two 2-digit numbers separated by comma
					const pattern = new RegExp(`^\\d{${numberLength}},\\d{${numberLength}}$`);
					return pattern.test(completionString);
				})();
				return completion.map(() => shouldPass);
			}
		};
	},
	sort: (config: NumberSequenceConfig) => {
		const tokenizer = vocabToSimpleTokenizer(taskMetadata.sort.vocab);
		return {
			tokenizer,
			generator: () => {
				const { seqLen, maxNum } = config as NumberSequenceConfig;
				return generateEvalExample(
					(c) => sortSequence({ ...c, seqLen, maxNum }),
					config,
					tokenizer
				);
			},
			metric: (completion, target) => {
				return completion.map((c, i) => c === target[i]);
			}
		};
	},
	add: (config: AdditionConfig) => {
		const tokenizer = vocabToSimpleTokenizer(taskMetadata.add.vocab);
		return {
			tokenizer,
			generator: () => {
				const { maxNum } = config as AdditionConfig;
				return generateEvalExample((c) => addSequence({ ...c, maxNum }), config, tokenizer);
			},
			metric: (completion, target) => {
				return completion.map((c, i) => c === target[i]);
			}
		};
	},
	mod_add: (config: ModAdditionConfig) => {
		const tokenizer = vocabToSimpleTokenizer(taskMetadata.mod_add.vocab);
		return {
			tokenizer,
			generator: () => {
				const { maxNum } = config as ModAdditionConfig;
				return generateEvalExample((c) => modAddSequence({ ...c, maxNum }), config, tokenizer);
			},
			metric: (completion, target) => {
				return completion.map((c, i) => c === target[i]);
			}
		};
	},
	zeros: (config: FixedLengthConfig) => {
		const tokenizer = vocabToSimpleTokenizer(taskMetadata.zeros.vocab);
		return {
			tokenizer,
			generator: () => {
				const { seqLen } = config as FixedLengthConfig;
				return generateEvalExample((c) => zerosSequence({ ...c, seqLen }), config, tokenizer);
			},
			metric: (completion, target) => {
				return completion.map((c, i) => c === target[i]);
			}
		};
	}
};
