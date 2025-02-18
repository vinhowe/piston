// Task-specific configuration types
interface TrainBatchConfig {
	batchSize: number;
}

interface NumberSequenceConfig {
	seqLen: number;
	maxNum: number;
}

interface AdditionConfig {
	maxNum: number;
}

interface ModAdditionConfig {
	maxNum: number;
}

interface FixedLengthConfig {
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
		}
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
		}
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
		}
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
		}
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
		}
	}
};

// Type for single sequence generation
type SequenceGenerator<K extends keyof TaskConfigMap> = (
	config: TaskConfigMap[K]
) => [string, string];

// Helper function to handle batch processing and tokenization
function generateTrainBatch<K extends keyof TaskConfigMap>(
	generator: SequenceGenerator<K>,
	config: TrainBatchConfig & TaskConfigMap[K]
): [number[][], number[][]] {
	const input: number[][] = [];
	const target: number[][] = [];

	for (let b = 0; b < config.batchSize; b++) {
		const sequence = generator(config).join('');
		const tokens = sequenceToTokens(sequence);
		input.push(tokens.slice(0, -1));
		target.push(tokens.slice(1));
	}

	return [input, target];
}

function generateEvalExample<K extends keyof TaskConfigMap>(
	generator: SequenceGenerator<K>,
	config: TaskConfigMap[K]
): [number[], number[]] {
	const [sequence, completion] = generator(config);
	const sequenceTokens = sequenceToTokens(sequence);
	const completionTokens = sequenceToTokens(completion);
	return [sequenceTokens, completionTokens];
}

// Helper function to convert a sequence to tokens
export function sequenceToTokens(sequence: string): number[] {
	return sequence.split('').map((c) => c.charCodeAt(0));
}

export function tokensToString(tokens: number[]): string {
	return tokens.map((t) => String.fromCharCode(t)).join('');
}

// A helper function to pad a number to the given number of digits
function pad(num: number, digits: number): string {
	return num.toString().padStart(digits, '0');
}

// Task-specific sequence generators
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

function sortSequence(config: NumberSequenceConfig): [string, string] {
	const { maxNum, seqLen } = config;
	const width = Math.floor(Math.log10(maxNum)) + 1;
	const nums = Array.from({ length: seqLen }, () => Math.floor(Math.random() * maxNum));
	const sorted = [...nums].sort((a, b) => a - b);
	return [`${nums.map((n) => pad(n, width)).join(',')}:`, `${sorted.map((n) => pad(n, width)).join(',')}`];
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
	return [`${pad(num1, width)}+${pad(num2, width)}%${pad(maxNum, width)}=`, `${pad(sum, width)}`];
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

// Update TrainBatchGenerator to be generic over task keys
export type TrainBatchGenerator<K extends keyof TaskConfigMap> = (
	config: TrainBatchConfig & TaskConfigMap[K]
) => [number[][], number[][]];

// Map of task names to their generator functions
export const trainBatchGenerators: {
	[K in keyof TaskConfigMap]: TrainBatchGenerator<K>;
} = {
	two_sum: (config: TrainBatchConfig & NumberSequenceConfig) => {
		const { seqLen, maxNum } = config;
		return generateTrainBatch((c) => twoSumSequence({ ...c, seqLen, maxNum }), config);
	},
	sort: (config: TrainBatchConfig & NumberSequenceConfig) => {
		const { seqLen, maxNum } = config;
		return generateTrainBatch((c) => sortSequence({ ...c, seqLen, maxNum }), config);
	},
	add: (config: TrainBatchConfig & AdditionConfig) => {
		const { maxNum } = config;
		return generateTrainBatch((c) => addSequence({ ...c, maxNum }), config);
	},
	mod_add: (config: TrainBatchConfig & ModAdditionConfig) => {
		const { maxNum } = config;
		return generateTrainBatch((c) => modAddSequence({ ...c, maxNum }), config);
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
		return generateTrainBatch((c) => zerosSequence({ ...c, seqLen }), config);
	}
};

export type EvalExampleGenerator<K extends keyof TaskConfigMap> = (
	config: TaskConfigMap[K]
) => [number[], number[]];

export type EvalMetric<K extends keyof TaskConfigMap> = (
	completion: number[],
	target: number[],
	sequence: number[],
	config: TaskConfigMap[K]
) => Array<boolean | null>;

export type EvalConfig<K extends keyof TaskConfigMap> = {
	metric: EvalMetric<K>;
	generator: EvalExampleGenerator<K>;
};

export const evalExampleGenerators: {
	[K in keyof TaskConfigMap]: EvalConfig<K>;
} = {
	two_sum: (() => {
		return {
			generator: (config: NumberSequenceConfig) => {
				const { seqLen, maxNum } = config as NumberSequenceConfig;
				return generateEvalExample((c) => twoSumSequence({ ...c, seqLen, maxNum }), config);
			},
			metric: (completion, _, sequence, config) => {
				// Passing is binary in this case, and depends on the whole completion
				const shouldPass = (() => {
					const completionString = tokensToString(completion);
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
	})(),
	sort: (() => {
		return {
			generator: (config: NumberSequenceConfig) => {
				const { seqLen, maxNum } = config as NumberSequenceConfig;
				return generateEvalExample((c) => sortSequence({ ...c, seqLen, maxNum }), config);
			},
			metric: (completion, target) => {
				return completion.map((c, i) => c === target[i]);
			}
		};
	})(),
	add: (() => {
		return {
			generator: (config: AdditionConfig) => {
				const { maxNum } = config as AdditionConfig;
				return generateEvalExample((c) => addSequence({ ...c, maxNum }), config);
			},
			metric: (completion, target) => {
				return completion.map((c, i) => c === target[i]);
			}
		};
	})(),
	mod_add: (() => {
		return {
			generator: (config: ModAdditionConfig) => {
				const { maxNum } = config as ModAdditionConfig;
				return generateEvalExample((c) => modAddSequence({ ...c, maxNum }), config);
			},
			metric: (completion, target) => {
				return completion.map((c, i) => c === target[i]);
			}
		};
	})(),
	zeros: (() => {
		return {
			generator: (config: FixedLengthConfig) => {
				const { seqLen } = config as FixedLengthConfig;
				return generateEvalExample((c) => zerosSequence({ ...c, seqLen }), config);
			},
			metric: (completion, target) => {
				return completion.map((c, i) => c === target[i]);
			}
		};
	})()
};
