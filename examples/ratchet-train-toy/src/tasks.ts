// Task-specific configuration types
export interface TrainBatchConfig {
	batchSize: number;
}

export interface NumberSequenceConfig {
	seqLen: number;
	maxNum: number;
	maskOutPrefix: boolean;
	includeCommas: boolean;
	includeColon: boolean;
}

export interface AdditionConfig {
	maxNum: number;
	includeExpressionTokens: boolean;
	maskOutPrefix: boolean;
}

export interface ModAdditionConfig {
	maxNum: number;
	includeExpressionTokens: boolean;
	maskOutPrefix: boolean;
}

export interface FixedLengthConfig {
	seqLen: number;
}

// Task metadata for UI configuration
export interface TaskParameter {
	name: string;
	description?: string;
	type?: 'number' | 'boolean'; // Default to 'number' if not specified
	min?: number; // Only for number type
	max?: number; // Only for number type
	default: number | boolean;
}

export interface TaskMetadata {
	name: string;
	description: string;
	parameters: Record<string, TaskParameter>;
	// For tasks using typical tokenization (zeros, count, slapjack)
	vocab?: string;
}

export const taskMetadata: Record<string, TaskMetadata> = {
	sort: {
		name: 'Sorting',
		description: 'Learn to sort a sequence of numbers',
		parameters: {
			seqLen: {
				name: 'Sequence Length',
				description: 'Number of items to sort',
				type: 'number',
				min: 2,
				max: 10,
				default: 2
			},
			maxNum: {
				name: 'Max Number',
				description: 'Maximum value in the sequence',
				type: 'number',
				min: 10,
				max: 26,
				default: 26
			},
			maskOutPrefix: {
				name: 'Mask Out Prefix',
				type: 'boolean',
				default: false
			},
			includeCommas: {
				name: 'Include Commas',
				type: 'boolean',
				default: false
			},
			includeColon: {
				name: 'Include Colon',
				type: 'boolean',
				default: true
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
				type: 'number',
				min: 10,
				max: 100,
				default: 100
			},
			maskOutPrefix: {
				name: 'Mask Out Prefix',
				type: 'boolean',
				default: false
			},
			includeExpressionTokens: {
				name: 'Include Expression Tokens (+, =)',
				type: 'boolean',
				default: false
			}
		}
	},
	mod_add: {
		name: 'Modular Addition',
		description: 'Learn to add two numbers with modulo',
		parameters: {
			maxNum: {
				name: 'Max Number',
				description: 'The modulo to use',
				type: 'number',
				min: 10,
				max: 113,
				default: 113
			},
			maskOutPrefix: {
				name: 'Mask Out Prefix',
				type: 'boolean',
				default: false
			},
			includeExpressionTokens: {
				name: 'Include Expression Tokens (+, =)',
				type: 'boolean',
				default: false
			}
		}
	},
	zeros: {
		name: 'Zeros',
		description: 'Learn to output zeros (baseline/debug task)',
		parameters: {
			seqLen: {
				name: 'Sequence Length',
				description: 'Length of the sequence',
				type: 'number',
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
				type: 'number',
				min: 2,
				max: 10,
				default: 3
			},
			maxNum: {
				name: 'Max Number',
				description: 'Maximum value in the sequence',
				type: 'number',
				min: 10,
				max: 26,
				default: 26
			},
			maskOutPrefix: {
				name: 'Mask Out Prefix',
				type: 'boolean',
				default: false
			},
			includeCommas: {
				name: 'Include Commas',
				type: 'boolean',
				default: false
			}
		}
	}
	/* 
	// (Commented: these tasks use the typical tokenization.)
	count: {
		name: 'Counting',
		description: 'Learn to count the occurrences of a character in a sequence',
		parameters: {
			seqLen: {
				name: 'Sequence Length',
				description: 'Length of the counting sequence',
				min: 2,
				max: 10,
				default: 5
			},
			maxNum: {
				name: 'Max Number',
				description: 'Maximum starting number',
				min: 10,
				max: 100,
				default: 100
			}
		}
	},
	slapjack: {
		name: 'Slapjack',
		description: 'Learn to find a specific number in a sequence',
		parameters: {
			seqLen: {
				name: 'Sequence Length',
				description: 'Length of the sequence',
				min: 2,
				max: 10,
				default: 5
			}
		}
	}
	*/
};

export type SimpleTokenizer = {
	vocab: Record<string, number>;
	ids: Record<number, string>;
	lastToken: number;
};

/**
 * Given a prompt and a completion (both as token arrays), concatenate them to form a full sequence.
 * Then, create an autoregressive pair where the input is fullSequence[:-1] and the target is fullSequence[1:],
 * with all tokens corresponding to the prompt (except for its final token) masked to -100.
 */
export function createAutoregressivePair<T>(
	prompt: T[],
	completion: T[],
	maskOutPrefix: boolean
): [T[], (T | number)[]] {
	const fullSequence = prompt.concat(completion);
	const input = fullSequence.slice(0, fullSequence.length - 1);
	const target = fullSequence.slice(1);
	// Mask the positions corresponding to the prompt (all but the last token of the prompt)
	if (maskOutPrefix) {
		const maskedTarget = target.map((tok, i) => (i < prompt.length - 1 ? -100 : tok));
		return [input, maskedTarget];
	} else {
		return [input, target];
	}
}

/**
 * For tasks that use number blocks we define custom tokenizers.
 *
 * For Add (and Mod-Add):
 *   token 0 → "+"
 *   token 1 → "="
 *   tokens 2 … maxNum+2 represent the numbers 0 … maxNum
 *   (When converting to a string, token id i ≥ 2 becomes `<${i-2}>`.)
 */
export function createAddTokenizer(
	maxNum: number,
	includeExpressionTokens: boolean
): SimpleTokenizer {
	const vocab: Record<string, number> = {};
	const ids: Record<number, string> = {};
	let offset = 0;
	if (includeExpressionTokens) {
		vocab['+'] = 0;
		ids[0] = '+';
		vocab['='] = 1;
		ids[1] = '=';
		offset = 2;
	}
	const width = maxNum.toString().length;
	for (let n = 0; n < maxNum; n++) {
		const token = `<${n.toString().padStart(width, '0')}>`;
		const id = n + offset;
		vocab[token] = id;
		ids[id] = token;
	}
	return { vocab, ids, lastToken: maxNum + offset };
}

/**
 * For Sort:
 *   token 0 → ":"
 *   token 1 → ","
 *   tokens 2 … maxNum+2 represent the numbers 0 … maxNum.
 */
export function createSortTokenizer(
	maxNum: number,
	includeColon: boolean,
	includeCommas: boolean
): SimpleTokenizer {
	const vocab: Record<string, number> = {};
	const ids: Record<number, string> = {};
	let offset = 0;
	if (includeColon) {
		vocab[':'] = 0;
		ids[0] = ':';
		offset++;
	}
	if (includeCommas) {
		vocab[','] = 1;
		ids[1] = ',';
		offset++;
	}
	const aCharCode = 'A'.charCodeAt(0);
	for (let n = 0; n < maxNum; n++) {
		const token = String.fromCharCode(aCharCode + n);
		const id = n + offset;
		vocab[token] = id;
		ids[id] = token;
	}
	return { vocab, ids, lastToken: maxNum + offset };
}

/**
 * For Two-Sum:
 *   token 0 → ":"
 *   token 1 → "="
 *   token 2 → ","
 *   tokens 3 … maxNum+3 represent the numbers 0 … maxNum.
 */
export function createTwoSumTokenizer(maxNum: number, includeCommas: boolean): SimpleTokenizer {
	const vocab: Record<string, number> = {};
	const ids: Record<number, string> = {};
	let offset = 2;
	vocab[':'] = 0;
	ids[0] = ':';
	vocab['='] = 1;
	ids[1] = '=';
	if (includeCommas) {
		vocab[','] = 2;
		ids[2] = ',';
		offset++;
	}
	const aCharCode = 'A'.charCodeAt(0);
	for (let n = 0; n < maxNum; n++) {
		const token = String.fromCharCode(aCharCode + n);
		const id = n + 3;
		vocab[token] = id;
		ids[id] = token;
	}
	return { vocab, ids, lastToken: maxNum + offset };
}

/**
 * Typical tokenizer (used by zeros, count, slapjack).
 */
export function vocabToSimpleTokenizer(vocab: string): SimpleTokenizer {
	const tokens = vocab.split('');
	const vocabMap = tokens.reduce(
		(acc, c, i) => {
			acc[c] = i;
			return acc;
		},
		{} as Record<string, number>
	);
	const ids = tokens.reduce(
		(acc, c, i) => {
			acc[i] = c;
			return acc;
		},
		{} as Record<number, string>
	);
	return { vocab: vocabMap, ids, lastToken: tokens.length };
}

// A helper for typical (character-based) tokenization.
export function sequenceToTokens(sequence: string, tokenizer: SimpleTokenizer): number[] {
	return sequence.split('').map((c) => tokenizer.vocab[c]);
}

// --------------------------
// Task-Specific Sequence Generators
// (These return token ID arrays directly for the number‐block tasks.)
// --------------------------

/**
 * Addition: generate two addends (in [0, maxNum]) chosen so that their sum ≤ maxNum.
 * Mapping: number token = number + 2, plus = 0, equals = 1.
 */
export function addSequenceTokenized(config: AdditionConfig): [number[], number[]] {
	const { maxNum, includeExpressionTokens } = config;
	const num1 = Math.floor(Math.random() * (maxNum + 1));
	const num2 = Math.floor(Math.random() * (maxNum - num1 + 1));
	const sum = num1 + num2;
	let prompt;
	if (includeExpressionTokens) {
		prompt = [num1 + 2, 0, num2 + 2, 1];
	} else {
		prompt = [num1 + 2, num2 + 2];
	}
	const target = [sum + 2];
	return [prompt, target];
}

/**
 * Modular Addition: generate two numbers (in [0, maxNum)) and compute (num1+num2)%maxNum.
 * Mapping is identical to Addition.
 */
export function modAddSequenceTokenized(config: ModAdditionConfig): [number[], number[]] {
	const { maxNum, includeExpressionTokens } = config;
	const num1 = Math.floor(Math.random() * maxNum);
	const num2 = Math.floor(Math.random() * maxNum);
	const sum = (num1 + num2) % maxNum;
	let prompt;
	if (includeExpressionTokens) {
		prompt = [num1 + 2, 0, num2 + 2, 1];
	} else {
		prompt = [num1 + 2, num2 + 2];
	}
	const target = [sum + 2];
	return [prompt, target];
}

/**
 * Sorting: generate a sequence of numbers (each in [0, maxNum]) and return:
 *  - prompt: the unsorted sequence with commas (token 1) between numbers and a colon (token 0) at the end.
 *  - target: the sorted sequence with commas between numbers.
 * Mapping: number token = number + 2.
 */
export function sortSequenceTokenized(
	config: NumberSequenceConfig,
	tokenizer: SimpleTokenizer
): [number[], number[]] {
	const { seqLen, maxNum, includeCommas, includeColon } = config;
	const nums = Array.from({ length: seqLen }, () => Math.floor(Math.random() * maxNum));
	const sorted = [...nums].sort((a, b) => a - b);
	const prompt: number[] = [];
	for (let i = 0; i < nums.length; i++) {
		prompt.push(tokenizer.vocab[String.fromCharCode('A'.charCodeAt(0) + nums[i])]);
		if (i < nums.length - 1) {
			if (includeCommas) {
				prompt.push(tokenizer.vocab[',']); // comma
			}
		} else if (includeColon) {
			prompt.push(tokenizer.vocab[':']); // colon
		}
	}
	const target: number[] = [];
	for (let i = 0; i < sorted.length; i++) {
		target.push(tokenizer.vocab[String.fromCharCode('A'.charCodeAt(0) + sorted[i])]);
		if (includeCommas && i < sorted.length - 1) {
			target.push(tokenizer.vocab[',']); // comma
		}
	}
	return [prompt, target];
}

/**
 * Two-Sum: generate a sequence of numbers (each in [0, maxNum]) and two random indices.
 * The prompt consists of:
 *   - the unsorted sequence (with commas, token 2)
 *   - a colon (token 0), the sum (token = sum+3) and an equals sign (token 1).
 * The target is the two numbers (from the chosen indices) separated by a comma.
 * Mapping: number token = number + 3.
 */
export function twoSumSequenceTokenized(config: NumberSequenceConfig): [number[], number[]] {
	const { seqLen, maxNum } = config;
	const nums = Array.from({ length: seqLen }, () => Math.floor(Math.random() * (maxNum / 2)));
	const i = Math.floor(Math.random() * seqLen);
	const j = Math.floor(Math.random() * seqLen);
	const sum = nums[i] + nums[j];
	const prompt: number[] = [];
	for (let k = 0; k < nums.length; k++) {
		prompt.push(nums[k] + 3);
		if (k < nums.length - 1) {
			prompt.push(2); // comma
		}
	}
	prompt.push(0); // colon
	prompt.push(sum + 3);
	prompt.push(1); // equals
	const target: number[] = [nums[i] + 3, 2, nums[j] + 3];
	return [prompt, target];
}

/**
 * Zeros: (Baseline) returns string outputs and uses typical tokenization.
 */
export function zerosSequence(config: FixedLengthConfig): [string, string] {
	const { seqLen } = config;
	const sequence = Array(seqLen - 1)
		.fill('0')
		.join('');
	return ['0', sequence];
}

// (The counting and slapjack tasks are left commented out. Note that slapjack will continue to use the typical tokenization.)

export interface TaskSpec<Config> {
	metadata: TaskMetadata;
	/**
	 * Build a tokenizer (which may depend on the config).
	 */
	createTokenizer: (config: Config) => SimpleTokenizer;
	/**
	 * Generate a training batch given the combined task config.
	 */
	trainBatch: (config: TrainBatchConfig & Config) => [number[][], number[][]];
	/**
	 * Return the evaluation configuration, including a generator and a metric.
	 */
	eval: (config: Config) => {
		example: () => [number[], number[]];
		metric: (completion: number[], target: number[], sequence?: number[]) => Array<boolean | null>;
	};
}

export type TaskConfigMap = {
	two_sum: NumberSequenceConfig;
	sort: NumberSequenceConfig;
	add: AdditionConfig;
	mod_add: ModAdditionConfig;
	zeros: FixedLengthConfig;
	// count: NumberSequenceConfig;
	// slapjack: FixedLengthConfig;
};

export const tasks: { [K in keyof TaskConfigMap]: TaskSpec<TaskConfigMap[K]> } = {
	add: (() => {
		const createTokenizer = (config: AdditionConfig) =>
			createAddTokenizer(config.maxNum, config.includeExpressionTokens);
		return {
			metadata: taskMetadata.add,
			createTokenizer,
			trainBatch: (config: TrainBatchConfig & AdditionConfig) => {
				const inputs: number[][] = [];
				const targets: (number | -100)[][] = [];
				for (let i = 0; i < config.batchSize; i++) {
					const [prompt, completion] = addSequenceTokenized(config);
					const [inputSeq, targetSeq] = createAutoregressivePair(
						prompt,
						completion,
						config.maskOutPrefix
					);
					inputs.push(inputSeq);
					targets.push(targetSeq);
				}
				return [inputs, targets];
			},
			eval: (config: AdditionConfig) => {
				return {
					example: () => addSequenceTokenized(config),
					metric: (completion, target) => {
						if (completion.length !== target.length) return completion.map(() => false);
						return completion.map((t, i) => t === target[i]);
					}
				};
			}
		};
	})(),
	mod_add: (() => {
		const createTokenizer = (config: ModAdditionConfig) =>
			createAddTokenizer(config.maxNum, config.includeExpressionTokens);
		return {
			metadata: taskMetadata.mod_add,
			createTokenizer,
			trainBatch: (config: TrainBatchConfig & ModAdditionConfig) => {
				const inputs: number[][] = [];
				const targets: (number | -100)[][] = [];
				for (let i = 0; i < config.batchSize; i++) {
					const [prompt, completion] = modAddSequenceTokenized(config);
					const [inputSeq, targetSeq] = createAutoregressivePair(
						prompt,
						completion,
						config.maskOutPrefix
					);
					inputs.push(inputSeq);
					targets.push(targetSeq);
				}
				return [inputs, targets];
			},
			eval: (config: ModAdditionConfig) => {
				return {
					example: () => modAddSequenceTokenized(config),
					metric: (completion, target) => {
						if (completion.length !== target.length) return completion.map(() => false);
						return completion.map((t, i) => t === target[i]);
					}
				};
			}
		};
	})(),
	sort: (() => {
		const createTokenizer = (config: NumberSequenceConfig) =>
			createSortTokenizer(config.maxNum, config.includeColon, config.includeCommas);
		return {
			metadata: taskMetadata.sort,
			createTokenizer,
			trainBatch: (config: TrainBatchConfig & NumberSequenceConfig) => {
				const tokenizer = createTokenizer(config);
				const inputs: number[][] = [];
				const targets: (number | -100)[][] = [];
				for (let i = 0; i < config.batchSize; i++) {
					const [prompt, completion] = sortSequenceTokenized(config, tokenizer);
					const [inputSeq, targetSeq] = createAutoregressivePair(
						prompt,
						completion,
						config.maskOutPrefix
					);
					inputs.push(inputSeq);
					targets.push(targetSeq);
				}
				return [inputs, targets];
			},
			eval: (config: NumberSequenceConfig) => {
				const tokenizer = createTokenizer(config);
				return {
					example: () => sortSequenceTokenized(config, tokenizer),
					metric: (completion, target) => {
						if (completion.length !== target.length) return completion.map(() => false);
						return completion.map((t, i) => t === target[i]);
					}
				};
			}
		};
	})(),
	two_sum: (() => {
		const createTokenizer = (config: NumberSequenceConfig) =>
			createTwoSumTokenizer(config.maxNum, config.includeCommas);
		return {
			metadata: taskMetadata.two_sum,
			createTokenizer,
			trainBatch: (config: TrainBatchConfig & NumberSequenceConfig) => {
				const inputs: number[][] = [];
				const targets: (number | -100)[][] = [];
				for (let i = 0; i < config.batchSize; i++) {
					const [prompt, completion] = twoSumSequenceTokenized(config);
					const [inputSeq, targetSeq] = createAutoregressivePair(
						prompt,
						completion,
						config.maskOutPrefix
					);
					inputs.push(inputSeq);
					targets.push(targetSeq);
				}
				return [inputs, targets];
			},
			eval: (config: NumberSequenceConfig) => {
				return {
					example: () => twoSumSequenceTokenized(config),
					metric: (completion, target) => {
						if (completion.length !== target.length) return completion.map(() => false);
						return completion.map((t, i) => t === target[i]);
					}
				};
			}
		};
	})(),
	zeros: {
		metadata: taskMetadata.zeros,
		createTokenizer: (_config: FixedLengthConfig) =>
			vocabToSimpleTokenizer(taskMetadata.zeros.vocab!),
		trainBatch: (config: TrainBatchConfig & FixedLengthConfig) => {
			const tokenizer = vocabToSimpleTokenizer(taskMetadata.zeros.vocab!);
			const inputs: number[][] = [];
			const targets: (number | -100)[][] = [];
			for (let i = 0; i < config.batchSize; i++) {
				// Zeros returns strings; tokenize them.
				const [promptStr, completionStr] = zerosSequence(config);
				const promptTokens = sequenceToTokens(promptStr, tokenizer);
				const completionTokens = sequenceToTokens(completionStr, tokenizer);
				const [inputSeq, targetSeq] = createAutoregressivePair(
					promptTokens,
					completionTokens,
					false
				);
				inputs.push(inputSeq);
				targets.push(targetSeq);
			}
			return [inputs, targets];
		},
		eval: (config: FixedLengthConfig) => {
			const tokenizer = vocabToSimpleTokenizer(taskMetadata.zeros.vocab!);
			return {
				example: () => {
					const [promptStr, targetStr] = zerosSequence(config);
					return [sequenceToTokens(promptStr, tokenizer), sequenceToTokens(targetStr, tokenizer)];
				},
				metric: (completion, target) => {
					if (completion.length !== target.length) return completion.map(() => false);
					return completion.map((t, i) => t === target[i]);
				}
			};
		}
	}
	// TODO: Add count and slapjack
};
