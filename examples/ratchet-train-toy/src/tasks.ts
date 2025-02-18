// Type for task generation functions
export type TaskGenerator = (
	maxNum: number,
	seqLen: number,
	batchSize: number
) => [number[][], number[][]];

// Helper function to convert a sequence to tokens
function sequenceToTokens(sequence: string): number[] {
	return sequence.split('').map((c) => c.charCodeAt(0));
}

// Helper function to pad a number to 2 digits
function pad2(num: number): string {
	return num.toString().padStart(2, '0');
}

// Helper function to pad a number to 3 digits
function pad3(num: number): string {
	return num.toString().padStart(3, '0');
}

export function generateTwoSumTask(
	maxNum: number,
	seqLen: number,
	batchSize: number
): [number[][], number[][]] {
	const input: number[][] = [];
	const target: number[][] = [];

	for (let b = 0; b < batchSize; b++) {
		const nums = Array.from({ length: seqLen }, () => Math.floor(Math.random() * maxNum));
		const i = Math.floor(Math.random() * seqLen);
		const j = Math.floor(Math.random() * seqLen);
		const sum = nums[i] + nums[j];

		const sequence = `${nums.map((n) => pad2(n)).join(',')}:${pad3(sum)}=${pad2(nums[i])},${pad2(nums[j])}`;
		const tokens = sequenceToTokens(sequence);

		input.push(tokens.slice(0, -1));
		target.push(tokens.slice(1));
	}

	return [input, target];
}

export function generateSortTask(
	maxNum: number,
	seqLen: number,
	batchSize: number
): [number[][], number[][]] {
	const input: number[][] = [];
	const target: number[][] = [];

	for (let b = 0; b < batchSize; b++) {
		const nums = Array.from({ length: seqLen }, () => Math.floor(Math.random() * maxNum));
		const sorted = [...nums].sort((a, b) => a - b);

		const sequence = `${nums.map((n) => pad2(n)).join(',')}:${sorted.map((n) => pad2(n)).join(',')}`;
		const tokens = sequenceToTokens(sequence);

		input.push(tokens.slice(0, -1));
		target.push(tokens.slice(1));
	}

	return [input, target];
}

export function generateAddTask(
	maxNum: number,
	seqLen: number,
	batchSize: number
): [number[][], number[][]] {
	const input: number[][] = [];
	const target: number[][] = [];

	for (let b = 0; b < batchSize; b++) {
		const num1 = Math.floor(Math.random() * maxNum);
		const num2 = Math.floor(Math.random() * maxNum);
		const sum = num1 + num2;

		const sequence = `${pad2(num1)}+${pad2(num2)}=${pad3(sum)}`;
		const tokens = sequenceToTokens(sequence);

		input.push(tokens.slice(0, -1));
		target.push(tokens.slice(1));
	}

	return [input, target];
}

export function generateModAddTask(
	maxNum: number,
	seqLen: number,
	batchSize: number
): [number[][], number[][]] {
	const input: number[][] = [];
	const target: number[][] = [];

	for (let b = 0; b < batchSize; b++) {
		const num1 = Math.floor(Math.random() * maxNum);
		const num2 = Math.floor(Math.random() * maxNum);
		const sum = (num1 + num2) % maxNum;

		const sequence = `${pad2(num1)}+${pad2(num2)}%${pad2(maxNum)}=${pad2(sum)}`;
		const tokens = sequenceToTokens(sequence);

		input.push(tokens.slice(0, -1));
		target.push(tokens.slice(1));
	}

	return [input, target];
}

export function generateCountTask(
	maxNum: number,
	seqLen: number,
	batchSize: number
): [number[][], number[][]] {
	const input: number[][] = [];
	const target: number[][] = [];

	for (let b = 0; b < batchSize; b++) {
		const start = Math.floor(Math.random() * (maxNum - seqLen));
		const sequence = Array.from({ length: seqLen }, (_, i) => pad2(start + i)).join(',');
		const tokens = sequenceToTokens(sequence);

		input.push(tokens.slice(0, -1));
		target.push(tokens.slice(1));
	}

	return [input, target];
}

export function generateSlapjackTask(
	maxNum: number,
	seqLen: number,
	batchSize: number
): [number[][], number[][]] {
	const input: number[][] = [];
	const target: number[][] = [];

	for (let b = 0; b < batchSize; b++) {
		const nums = Array.from({ length: seqLen }, () => Math.floor(Math.random() * maxNum));
		const jackPosition = Math.floor(Math.random() * seqLen);
		nums[jackPosition] = maxNum - 1; // Use max number as "jack"

		const sequence = `${nums.map((n) => pad2(n)).join(',')}:${pad2(jackPosition)}`;
		const tokens = sequenceToTokens(sequence);

		input.push(tokens.slice(0, -1));
		target.push(tokens.slice(1));
	}

	return [input, target];
}

export function generateZerosTask(
	_maxNum: number,
	seqLen: number,
	batchSize: number
): [number[][], number[][]] {
	const input: number[][] = [];
	const target: number[][] = [];

	for (let b = 0; b < batchSize; b++) {
		const sequence = Array(seqLen).fill('00').join(',');
		const tokens = sequenceToTokens(sequence);

		input.push(tokens.slice(0, -1));
		target.push(tokens.slice(1));
	}

	return [input, target];
}

// Map of task names to their generator functions
export const taskGenerators: Record<string, TaskGenerator> = {
	two_sum: generateTwoSumTask,
	sort: generateSortTask,
	add: generateAddTask,
	mod_add: generateModAddTask,
	count: generateCountTask,
	slapjack: generateSlapjackTask,
	zeros: generateZerosTask
};
