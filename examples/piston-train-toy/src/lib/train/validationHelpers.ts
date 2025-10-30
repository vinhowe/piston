import type { TokenRollout } from '$lib/workspace/runs.svelte';

import { type Tensor } from '@piston-ml/piston-web';

import { tensorWrap } from './data';
import { generateStream } from './generate';
import {
	DecoderTransformer,
	EncoderDecoderTransformer,
	EncoderTransformer
} from './model/transformer';

export type DecoderGenerationOptions = {
	maxTokens?: number;
	stopTokens?: number[];
	temperature: number;
	useKvCache: boolean;
};

export type EncoderDecoderGenerationOptions = {
	maxTokens?: number;
	startToken?: number;
	stopTokens?: number[];
	temperature: number;
	useKvCache: boolean;
};

export type EncoderOnlyPredictionOptions = {
	attentionMask?: number[][]; // Only for transformer encoders
	temperature?: number;
};

export async function generateDecoderCompletions(
	model: DecoderTransformer,
	startSequences: number[][],
	options: DecoderGenerationOptions
): Promise<{ completions: TokenRollout[]; tokensPerSecond?: number }> {
	const { maxTokens, stopTokens, temperature, useKvCache } = options;

	model.eval();

	const completions: TokenRollout[] = [];
	let lastTPS: number | undefined;

	for (let bi = 0; bi < startSequences.length; bi++) {
		const startTokens = startSequences[bi] ?? [];
		let seq: number[] = [];
		const perStepProbs: number[][] = [];
		let stepIndex = 0;
		for await (const generationResult of generateStream(model, startTokens, {
			maxTokens,
			stopTokens: stopTokens ?? [],
			temperature,
			useKvCache
		})) {
			seq = generationResult.sequences[0] ? [...generationResult.sequences[0]] : [];
			lastTPS = generationResult.tokensPerSecond ?? lastTPS;
			if (generationResult.probs) {
				const probsArray = await (await generationResult.probs.to('cpu')).toVec();
				const [_b, v] = generationResult.probs.shape;
				const row: number[] = [];
				for (let vi = 0; vi < v; vi++) {
					row.push(probsArray[0 * v + vi]);
				}
				perStepProbs.push(row);
			}
			stepIndex++;
			if (typeof maxTokens === 'number' && stepIndex >= maxTokens) break;
		}
		completions[bi] = { tokenIds: seq, probs: perStepProbs };
	}

	model.train();

	return { completions, tokensPerSecond: lastTPS };
}

export async function generateEncoderDecoderCompletions(
	model: EncoderDecoderTransformer,
	sourceSequences: number[][],
	options: EncoderDecoderGenerationOptions
): Promise<{ completions: TokenRollout[]; tokensPerSecond?: number }> {
	const { maxTokens, startToken, stopTokens, temperature, useKvCache } = options;

	model.eval();

	const completions: TokenRollout[] = [];
	let lastTPS: number | undefined;

	for (let bi = 0; bi < sourceSequences.length; bi++) {
		const sourceTokens = sourceSequences[bi] ?? [];
		let seq: number[] = [];
		const perStepProbs: number[][] = [];
		let stepIndex = 0;
		for await (const generationResult of generateStream(model, sourceTokens, {
			maxTokens,
			startToken,
			stopTokens: stopTokens ?? [],
			temperature,
			useKvCache
		})) {
			seq = generationResult.sequences[0] ? [...generationResult.sequences[0]] : [];
			lastTPS = generationResult.tokensPerSecond ?? lastTPS;
			if (generationResult.probs) {
				const probsArray = await (await generationResult.probs.to('cpu')).toVec();
				const [_b, v] = generationResult.probs.shape;
				const row: number[] = [];
				for (let vi = 0; vi < v; vi++) {
					row.push(probsArray[0 * v + vi]);
				}
				perStepProbs.push(row);
			}
			stepIndex++;
			if (typeof maxTokens === 'number' && stepIndex >= maxTokens) break;
		}
		completions[bi] = { tokenIds: seq, probs: perStepProbs };
	}

	model.train();

	return { completions, tokensPerSecond: lastTPS };
}

export async function predictEncoderOnlyCompletions(
	model: EncoderTransformer,
	inputs: number[][],
	labels: number[][],
	options: EncoderOnlyPredictionOptions
): Promise<{ completions: TokenRollout[] }> {
	const { attentionMask, temperature } = options;

	model.eval();

	const inputsTensor = tensorWrap(inputs);
	let mlmLogits: Tensor | null = null;
	if (attentionMask) {
		const attentionMaskTensor = tensorWrap(attentionMask);
		[, , mlmLogits] = model.forward(await inputsTensor.to('gpu'), {
			attentionMask: await attentionMaskTensor.to('gpu')
		});
	} else {
		[, , mlmLogits] = model.forward(await inputsTensor.to('gpu'));
	}

	if (!mlmLogits) {
		throw new Error('No mlmLogits returned from encoder-only model');
	}

	let logitsAdj = mlmLogits;
	if (typeof temperature === 'number' && temperature > 0) {
		logitsAdj = logitsAdj.div(temperature);
	}
	const probs = logitsAdj.softmax(-1);

	let pred: Tensor; // [B, T]
	if (typeof temperature === 'number' && temperature > 0) {
		const [B, T, V] = probs.size();
		pred = probs
			.view([B * T, V])
			.multinomial(1, { replacement: false })
			.view([B, T]);
	} else {
		pred = probs.argmax({ dim: -1 });
	}
	const predArr = await (await pred.to('cpu')).toVec();
	const [B, T] = pred.size();

	const completions: TokenRollout[] = [];
	for (let bi = 0; bi < B; bi++) {
		const inputRow = inputs[bi] ?? [];
		const labelsRow = labels[bi] ?? [];
		const outRow: number[] = new Array(T);
		for (let ti = 0; ti < T; ti++) {
			const label = labelsRow[ti];
			const predId = predArr[bi * T + ti];
			outRow[ti] = label !== -100 ? predId : (inputRow[ti] ?? predId);
		}
		completions.push({ tokenIds: outRow, probs: [] });
	}

	model.train();

	return { completions };
}
