import type { TokenRollout } from '$lib/workspace/runs.svelte';

import { pin, weak } from '@piston-ml/piston-web';

import { generateGPTStream } from './generate';
import { GPT } from './model/gpt';

export type DecoderGenerationOptions = {
	maxTokens?: number;
	stopTokens?: number[];
	temperature: number;
	useKvCache: boolean;
};

export async function generateDecoderCompletions(
	model: GPT,
	startSequences: number[][],
	options: DecoderGenerationOptions
): Promise<{ completions: TokenRollout[]; tokensPerSecond?: number }> {
	const { maxTokens, stopTokens, temperature, useKvCache } = options;

	model.eval();

	const completions: TokenRollout[] = [];
	let lastTPS: number | undefined;

	for (let bi = 0; bi < startSequences.length; bi++) {
		await weak(async () => {
			const startTokens = startSequences[bi] ?? [];
			let seq: number[] = [];
			const perStepProbs: number[][] = [];
			let stepIndex = 0;
			for await (const generationResult of generateGPTStream(model, startTokens, {
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
			completions[bi] = { tokenIds: seq, probs: pin(perStepProbs) };
		});
	}

	model.train();

	return { completions, tokensPerSecond: lastTPS };
}
