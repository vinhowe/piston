import type { GPT2ModelType } from '$lib/workspace/config';

import { GPT2_BLOCK_SIZE, GPT2_VOCAB_SIZE } from './gpt';

export interface GPT2Config {
	modelType: GPT2ModelType;
	nLayer: number;
	nHead: number;
	nEmbd: number;
	vocabSize: number;
	blockSize: number;
	embdPdrop: number;
	residPdrop: number;
	attnPdrop: number;
}

function getModelParametersFromType(modelType: GPT2ModelType): {
	nLayer: number;
	nHead: number;
	nEmbd: number;
} {
	switch (modelType) {
		case 'distilgpt2':
			return { nLayer: 6, nHead: 12, nEmbd: 768 };
		case 'gpt2':
			return { nLayer: 12, nHead: 12, nEmbd: 768 };
		case 'gpt2-medium':
			return { nLayer: 24, nHead: 16, nEmbd: 1024 };
		case 'gpt2-large':
			return { nLayer: 36, nHead: 20, nEmbd: 1280 };
		case 'gpt2-xl':
			return { nLayer: 48, nHead: 25, nEmbd: 1600 };
	}
}

export function buildGPT2Config(modelType: GPT2ModelType): GPT2Config {
	const { nLayer, nHead, nEmbd } = getModelParametersFromType(modelType);
	return {
		vocabSize: GPT2_VOCAB_SIZE,
		blockSize: GPT2_BLOCK_SIZE,
		modelType,
		nLayer,
		nHead,
		nEmbd,
		embdPdrop: 0.1,
		residPdrop: 0.1,
		attnPdrop: 0.1
	};
}
