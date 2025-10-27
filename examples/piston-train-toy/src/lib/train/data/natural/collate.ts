import type { Tensor } from '@piston-ml/piston-web';
import type { Random } from 'random-js';

import { type CollateWrapFunction, tensorWrap } from '../collate';

export type NaturalLanguageAutoregressiveBatch<T> = {
	tensors: [T, T]; // [input, target]
	samples: number[][];
};

export type NaturalLanguageBidirectionalBatch<T> = {
	tensors: [T, T, T]; // [input, labels, attentionMask]
	samples: number[][];
};

export function naturalLanguageAutoregressiveCollate<T = Tensor>(
	batch: number[][],
	options: { wrapFunction?: CollateWrapFunction<T> | null } = {}
): NaturalLanguageAutoregressiveBatch<T> {
	const wrap = options.wrapFunction === undefined ? tensorWrap : options.wrapFunction;

	const inputsArr: number[][] = [];
	const targetsArr: number[][] = [];
	for (const sequence of batch) {
		const L = sequence.length;
		if (L < 2) {
			inputsArr.push([]);
			targetsArr.push([]);
			continue;
		}
		inputsArr.push(sequence.slice(0, L - 1));
		targetsArr.push(sequence.slice(1));
	}

	const inputs = wrap ? wrap(inputsArr) : inputsArr;
	const targets = wrap ? wrap(targetsArr) : targetsArr;

	return { tensors: [inputs as T, targets as T], samples: batch };
}

export function naturalLanguageBidirectionalCollate<T = Tensor>(
	batch: number[][],
	options: {
		maskRatio: number;
		generator: Random;
		maskTokenId: number;
		wrapFunction?: CollateWrapFunction<T> | null;
	}
): NaturalLanguageBidirectionalBatch<T> {
	const { maskRatio, generator, maskTokenId, wrapFunction = tensorWrap } = options;

	const inputIdsArr: number[][] = [];
	const labelsArr: number[][] = [];
	const attentionMaskArr: number[][] = [];

	for (const sequence of batch) {
		const labels: number[] = new Array(sequence.length).fill(-100);
		const inputs: number[] = [...sequence];
		const attn: number[] = new Array(sequence.length).fill(1);

		for (let i = 0; i < sequence.length; i++) {
			if (generator.real(0, 1) < maskRatio) {
				labels[i] = sequence[i];
				inputs[i] = maskTokenId;
			}
		}

		inputIdsArr.push(inputs);
		labelsArr.push(labels);
		attentionMaskArr.push(attn);
	}

	const inputIds = wrapFunction ? wrapFunction(inputIdsArr) : inputIdsArr;
	const labels = wrapFunction ? wrapFunction(labelsArr) : labelsArr;
	const attentionMask = wrapFunction ? wrapFunction(attentionMaskArr) : attentionMaskArr;

	return { tensors: [inputIds as T, labels as T, attentionMask as T], samples: batch };
}
