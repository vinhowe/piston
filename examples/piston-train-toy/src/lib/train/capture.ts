import type { BaseStepData } from '$lib/workspace/runs.svelte';

import {
	CapturePlan,
	type CaptureResult,
	CaptureSession,
	Module,
	pin,
	Tensor,
	type TensorQuery
} from '@piston-ml/piston-web';

import type {
	BidirectionalBatchType,
	EncoderDecoderBatchType,
	GeneratableModel,
	NaturalCollateFnType,
	PistonCollateFnType,
	ToyCollateFnType
} from './types';
import type { ValidationExamples } from './validation';

import { RNNDecoder, RNNEncoder, RNNEncoderDecoder } from './model/rnn';
import {
	DecoderTransformer,
	EncoderDecoderTransformer,
	EncoderTransformer
} from './model/transformer';

export interface CaptureMatch {
	matchId: number;
	buffer?: GPUBuffer | null;
	type: 'module' | 'parameter' | 'op';
	op?: string;
	parameter?: string;
	moduleSite?: 'input' | 'output';
	batchIndex?: number;
	source: TensorQuery;
	queryIndex?: number;
	path: string;
	shape: number[];
	tensor: Tensor;
	mean?: number;
	variance?: number;
}

export type RemoteCaptureMatch = Omit<CaptureMatch, 'buffer' | 'tensor'>;

export type CaptureStep = BaseStepData & {
	type: 'capture';
	matches: CaptureMatch[];
};

export type RemoteCaptureStep = BaseStepData & {
	type: 'capture';
	matches: RemoteCaptureMatch[];
};

export function createCapturePlan(query: string, module: Module): CapturePlan | null {
	const capturePlan = new CapturePlan();
	try {
		return capturePlan.parseScript(query).hookModule(module);
	} catch (e) {
		console.warn('Failed to create CapturePlan', e);
		console.warn(capturePlan.getDiagnostics());
		return null;
	}
}

export function processCaptureResults(
	results: CaptureResult[],
	batchIndex: number = 0
): CaptureMatch[] {
	const processedMatches: (CaptureMatch | null)[] = [];

	for (const result of results) {
		for (const [path, matches] of Object.entries(result.matches)) {
			for (const match of matches) {
				// For now we just ignore array-tensors until we know what to do with them
				if (match.bufferTensor !== undefined && !Array.isArray(match.bufferTensor)) {
					const t = match.bufferTensor;
					// Alas, debugTensor doesn't seem to work in all scenarios, including when doing
					// validation. Seems like we don't copy the buffer to the GPU at the right time.
					// const effectiveTensor = match.type === 'parameter' ? t._clone() : t._clone().debugTensor;
					const effectiveTensor = t._clone();
					processedMatches.push({
						matchId: t.id,
						type: match.type,
						...(match.type !== 'parameter' ? { batchIndex } : {}),
						source: result.source,
						queryIndex: match.queryIndex ?? undefined,
						op: (match as { op?: string }).op,
						parameter: (match as { parameter?: string }).parameter,
						moduleSite: (match as { site?: 'input' | 'output' }).site,
						path: path,
						shape: t.shape,
						tensor: pin(effectiveTensor)
					});
				}
			}
		}
	}

	return processedMatches.filter((m): m is CaptureMatch => m !== null);
}

export function makeCaptureMatchRemote(
	match: RemoteCaptureMatch & { tensor?: unknown; buffer?: unknown }
): RemoteCaptureMatch {
	const { buffer: _1, tensor: _2, ...rest } = match;
	return rest;
}

export async function runValidationExampleForCapture(
	model: GeneratableModel,
	sequences: ValidationExamples,
	collateFn: PistonCollateFnType<Tensor>,
	batchIndex: number
): Promise<void> {
	model.eval();

	let valLoss: number | null = null;
	try {
		let collated;
		if ('toySequences' in sequences) {
			collated = (collateFn as ToyCollateFnType<Tensor>)([sequences.toySequences[batchIndex]]);
		} else {
			collated = (collateFn as NaturalCollateFnType<Tensor>)([
				sequences.naturalSequences[batchIndex]
			]);
		}

		let loss: Tensor | null = null;
		let modelName = '';
		if (model instanceof DecoderTransformer || model instanceof RNNDecoder) {
			const [inputs, targets] = collated.tensors;
			[, loss] = model.forward(await inputs.to('gpu'), {
				targets: await targets.to('gpu')
			});
			modelName = 'decoder-only';
		} else if (model instanceof EncoderDecoderTransformer || model instanceof RNNEncoderDecoder) {
			const [encoderInputs, decoderInputs, decoderTargets] = (
				collated as EncoderDecoderBatchType<Tensor>
			).tensors;
			[, loss] = model.forward(await encoderInputs.to('gpu'), await decoderInputs.to('gpu'), {
				targets: await decoderTargets.to('gpu')
			});
			modelName = 'encoder-decoder';
		} else if (model instanceof EncoderTransformer || model instanceof RNNEncoder) {
			// Encoder-only: compute MLM loss over masked tokens
			const [inputs, labels, attentionMask] = (collated as BidirectionalBatchType<Tensor>).tensors;
			modelName = 'encoder-only';
			if (model instanceof EncoderTransformer) {
				[, , , loss] = model.forward(await inputs.to('gpu'), {
					attentionMask: await attentionMask.to('gpu'),
					targets: await labels.to('gpu')
				});
			} else {
				// No attention mask here
				[, , loss] = model.forward(await inputs.to('gpu'), { targets: await labels.to('gpu') });
			}
		} else {
			throw new Error('Unsupported model for validation');
		}

		if (!loss) {
			throw new Error(`No loss tensor returned from ${modelName} model during validation`);
		}
		valLoss = await (await loss.to('cpu')).item();
		if (valLoss === null) {
			throw new Error(`Validation loss item is null for ${modelName} model`);
		}
		// We do a backward pass to make sure we can capture the gradients
		loss.backward();
	} finally {
		model.train();
	}
}

export type CapturePlanConfig = {
	enabled: boolean;
	script?: string | null;
};

export class CaptureManager {
	private _plan: CapturePlan | null = null;

	build(model: Module, config: CapturePlanConfig): void {
		this._plan = null;
		if (!config.enabled) return;
		if (!config.script) return;
		this._plan = createCapturePlan(config.script, model);
	}

	createSession(): CaptureSession | null {
		return this._plan ? this._plan.createSession() : null;
	}

	finalize(session: CaptureSession, batchIndex: number = 0): CaptureMatch[] {
		const results = session.finalize();
		return processCaptureResults(results, batchIndex);
	}

	get queries(): unknown[] {
		return this._plan?.queries ?? [];
	}

	get plan(): CapturePlan | null {
		return this._plan;
	}
}
