import type { Config } from '$lib/workspace/config';

import * as piston from '@piston-ml/piston-web';
import {
	type ConstantConfig,
	type CosineAnnealingConfig,
	type ExponentialConfig,
	GradScaler,
	type GradScalerStateDict,
	type LinearConfig,
	LRScheduler,
	Optimizer,
	type OptimizerParamState,
	type ParamGroupConfig,
	type SchedulerStateDict,
	type StateDict,
	type StepConfig,
	Tensor
} from '@piston-ml/piston-web';

/**
 * Recursively walks an object and extracts any Tensor values into `out` as Buffers.
 * Replaces extracted tensors in the returned structure with a small marker object
 * containing the tensor storage key that was used in `out`.
 */
export function splitTensorsFromObject(
	value: unknown,
	baseKey: string,
	out: Record<string, piston.Parameter | piston.Buffer>
): unknown {
	if (value instanceof Tensor) {
		out[baseKey] = new piston.Buffer(value, true);
		return { __tensor__: baseKey };
	}
	if (Array.isArray(value)) {
		return value.map((v, i) => splitTensorsFromObject(v, `${baseKey}.${i}`, out));
	}
	if (value && typeof value === 'object') {
		const result: Record<string, unknown> = {};
		for (const [k, v] of Object.entries(value)) {
			result[k] = splitTensorsFromObject(v, `${baseKey}.${k}`, out);
		}
		return result;
	}
	return value;
}

export type AnySchedulerState = SchedulerStateDict<
	StepConfig | CosineAnnealingConfig | ExponentialConfig | LinearConfig | ConstantConfig | unknown
>;

export type CheckpointOptimizerExtra = {
	name: string;
	// JSON with __tensor__ markers
	state: unknown;
	paramGroups: ParamGroupConfig[];
} | null;

export interface CheckpointDataState {
	blockSize?: number | { source: number; target: number };
	toy?: { cursor: number; baseSeed?: number; datasetName?: string } | null;
	natural?: { shardIndex: number; cursor: number } | null;
}

export interface CheckpointExtra {
	config: Config;
	optimizer: CheckpointOptimizerExtra;
	numSteps: number;
	lrScheduler?: { state: AnySchedulerState };
	dataState?: CheckpointDataState;
	// Optional wall-clock training start time in ms to persist across restarts
	startTimeMs?: number;
	// GradScaler state for AMP
	gradScaler?: GradScalerStateDict;
}

/**
 * Builds a checkpoint payload by combining model parameters with optimizer state.
 * - Model parameters/buffers go into `tensors` directly
 * - Any Tensor values found inside optimizer.stateDict().state are lifted into `tensors`
 *   under keys prefixed with `optimizer/state/...`
 * - Extra contains { config, optimizer, numSteps }
 */

export function buildCheckpoint(
	model: piston.Module,
	optimizer: Optimizer,
	numSteps: number,
	configForExtra: Config,
	scheduler?: LRScheduler<unknown>,
	dataState?: CheckpointDataState,
	startTimeMs?: number,
	gradScaler?: GradScaler
): { tensors: Record<string, piston.Parameter | piston.Buffer>; extra: CheckpointExtra } {
	const tensors: Record<string, piston.Parameter | piston.Buffer> = model.stateDict();

	let optimizerExtra: CheckpointOptimizerExtra = null;
	try {
		const name = optimizer.constructor.name ?? 'Optimizer';
		const packed = optimizer.stateDict();
		const tensorSlots: Record<string, piston.Buffer> = {};
		const jsonState = splitTensorsFromObject(packed.state, 'optimizer.state', tensorSlots);
		Object.assign(tensors, tensorSlots);
		optimizerExtra = {
			name,
			state: jsonState,
			paramGroups: packed.paramGroups
		};
	} catch (e) {
		console.warn('Failed to pack optimizer stateDict for checkpoint extra:', e);
	}

	const extra: CheckpointExtra = {
		config: configForExtra,
		optimizer: optimizerExtra,
		numSteps,
		lrScheduler: scheduler ? { state: scheduler.stateDict() } : undefined,
		dataState,
		startTimeMs,
		gradScaler: gradScaler ? gradScaler.stateDict() : undefined
	};

	return { tensors, extra };
}

/**
 * Replace any marker objects of the form { __tensor__: key } inside a JSON structure
 * with actual Tensors from the provided mapping.
 */
export function rehydrateTensorsInObject<T>(value: unknown, lifted: Record<string, Tensor>): T {
	if (value && typeof value === 'object' && !Array.isArray(value)) {
		const marker = value as { __tensor__?: string };
		if (typeof marker.__tensor__ === 'string') {
			const key = marker.__tensor__;
			if (!(key in lifted)) {
				throw new Error(`Missing lifted tensor for key '${key}' during optimizer rehydration`);
			}
			return lifted[key] as unknown as T;
		}
		const out: Record<string, unknown> = {};
		for (const [k, v] of Object.entries(value)) {
			out[k] = rehydrateTensorsInObject(v, lifted);
		}
		return out as unknown as T;
	}
	if (Array.isArray(value)) {
		return value.map((v) => rehydrateTensorsInObject(v, lifted)) as unknown as T;
	}
	return value as T;
}

export interface SplitLoadedStateResult {
	modelState: Record<string, Tensor>;
	schedulerState?: AnySchedulerState;
	optimizerState: StateDict<OptimizerParamState>;
	numSteps: number;
	config: Config;
	dataState?: CheckpointDataState;
	startTimeMs?: number;
	gradScalerState?: GradScalerStateDict;
}

/**
 * Given loaded state from piston.load, split out model state from lifted optimizer tensors
 * and rehydrate optimizer and scheduler states from extras.
 */
export function splitLoadedState(loaded: {
	state: Record<string, Tensor>;
	extra?: CheckpointExtra;
}): SplitLoadedStateResult {
	const prefix = 'optimizer.state';
	const liftedOptimizerTensors: Record<string, Tensor> = {};
	const modelState: Record<string, Tensor> = {};

	for (const [key, t] of Object.entries(loaded.state)) {
		if (key.startsWith(prefix)) {
			liftedOptimizerTensors[key] = t;
		} else {
			modelState[key] = t;
		}
	}

	let optimizerState: StateDict<OptimizerParamState> | undefined;
	let schedulerState: AnySchedulerState | undefined = undefined;
	let numSteps = 0;
	let config: Config | null = null;
	let dataState: CheckpointDataState | undefined = undefined;
	let startTimeMs: number | undefined = undefined;
	let gradScalerState: GradScalerStateDict | undefined = undefined;

	const { extra } = loaded;

	if (extra) {
		config = extra.config;
		numSteps = extra.numSteps;
		if (extra.optimizer) {
			const rehydratedState = rehydrateTensorsInObject<Record<number, OptimizerParamState>>(
				extra.optimizer.state,
				liftedOptimizerTensors
			);
			optimizerState = {
				state: rehydratedState,
				paramGroups: extra.optimizer.paramGroups ?? []
			};
		}
		if (extra.lrScheduler && extra.lrScheduler.state) {
			schedulerState = extra.lrScheduler.state;
		}
		if (extra.dataState) {
			dataState = extra.dataState;
		}
		if (typeof extra.startTimeMs === 'number') {
			startTimeMs = extra.startTimeMs;
		}
		if (extra.gradScaler) {
			gradScalerState = extra.gradScaler;
		}
	}

	if (!config) {
		throw new Error('No config found in checkpoint');
	}

	if (numSteps == null) {
		throw new Error('No numSteps found in checkpoint');
	}

	if (!optimizerState) {
		throw new Error('No optimizer state found in checkpoint');
	}

	// Some runs don't use a scheduler, so we don't validate that it's present

	return {
		modelState,
		optimizerState,
		schedulerState,
		numSteps,
		config,
		dataState,
		startTimeMs,
		gradScalerState
	};
}
