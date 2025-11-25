import type { Config } from '$lib/workspace/config';
import type { StepData } from '$lib/workspace/runs.svelte';
import type { IndexState } from '@piston-ml/piston-web';

type WithRunId = { runId: string };

/** Category of a profile event */
export type ProfileCategory =
	| 'scope'
	| 'kernel'
	| 'allocation'
	| 'deallocation'
	| 'tensor_allocation'
	| 'tensor_deallocation';

/** A single profile event from the Rust profiler */
export interface ProfileEventData {
	/** Name of the operation or scope */
	name: string;
	/** Category of the event */
	category: ProfileCategory;
	/** Start timestamp in microseconds since profiling started */
	start_us: number;
	/** Duration in microseconds (0 for instant events) */
	duration_us: number;
	/** Additional metadata (shape, dtype, buffer size, workgroups, etc.) */
	metadata?: Record<string, string>;
	/** Scope stack at the time of the event */
	stack?: string[];
}

export type WorkerCommand =
	| {
			type: 'start';
			data: { runId: string; config: Config; resumeFrom?: Uint8Array<ArrayBufferLike> };
	  }
	| {
			type: 'checkpoint.peekConfig';
			data: { requestId: string; buffer: Uint8Array<ArrayBufferLike> };
	  }
	| { type: 'pause' }
	| { type: 'resume' }
	| { type: 'step' }
	| { type: 'save' }
	| { type: 'stop' }
	| {
			type: 'visualizer.updateScript';
			data: { example: string; script: string | null };
	  }
	| {
			type: 'visualizer.canvas';
			data: { canvas: OffscreenCanvas; labelPaddingCssPx?: number };
	  }
	| { type: 'visualizer.resize'; data: { width: number } }
	| { type: 'visualizer.setTarget'; data: { target: 'train' | 'validation' } }
	| {
			type: 'visualizer.setSelectedValidation';
			data: { exampleIndex: number; tokenIndex: number };
	  }
	| { type: 'inspectModel'; data: { requestId: string; config: Config } };

type ReadyWorkerEvent = {
	type: 'ready';
};

type LogWorkerEvent = {
	type: 'log';
	level: 'debug' | 'info' | 'warn' | 'error';
	message: string;
	source?: string;
	lineno?: number;
	colno?: number;
};

type ErrorWorkerEvent = {
	type: 'error';
	name?: string;
	message: string;
	stack?: string;
};

export type RunWorkerEventWithoutRunId =
	| {
			type: 'metrics';
			data: { [metricName: string]: Omit<StepData, 'step'> };
			metadata?: { step?: number };
	  }
	| {
			type: 'capture';
			step: number;
			boxes: unknown[];
			statsById: Record<string, unknown>;
			width: number;
			height: number;
			queries: unknown[];
	  }
	| {
			type: 'profiling';
			step: number;
			events: ProfileEventData[];
	  }
	| { type: 'checkpoint'; buffer: Uint8Array<ArrayBufferLike> }
	| { type: 'restart'; buffer: Uint8Array<ArrayBufferLike> }
	| { type: 'paused' }
	| { type: 'resumed' }
	| { type: 'complete' }
	| LogWorkerEvent
	| ErrorWorkerEvent;

export type RunWorkerEvent = RunWorkerEventWithoutRunId & WithRunId;

export type WorkerEvent =
	| ReadyWorkerEvent
	| LogWorkerEvent
	| ErrorWorkerEvent
	| RunWorkerEvent
	| { type: 'checkpoint.config'; requestId: string; config: Config }
	| { type: 'visualizer.ready' }
	| { type: 'visualizer.error'; message: string }
	| {
			type: 'modelInspection';
			requestId: string;
			parameterCount: number;
			vocabSize: number;
			modelIndex: IndexState;
	  }
	| { type: 'modelInspectionError'; requestId: string; message: string };
