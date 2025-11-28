import type { Config } from '$lib/workspace/config';
import type { StepData } from '$lib/workspace/runs.svelte';
import type { IndexState } from '@piston-ml/piston-web';

type WithRunId = { runId: string };

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
	| {
			type: 'modelInspection';
			requestId: string;
			parameterCount: number;
			vocabSize: number;
			modelIndex: IndexState;
	  }
	| { type: 'modelInspectionError'; requestId: string; message: string };
