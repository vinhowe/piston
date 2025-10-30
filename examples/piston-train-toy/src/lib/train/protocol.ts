import type { Config } from '$lib/workspace/config';
import type { StepData } from '$lib/workspace/runs.svelte';

type WithRunId = { runId: string };

export type WorkerCommand =
	| {
			type: 'start';
			data: { runId: string; config: Config };
	  }
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
	| { type: 'complete' }
	| LogWorkerEvent
	| ErrorWorkerEvent;

export type RunWorkerEvent = RunWorkerEventWithoutRunId & WithRunId;

export type WorkerEvent = ReadyWorkerEvent | LogWorkerEvent | ErrorWorkerEvent | RunWorkerEvent;
