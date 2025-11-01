import type { CaptureStep } from '$lib/train/capture';
import type { ValidationStep } from '$lib/train/validation';

import { generateMemorableName } from '$lib/workspace/utils';
import { SvelteMap, SvelteSet } from 'svelte/reactivity';

import { type Config } from './config';

export type BaseStepData = { step: number };

export type Point = BaseStepData & { y: number };

export type TokenRollout = {
	tokenIds: number[];
	probs: number[][];
};
export type StepData = Point | ValidationStep;

export type MetricData = {
	metricName: string;
	data: StepData[];
};

export type RunData = {
	runId: string;
	color: string;
	config: Config;
	metrics: SvelteMap<string, MetricData>;
	step: number;
	lastUpdated: number;
	createdAt: number;
};

export type RunMeta = {
	runId: string | null;
	config: Config | null;
};

// A palette of distinct colors for runs
const RUN_COLORS = ['#f5493b', '#dfa300', '#3bbc4a', '#00b7c0', '#4475f6', '#cc5cc5'];

export const runsMap = new SvelteMap<string, RunData>();
export const runCounter = $state({ current: 0 });
export const currentRun = $state<{ current: RunMeta | null }>({
	current: null
});

export function getRuns(): ReadonlyArray<RunData> {
	return [...runsMap.values()].sort((a, b) => a.runId.localeCompare(b.runId));
}

export function getAllMetricNames(): ReadonlyArray<string> {
	const names = new SvelteSet<string>();
	for (const run of runsMap.values()) {
		for (const metricName of run.metrics.keys()) {
			names.add(metricName);
		}
	}
	return [...names].sort();
}

/**
 * Gets all metric names from the last n runs.
 * @param n - The number of recent runs to consider
 * @returns Array of unique metric names sorted alphabetically
 */
export function getMetricNamesFromLastNRuns(n: number): ReadonlyArray<string> {
	const names = new SvelteSet<string>();
	const recentRuns = getLastNRuns(n);
	for (const run of recentRuns) {
		for (const metricName of run.metrics.keys()) {
			names.add(metricName);
		}
	}
	return [...names].sort();
}

export function getCurrentRun(): RunMeta | null {
	return currentRun.current;
}

export function getLatestRun(): RunMeta | null {
	if (currentRun.current !== null) {
		return currentRun.current;
	}

	const allRuns = [...runsMap.values()];
	allRuns.sort((a, b) => b.lastUpdated - a.lastUpdated);
	return allRuns[0] ?? null;
}

/**
 * Gets the last n runs sorted by most recently updated, ensuring the current run is always
 * included.
 */
export function getLastNRuns(n: number): ReadonlyArray<RunData> {
	const allRuns = [...runsMap.values()];

	// Sort by lastUpdated descending (most recent first)
	allRuns.sort((a, b) => b.lastUpdated - a.lastUpdated);

	// If we have fewer runs than requested, return all of them
	if (allRuns.length <= n) {
		return allRuns;
	}

	return allRuns.slice(0, n);
}

export function newRun(config: Config, id?: string): RunData {
	if (id && runsMap.has(id)) {
		throw new Error(`Run with id ${id} already exists`);
	}

	const runId = id ?? generateMemorableName(runCounter.current);
	const color = RUN_COLORS[runCounter.current % RUN_COLORS.length];
	const now = Date.now();
	// Find baseline as immediately-previous-by-creation
	const existingRuns = [...runsMap.values()];
	existingRuns.sort((a, b) => (a.createdAt ?? a.lastUpdated) - (b.createdAt ?? b.lastUpdated));

	const run = {
		runId: runId,
		config,
		color: color,
		metrics: new SvelteMap<string, MetricData>(),
		step: 0,
		lastUpdated: now,
		createdAt: now
	};
	runsMap.set(runId, run);
	runCounter.current += 1;
	currentRun.current = { runId: runId, config: config };
	return run;
}

export function endRun() {
	currentRun.current = null;
}

/**
 * Logs metric data for a specific step in a run.
 */
export function log(
	runId: string,
	data: { [metricName: string]: Omit<StepData, 'step'> },
	{ step }: { step?: number } = {}
): void {
	const run = runsMap.get(runId);
	const determinedStep = step !== undefined ? step : (runsMap.get(runId)?.step ?? 0);

	// Create run if it doesn't exist
	if (!run) {
		throw new Error(`Run with id ${runId} does not exist`);
	}

	const currentStep = determinedStep;

	// Update metrics for the specified step
	for (const [metricName, value] of Object.entries(data)) {
		let metric = run.metrics.get(metricName);
		if (!metric) {
			metric = {
				metricName,
				data: []
			};
			run.metrics.set(metricName, metric);
		}

		let stepData: StepData;
		if (typeof value === 'number') {
			stepData = { step: currentStep, y: value };
		} else if ('matches' in value) {
			stepData = { step: currentStep, ...value } as CaptureStep;
		} else {
			stepData = { step: currentStep, ...value } as ValidationStep;
		}

		const updatedMetric = {
			...metric,
			data: [...metric.data, stepData].sort((a: StepData, b: StepData) => a.step - b.step)
		};

		run.metrics.set(metricName, updatedMetric);
	}

	// Update step counter
	if (step === undefined) {
		run.step += 1;
	} else if (step > run.step) {
		run.step = step;
	}

	run.lastUpdated = Date.now();
	runsMap.set(runId, run);
}

export type LogFn = typeof log;

export function resetWorkspace(): void {
	runsMap.clear();
	runCounter.current = 0;
	currentRun.current = null;
	console.debug('[workspaceState] Reset.');
}

// Group metrics by prefix (everything before the first '/')
export function getMetricGroups(metricNames: ReadonlyArray<string>): Record<string, string[]> {
	const groups: Record<string, string[]> = {};

	const allMetricNames = metricNames;

	allMetricNames.forEach((metricName) => {
		// For now we're just special-casing this, but we might want to bring it into the metrics view
		// anyway
		if (metricName === 'visualization/matches') {
			return;
		}

		const parts = metricName.split('/');
		const groupName = parts.length > 1 ? parts[0] : 'default';

		if (!groups[groupName]) {
			groups[groupName] = [];
		}
		groups[groupName].push(metricName);
	});

	return groups;
}
