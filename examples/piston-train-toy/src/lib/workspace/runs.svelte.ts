import type { CaptureStep } from '$lib/train/capture';
import type { ValidationStep } from '$lib/train/validation';

import { generateMemorableName } from '$lib/workspace/utils';
import { SvelteMap, SvelteSet } from 'svelte/reactivity';

import { type Config, CONFIG_DESCRIPTIONS } from './config';

export type BaseStepData = { step: number };

export type Point = BaseStepData & { y: number };

export type TokenRollout = {
	tokenIds: number[];
	probs: number[][];
};

export type VisualizationStep = BaseStepData & {
	type: 'visualization';
};

export type StepData = Point | ValidationStep | CaptureStep;

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
	diffSummary: string;
};

export type RunMeta = {
	runId: string | null;
	config: Config | null;
};

// A palette of distinct colors for runs
const RUN_COLORS = ['#f5493b', '#dfa300', '#3bbc4a', '#00b7c0', '#4475f6', '#cc5cc5'];

export const PREFIX_BOOSTS: Record<string, number> = { data: 10 };

type DiffItem = {
	label: string;
	path: string[];
	display: string;
	boost: number;
};

function pathToString(path: ReadonlyArray<string>): string {
	return path.join('.');
}

function getAtPath(obj: unknown, path: ReadonlyArray<string>): unknown {
	let cur = obj;
	for (const key of path) {
		if (cur == null || typeof cur !== 'object') return undefined;
		cur = (cur as Record<string, unknown>)[key];
	}
	return cur;
}

function getDescriptor(path: ReadonlyArray<string>): { label: string | null; itemBoost?: number } {
	let cur = CONFIG_DESCRIPTIONS as unknown;

	for (const key of path) {
		if (cur == null || typeof cur !== 'object') return { label: null };
		cur = (cur as Record<string, unknown>)[key];
	}
	if (cur == null) return { label: null };
	if (Array.isArray(cur)) {
		const [shortName, boost] = cur;
		return { label: shortName, itemBoost: typeof boost === 'number' ? boost : undefined };
	}
	if (typeof cur === 'string') return { label: cur };
	if (typeof cur === 'object' && 'shortName' in cur && typeof cur.shortName === 'string') {
		return { label: cur.shortName };
	}
	return { label: null };
}

function maxPrefixBoost(path: ReadonlyArray<string>, boosts: Record<string, number>): number {
	let maxB = 0;
	for (let i = 1; i <= path.length; i++) {
		const pref = path.slice(0, i).join('.');
		const b = boosts[pref];
		if (typeof b === 'number' && b > maxB) maxB = b;
	}
	return maxB;
}

function orderOfMagnitude(n: number): number {
	if (n === 0) return 0;
	return Math.floor(Math.log10(Math.abs(n)));
}

function formatScientific(n: number, significantDigits = 1): string {
	if (n === 0) return '0';
	const s = n.toExponential(Math.max(0, significantDigits - 1));
	const [mant, expRaw] = s.split('e');
	const mantTrim = mant
		.replace(/\.0+$/, '')
		.replace(/(\.[0-9]*?)0+$/, '$1')
		.replace(/\.$/, '');
	const exp = String(parseInt(expRaw, 10));
	return `${mantTrim}e${exp}`;
}

function formatNumberDiff(a: number, b: number): string {
	const bothSmallInts =
		Number.isInteger(a) && Number.isInteger(b) && Math.abs(a) < 100 && Math.abs(b) < 100;
	if (bothSmallInts) return `${a}→${b}`;

	// If signs differ or either is non-integer, prefer concise scientific form
	const expA = orderOfMagnitude(a);
	const expB = orderOfMagnitude(b);
	const bothInts = Number.isInteger(a) && Number.isInteger(b);
	const base = Math.pow(10, expA);
	if (
		bothInts &&
		Math.sign(a) >= 0 &&
		Math.sign(b) >= 0 &&
		expA === expB &&
		// Only use suffix form for small changes
		Math.abs(b - a) < base
	) {
		const lead = Math.floor(a / base);
		const suffixA = a - lead * base;
		const suffixB = b - lead * base;
		return `${lead}e${expA}+(${suffixA}→${suffixB})`;
	}
	const sa = formatScientific(a, 1);
	const sb = formatScientific(b, 1);
	return `${sa}⥵${sb}`;
}

function formatValue(v: unknown): string {
	if (typeof v === 'number') return formatScientific(v, 1);
	if (typeof v === 'string') return v;
	if (typeof v === 'boolean') return v ? 'true' : 'false';
	return String(v);
}

function comparePrimitive(a: unknown, b: unknown): boolean {
	// Strict equality suffices for primitives we expect here
	return a === b;
}

function collectLeafPaths(obj: unknown, prefix: string[] = []): string[][] {
	const out: string[][] = [];
	if (obj == null || typeof obj !== 'object') return out;
	for (const key of Object.keys(obj)) {
		const nextPath = [...prefix, key];
		const val = (obj as Record<string, unknown>)[key];
		if (val != null && typeof val === 'object') {
			// Only traverse objects that are not arrays
			if (!Array.isArray(val)) out.push(...collectLeafPaths(val, nextPath));
		} else {
			out.push(nextPath);
		}
	}
	return out;
}

export function describeConfigDiff(
	prev: Config | null,
	curr: Config,
	opts?: { topK?: number; prefixBoosts?: Record<string, number> }
): string {
	const topK = opts?.topK ?? 3;
	const boosts = opts?.prefixBoosts ?? PREFIX_BOOSTS;
	if (!prev) return 'initial experiment';

	const prevPaths = collectLeafPaths(prev);
	const currPaths = collectLeafPaths(curr);
	const allKey = new SvelteSet<string>();
	for (const p of prevPaths) allKey.add(pathToString(p));
	for (const p of currPaths) allKey.add(pathToString(p));

	const diffs: DiffItem[] = [];
	for (const key of allKey) {
		const path = key.split('.');
		// if (shouldSkipPath(path)) continue;
		const { label, itemBoost } = getDescriptor(path);
		if (!label) continue;
		const a = getAtPath(prev, path);
		const b = getAtPath(curr, path);
		if (comparePrimitive(a, b)) continue;

		const prefixB = maxPrefixBoost(path, boosts);
		let effBoost = prefixB;
		if (typeof itemBoost === 'number') {
			if (itemBoost < prefixB) {
				console.warn(
					`item boost lower than parent prefix; path=${key} itemBoost=${itemBoost} parentMax=${prefixB}`
				);
			}
			effBoost = Math.max(effBoost, itemBoost);
		}

		let display: string;
		if (typeof a === 'boolean' && typeof b === 'boolean') {
			display = b ? `+${label}` : `-${label}`;
		} else if (typeof a === 'number' && typeof b === 'number') {
			display = `${label}:${formatNumberDiff(a, b)}`;
		} else {
			display = `${label}:${formatValue(a)}→${formatValue(b)}`;
		}

		diffs.push({ label, path, display, boost: effBoost });
	}

	diffs.sort((x, y) => {
		if (y.boost !== x.boost) return y.boost - x.boost;
		return x.label.localeCompare(y.label);
	});

	if (diffs.length === 0) return 'no changes';
	const top = diffs.slice(0, topK).map((d) => d.display);
	const rest = diffs.length - top.length;
	return rest > 0 ? `${top.join(', ')}, etc ${rest} more` : top.join(', ');
}

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

export function clearPastRuns(): void {
	const keepRunId = currentRun.current;
	if (keepRunId === null) {
		// No current run tracked; clear everything
		runsMap.clear();
		return;
	}
	for (const runId of [...runsMap.keys()]) {
		if (runId !== keepRunId.runId) {
			runsMap.delete(runId);
		}
	}
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
	const prevRun = existingRuns.length > 0 ? existingRuns[existingRuns.length - 1] : undefined;
	const diffSummary = describeConfigDiff(prevRun?.config ?? null, config, {
		topK: 3,
		prefixBoosts: PREFIX_BOOSTS
	});

	const run = {
		runId: runId,
		config,
		color: color,
		metrics: new SvelteMap<string, MetricData>(),
		step: 0,
		lastUpdated: now,
		createdAt: now,
		diffSummary
	};
	runsMap.set(runId, run);
	runCounter.current += 1;
	currentRun.current = { runId: runId, config: config };
	return run;
}

export function endRun() {
	currentRun.current = null;
}

export function restoreRun(run: RunData): RunData {
	runsMap.set(run.runId, run);
	currentRun.current = { runId: run.runId, config: run.config };
	return run;
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
