import type { Attachment } from 'svelte/attachments';

import * as echarts from 'echarts/core';

type EChartsAttachmentParams = {
	opts: () => echarts.EChartsCoreOption | null;
	// Optional setup hook to programmatically configure the instance (e.g., event bridging)
	// May return a cleanup function that will be called on detach.
	setup?: (chart: echarts.ECharts, getPeers: undefined) => void | (() => void);
};

export function setupAxisSync(chart: echarts.ECharts, getPeers: () => echarts.ECharts[]) {
	let isRelaying = false;
	chart.on('updateAxisPointer', (evt) => {
		if (isRelaying) return;
		const e = evt as { axesInfo?: Array<{ axisDim: string; value: number }> };
		const axesInfo = e?.axesInfo;
		if (!axesInfo || axesInfo.length === 0) return;
		const xInfo = axesInfo.find((a) => a.axisDim === 'x');
		if (!xInfo) return;

		const catIndex = Math.max(0, Math.floor(xInfo.value ?? 0));
		const opt = chart.getOption?.() as
			| { xAxis?: Array<{ data?: (number | string)[] }> }
			| undefined;
		const cats = opt?.xAxis?.[0]?.data ?? [];
		const step = Number(cats[catIndex] ?? catIndex);

		isRelaying = true;
		for (const peer of getPeers()) {
			const hit = findPeerDataIndexByStep(peer, step);
			peer.dispatchAction({
				type: 'updateAxisPointer',
				seriesIndex: hit?.seriesIndex ?? 0,
				dataIndex: hit?.dataIndex ?? catIndex
			});
		}
		Promise.resolve().then(() => {
			isRelaying = false;
		});
	});
}

// Binary-search utilities for sorted numeric step arrays
export function exactIndex(steps: number[], target: number): number {
	let lo = 0,
		hi = steps.length - 1;
	while (lo <= hi) {
		const mid = (lo + hi) >> 1;
		const v = steps[mid];
		if (v === target) return mid;
		if (v < target) lo = mid + 1;
		else hi = mid - 1;
	}
	return -1;
}

export function nearestIndex(steps: number[], target: number): number {
	if (steps.length === 0) return -1;
	const first = steps[0],
		last = steps[steps.length - 1];
	if (target < first || target > last) return -1;
	let lo = 0,
		hi = steps.length;
	while (lo < hi) {
		const mid = (lo + hi) >> 1;
		if (steps[mid] < target) lo = mid + 1;
		else hi = mid;
	}
	if (lo === 0) return 0;
	if (lo === steps.length) return steps.length - 1;
	return target - steps[lo - 1] <= steps[lo] - target ? lo - 1 : lo;
}

// Extract numeric x/step from an ECharts updateAxisPointer event
export function extractStepFromAxisPointerEvent(evt: unknown): number | null {
	const e = evt as { axesInfo?: Array<{ axisDim: string; value: number }> } | undefined;
	const xInfo = e?.axesInfo?.find((a) => a.axisDim === 'x');
	const step = Number(xInfo?.value);
	return Number.isFinite(step) ? step : null;
}

function extractXValue(datum: unknown): number | null {
	if (Array.isArray(datum)) {
		const x = Number(datum[0]);
		return Number.isFinite(x) ? x : null;
	}
	if (datum && typeof datum === 'object') {
		const obj = datum as { value?: unknown; x?: unknown; step?: unknown };
		if (Array.isArray(obj.value)) {
			const x = Number(obj.value[0]);
			return Number.isFinite(x) ? x : null;
		}
		const xCandidate = obj.x ?? obj.step;
		if (typeof xCandidate === 'number') return xCandidate;
		if (typeof xCandidate === 'string') {
			const parsed = Number(xCandidate);
			return Number.isFinite(parsed) ? parsed : null;
		}
	}
	return null;
}

function linearSearchByX(data: unknown[], targetStep: number): number {
	for (let i = 0; i < data.length; i++) {
		const x = extractXValue(data[i]);
		if (x === targetStep) return i;
	}
	return -1;
}

function binarySearchByX(data: unknown[], targetStep: number): number {
	if (data.length === 0) return -1;
	const first = extractXValue(data[0]);
	const last = extractXValue(data[data.length - 1]);
	if (first === null || last === null) return linearSearchByX(data, targetStep);

	const ascending = last >= first;
	let lo = 0;
	let hi = data.length - 1;
	while (lo <= hi) {
		const mid = lo + ((hi - lo) >> 1);
		const x = extractXValue(data[mid]);
		if (x === null) return linearSearchByX(data, targetStep);
		if (x === targetStep) return mid;
		if (ascending ? x < targetStep : x > targetStep) lo = mid + 1;
		else hi = mid - 1;
	}
	return -1;
}

function getSeriesArrayFromOption(opt: unknown): unknown[] {
	const o = opt as { series?: unknown[] } | undefined;
	return Array.isArray(o?.series) ? (o!.series as unknown[]) : [];
}

function getDataArrayFromSeries(seriesItem: unknown): unknown[] {
	const s = seriesItem as { data?: unknown[] } | undefined;
	return Array.isArray(s?.data) ? (s!.data as unknown[]) : [];
}

export function findPeerDataIndexByStep(
	peer: echarts.ECharts,
	step: number
): { seriesIndex: number; dataIndex: number } | null {
	const opt = peer.getOption() as unknown;
	const seriesArr = getSeriesArrayFromOption(opt);
	if (seriesArr.length === 0) return null;
	const seriesIndex = seriesArr.length - 1; // highest series index
	const data = getDataArrayFromSeries(seriesArr[seriesIndex]);
	if (data.length === 0) return null;
	const dataIndex = binarySearchByX(data, step);
	if (dataIndex < 0) return null;
	return { seriesIndex, dataIndex };
}

export default function createEChartsAttachment(params: EChartsAttachmentParams): Attachment {
	return (node: Element) => {
		if (!(node instanceof HTMLDivElement)) {
			throw new Error('ECharts attachment requires a div element');
		}

		const chart = echarts.init(node);
		const getPeers = undefined;

		// Allow caller to set up custom behavior (e.g., axis-pointer mapping)
		let setupCleanup: (() => void) | undefined;
		const maybeCleanup = params.setup?.(chart, getPeers);
		if (typeof maybeCleanup === 'function') setupCleanup = maybeCleanup;

		const resizeObserver = new ResizeObserver(() => {
			chart.resize();
		});
		resizeObserver.observe(node);

		$effect(() => {
			const options = params.opts?.();
			if (options) {
				chart.setOption(options, {
					notMerge: false,
					replaceMerge: ['series'],
					lazyUpdate: false
				});
			}
		});

		return () => {
			resizeObserver.unobserve(node);
			setupCleanup?.();
			chart.dispose();
		};
	};
}

type MoveDetail = {
	sourceId: string;
	runId: string;
	step: number;
};

type ClearDetail = {
	sourceId: string;
};

const MOVE_EVENT = 'run-pointer:move';
const CLEAR_EVENT = 'run-pointer:clear';

const bus: EventTarget = new EventTarget();

export function publishMove(detail: MoveDetail): void {
	bus.dispatchEvent(new CustomEvent<MoveDetail>(MOVE_EVENT, { detail }));
}

export function publishClear(detail: ClearDetail): void {
	bus.dispatchEvent(new CustomEvent<ClearDetail>(CLEAR_EVENT, { detail }));
}

export function subscribe(
	onMove?: (detail: MoveDetail) => void,
	onClear?: (detail: ClearDetail) => void
): () => void {
	const moveListener = (e: Event) => {
		const ce = e as CustomEvent<MoveDetail>;
		if (onMove) onMove(ce.detail);
	};
	const clearListener = (e: Event) => {
		const ce = e as CustomEvent<ClearDetail>;
		if (onClear) onClear(ce.detail);
	};
	if (onMove) bus.addEventListener(MOVE_EVENT, moveListener as EventListener);
	if (onClear) bus.addEventListener(CLEAR_EVENT, clearListener as EventListener);
	return () => {
		if (onMove) bus.removeEventListener(MOVE_EVENT, moveListener as EventListener);
		if (onClear) bus.removeEventListener(CLEAR_EVENT, clearListener as EventListener);
	};
}
