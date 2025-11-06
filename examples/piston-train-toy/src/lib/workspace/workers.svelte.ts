import type { MatchBox } from '$lib/train/visualizer';
import type { Config } from '$lib/workspace/config';
import type { IndexState, TensorQuery } from '@piston-ml/piston-web';

import { SvelteMap } from 'svelte/reactivity';

import { config } from './config.svelte';
import { lastSessionStore } from './lastSessionStore';
import { currentRun, log, runsMap } from './runs.svelte';
import {
	gpuPowerPreference,
	setGpuName,
	triggerLowDiversityDatasetError,
	triggerVramLimitFlash
} from './ui.svelte';

// Train state
let trainWorker: Worker | null = $state(null);
export const workerReady = $state({ current: false });
export const workerVersion = $state({ current: 0 });
export const trainingState = $state<{ current: 'training' | 'paused' | 'stopped' }>({
	current: 'stopped'
});

// Visualizer layout state
let visualizerBoxes = $state<MatchBox[] | null>(null);
let visualizerQueries = $state<TensorQuery[] | null>(null);
let visualizerLayoutStep = $state<number | null>(null);
let visualizerLayoutRunId = $state<string | null>(null);
let visualizerRenderWidth = $state<number | null>(null);
let visualizerRenderHeight = $state<number | null>(null);

// UA memory measurement state (main thread only)
let uaMemoryInterval: ReturnType<typeof setInterval> | null = null;
let lastUAMemoryBytes: number | null = null;

let screenWakeLock: WakeLockSentinel | null = null;

type CheckpointPayload = { runId: string; buffer: Uint8Array<ArrayBufferLike> };
const pendingCheckpointWaiters: Array<(p: CheckpointPayload) => void> = [];
const pendingPeekResolvers = new SvelteMap<string, (cfg: Config) => void>();

async function acquireScreenWakeLock() {
	// Only attempt in browser/secure contexts that support it
	if (typeof navigator === 'undefined' || !('wakeLock' in navigator)) return;
	try {
		// Request a screen wake lock
		screenWakeLock = await navigator.wakeLock.request('screen');
		// Ensure our local reference is cleared if the system revokes the lock
		screenWakeLock?.addEventListener?.('release', () => {
			screenWakeLock = null;
		});
	} catch (err) {
		console.warn('Screen Wake Lock request failed:', err);
	}
}

async function releaseScreenWakeLock() {
	if (!screenWakeLock) return;
	try {
		await screenWakeLock.release();
	} catch (err) {
		console.warn('Screen Wake Lock release failed:', err);
	} finally {
		screenWakeLock = null;
	}
}

export async function initializeWorker() {
	return new Promise<void>((resolve, reject) => {
		try {
			// Create the dedicated module worker
			// eslint-disable-next-line svelte/prefer-svelte-reactivity
			trainWorker = new Worker(new URL('$lib/train/moduleWorker.ts', import.meta.url), {
				type: 'module',
				name: 'moduleWorker'
			});

			console.log('[Main] Module worker created successfully.');

			// Set up UA memory measurement (immediate + interval) on main thread (only once)
			if (!uaMemoryInterval) {
				const measure = (
					performance as Performance & {
						measureUserAgentSpecificMemory?: () => Promise<{ bytes: number }>;
					}
				).measureUserAgentSpecificMemory;
				if (typeof measure === 'function') {
					const measureAndStore = async () => {
						try {
							const { bytes } = await (
								performance as Performance & {
									measureUserAgentSpecificMemory?: () => Promise<{ bytes: number }>;
								}
							).measureUserAgentSpecificMemory!();
							if (typeof bytes === 'number' && Number.isFinite(bytes)) {
								lastUAMemoryBytes = bytes;
							}
						} catch (err) {
							console.warn('Error measuring UA memory:', err);
							// Ignore measurement errors
						}
					};
					// Immediate measurement so first log can include it
					void measureAndStore();
					uaMemoryInterval = setInterval(() => {
						void measureAndStore();
					}, 10_000);
				} else {
					console.debug(
						'performance.measureUserAgentSpecificMemory is not available; skipping UA memory interval'
					);
				}
			}

			trainWorker.onmessage = (event) => {
				const { type, ...data } = event.data;

				switch (type) {
					case 'visualizer.ready':
						console.log('[Main] Visualizer ready');
						break;
					case 'capture':
						visualizerQueries = data.queries as TensorQuery[];
						visualizerBoxes = data.boxes as MatchBox[];
						visualizerLayoutStep = data.step as number;
						visualizerLayoutRunId = data.runId as string;
						visualizerRenderWidth = data.width as number;
						visualizerRenderHeight = data.height as number;
						break;
					case 'visualizer.error':
						console.error('[Main] Visualizer error:', data.message);
						break;
					case 'ready':
						console.log('[Main] Worker is ready');
						resolve();
						workerReady.current = true;
						workerVersion.current += 1;
						break;
					case 'checkpoint.config': {
						const { requestId, config: cfg } = data as {
							requestId: string;
							config: Config;
						};
						const resolver = pendingPeekResolvers.get(requestId);
						if (resolver) {
							pendingPeekResolvers.delete(requestId);
							resolver(cfg);
						}
						break;
					}
					case 'metrics': {
						// Handle training metric logs
						if (!data.runId || !data.data) {
							console.error('[Main] Invalid metrics data:', data);
							return;
						}
						const step = data.metadata?.step as number | undefined;
						const combinedMetrics: Record<string, number | Record<string, unknown>> = {};
						for (const [metricName, value] of Object.entries(data.data)) {
							combinedMetrics[metricName] = value as number | Record<string, unknown>;
						}
						if (lastUAMemoryBytes !== null) {
							combinedMetrics['allocation/cpu_memory_mb'] = lastUAMemoryBytes / (1024 * 1024);
							lastUAMemoryBytes = null;
						}
						log(data.runId, combinedMetrics, { step });
						break;
					}
					case 'complete':
						console.log(`[Main] Training completed for run ${data.runId}`);
						trainingState.current = 'stopped';
						currentRun.current = null;
						void releaseScreenWakeLock();
						break;

					case 'restart': {
						console.log(`[Main] Worker requested restart for run ${data.runId}`);
						const buffer = data.buffer as Uint8Array<ArrayBufferLike>;
						const runId = data.runId as string;
						// Persist last session snapshot with checkpoint
						const run = runsMap.get(runId);
						if (run) {
							void lastSessionStore.set(run, buffer);
						}
						// Terminate and recreate worker
						trainWorker?.terminate();
						workerReady.current = false;
						// Ensure training state reflects continuity across restart
						trainingState.current = 'training';
						initializeWorker().then(() => {
							// Send start with resumeFrom to resume same run id
							trainWorker!.postMessage({
								type: 'start',
								data: { runId, config: $state.snapshot(config), resumeFrom: buffer }
							});
						});
						break;
					}
					case 'checkpoint': {
						const uint8array = data.buffer as Uint8Array<ArrayBufferLike> | undefined;
						const runId = data.runId as string | undefined;

						// Persist last session snapshot with checkpoint (always)
						if (uint8array && runId) {
							const run = runsMap.get(runId);
							if (run) {
								void lastSessionStore.set(run, uint8array);
							}

							// Fulfill all waiters if present
							for (const waiter of pendingCheckpointWaiters) {
								void waiter({ runId, buffer: uint8array });
							}
							pendingCheckpointWaiters.length = 0;
						}

						break;
					}
					case 'error':
						if (data.name === 'VRAMLimitExceededError') {
							console.error(`[Main] VRAM limit exceeded for run ${data.runId}:`, data.message);
							triggerVramLimitFlash();
						} else if (data.name === 'LowDiversityDatasetError') {
							console.error(
								`[Main] Low diversity dataset error for run ${data.runId}:`,
								data.message
							);
							triggerLowDiversityDatasetError();
						} else {
							console.error(`[Main] Training error for run ${data.runId}:`, data.message);
						}
						workerStopTraining();
						break;
					case 'paused':
						console.log('[Main] Training paused');
						trainingState.current = 'paused';
						// Request a save to persist checkpoint; session will be stored alongside when checkpoint arrives
						if (currentRun.current?.runId) {
							workerRequestSave();
						}
						break;
					case 'resumed':
						console.log('[Main] Training resumed');
						trainingState.current = 'training';
						break;
				}
			};

			trainWorker.onerror = (event) => {
				console.error('[Main] Worker onerror:', event);
				reject(new Error(event.error));
			};
		} catch (error) {
			console.error('[Main] Failed to create worker:', error);
			reject(error);
		}
	});
}

export function workerStartTraining(runId: string, resumeFrom?: Uint8Array<ArrayBufferLike>) {
	if (!trainWorker) {
		throw new Error('Worker not initialized');
	}

	trainWorker.postMessage({
		type: 'start',
		data: {
			runId: runId,
			config: $state.snapshot(config),
			resumeFrom,
			gpuPowerPreference: gpuPowerPreference.current
		}
	});

	trainingState.current = 'training';
	void acquireScreenWakeLock();
}

export function workerRequestSave() {
	if (!trainWorker) {
		throw new Error('Worker not initialized');
	}
	trainWorker.postMessage({ type: 'save' });
}

export function waitForNextCheckpoint(): Promise<CheckpointPayload> {
	return new Promise((resolve) => {
		pendingCheckpointWaiters.push(resolve);
	});
}

export function peekCheckpointConfig(buffer: Uint8Array<ArrayBufferLike>): Promise<Config> {
	if (!trainWorker) {
		return Promise.reject(new Error('Worker not initialized'));
	}
	const requestId = crypto.randomUUID();
	return new Promise<Config>((resolve) => {
		pendingPeekResolvers.set(requestId, resolve);
		trainWorker!.postMessage({ type: 'checkpoint.peekConfig', data: { requestId, buffer } });
	});
}

export async function workerStopTraining() {
	if (!trainWorker || trainingState.current === 'stopped') return;
	void releaseScreenWakeLock();

	// For now, we'll just terminate and recreate the worker
	// In a more sophisticated implementation, we'd send a stop message
	trainWorker.terminate();
	workerReady.current = false;
	trainingState.current = 'stopped';
	currentRun.current = null;

	// Recreate worker
	await initializeWorker();
}

export function workerPauseTraining() {
	if (!trainWorker || trainingState.current !== 'training') return;
	trainWorker.postMessage({ type: 'pause' });
}

export function workerResumeTraining() {
	if (!trainWorker || trainingState.current !== 'paused') return;
	trainWorker.postMessage({ type: 'resume' });
}

export function workerStep() {
	if (!trainWorker || trainingState.current === 'stopped') return;
	trainWorker.postMessage({ type: 'step' });
}

//
// Model inspection state
//

let parameterCount = $state<number | null>(null);
let modelIndex = $state<IndexState | null>(null);
let modelInspectionRequestId = $state<string | null>(null);
let isInspectingModel = $state(false);
let modelInspectionWorker: Worker | null = $state(null);

export function getParameterCount() {
	return parameterCount;
}

export function getModelIndex() {
	return modelIndex;
}

export function getIsInspectingModel() {
	return isInspectingModel;
}

export function setModelInspectionWorker(workerInstance: Worker | null) {
	modelInspectionWorker = workerInstance;
	// Trigger initial model inspection when worker is set
	if (modelInspectionWorker && !isInspectingModel) {
		setTimeout(() => requestModelInspection(), 0);
	}
}

// Export a function to manually trigger model inspection
export function triggerModelInspection() {
	if (modelInspectionWorker && !isInspectingModel) {
		setTimeout(() => requestModelInspection(), 0);
	}
}

function requestModelInspection() {
	if (!modelInspectionWorker || isInspectingModel) return;

	isInspectingModel = true;
	modelInspectionRequestId = crypto.randomUUID();

	try {
		modelInspectionWorker.postMessage({
			type: 'inspectModel',
			data: {
				config: $state.snapshot(config),
				requestId: modelInspectionRequestId,
				gpuPowerPreference: gpuPowerPreference.current
			}
		});
	} catch (error) {
		console.error('Failed to request model inspection:', error);
		isInspectingModel = false;
		modelInspectionRequestId = null;
	}
}

export function handleModelInspectionResponse(data: {
	requestId: string;
	parameterCount: number;
	vocabSize: number;
	modelIndex: IndexState;
}) {
	if (data.requestId === modelInspectionRequestId) {
		parameterCount = data.parameterCount;
		modelIndex = data.modelIndex;
		isInspectingModel = false;
		modelInspectionRequestId = null;
	}
}

export function handleModelInspectionError(data: { requestId: string; message: string }) {
	if (data.requestId === modelInspectionRequestId) {
		console.error('Model inspection error:', data.message);
		isInspectingModel = false;
		modelInspectionRequestId = null;
	}
}

export async function initializeModelInspectionWorker() {
	return new Promise<void>((resolve, reject) => {
		try {
			// Create the dedicated model inspection worker
			// eslint-disable-next-line svelte/prefer-svelte-reactivity
			modelInspectionWorker = new Worker(new URL('$lib/train/moduleWorker.ts', import.meta.url), {
				type: 'module',
				name: 'modelInspectionWorker'
			});

			console.log('[Main] Model inspection worker created successfully.');

			modelInspectionWorker.onmessage = (event) => {
				const { type, ...data } = event.data;

				switch (type) {
					case 'ready':
						console.log('[Main] Model inspection worker is ready');
						resolve();
						// Set the worker reference for model inspection
						setModelInspectionWorker(modelInspectionWorker);
						break;

					case 'modelInspection':
						handleModelInspectionResponse(data);
						break;

					case 'modelInspectionError':
						handleModelInspectionError(data);
						break;

					case 'error':
						console.error('[Main] Model inspection worker error:', data.message);
						break;
					case 'gpu.info': {
						const name = (data as { name?: string | null }).name ?? null;
						setGpuName(name ?? null);
						break;
					}
				}
			};

			modelInspectionWorker.onerror = (event) => {
				console.error('[Main] Model inspection worker error:', event.message, event);
				reject(new Error(event.error));
			};
		} catch (error) {
			console.error('[Main] Failed to create model inspection worker:', error);
			reject(error);
		}
	});
}

export async function initializeWorkers() {
	return Promise.all([initializeWorker(), initializeModelInspectionWorker()]);
}

export function cleanupWorkers() {
	if (trainWorker) {
		trainWorker.terminate();
		trainWorker = null;
	}
	if (modelInspectionWorker) {
		modelInspectionWorker.terminate();
		modelInspectionWorker = null;
	}
	// Clear UA memory interval
	if (uaMemoryInterval) {
		clearInterval(uaMemoryInterval);
		uaMemoryInterval = null;
	}
	lastUAMemoryBytes = null;

	void releaseScreenWakeLock();
}

//
// Visualizer APIs
//
// eslint-disable-next-line svelte/prefer-svelte-reactivity
const canvasesWithAttemptedInitialization = new Set<HTMLCanvasElement>();
export function initializeVisualizerCanvas(
	canvas: HTMLCanvasElement,
	labelPaddingCssPx: number = 0
) {
	if (!trainWorker) throw new Error('Worker not initialized');
	if (canvasesWithAttemptedInitialization.has(canvas)) return;
	const offscreen = canvas.transferControlToOffscreen();
	trainWorker.postMessage(
		{ type: 'visualizer.canvas', data: { canvas: offscreen, labelPaddingCssPx } },
		[offscreen]
	);
	canvasesWithAttemptedInitialization.add(canvas);
}

export function resizeVisualizer(width: number) {
	if (!trainWorker) return;
	trainWorker.postMessage({ type: 'visualizer.resize', data: { width } });
}

export function getVisualizerLayout() {
	return {
		boxes: visualizerBoxes,
		queries: visualizerQueries,
		step: visualizerLayoutStep,
		runId: visualizerLayoutRunId,
		width: visualizerRenderWidth,
		height: visualizerRenderHeight
	};
}

export function getWorkerVersion() {
	return workerVersion;
}

export function updateVisualizerScript(example: string, script: string | null) {
	if (!trainWorker) return;
	trainWorker.postMessage({ type: 'visualizer.updateScript', data: { example, script } });
	void lastSessionStore.updateVisualization({ example, script });
}

export function updateVisualizerTarget(target: 'train' | 'validation') {
	if (!trainWorker) return;
	config.visualization.target = target;
	trainWorker.postMessage({ type: 'visualizer.setTarget', data: { target } });
	void lastSessionStore.updateVisualization({ target });
}

export function updateVisualizerSelectedValidation({
	exampleIndex,
	tokenIndex
}: {
	exampleIndex: number;
	tokenIndex: number;
}) {
	if (!trainWorker) return;
	config.visualization.selectedValidation.exampleIndex = exampleIndex;
	config.visualization.selectedValidation.tokenIndex = tokenIndex;
	trainWorker.postMessage({
		type: 'visualizer.setSelectedValidation',
		data: { exampleIndex, tokenIndex }
	});
	void lastSessionStore.updateVisualization({ selectedValidation: { exampleIndex, tokenIndex } });
}
