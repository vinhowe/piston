import { config } from './config.svelte';
import { currentRun, log } from './runs.svelte';
import { triggerLowDiversityDatasetError, triggerVramLimitFlash } from './ui.svelte';

// Train state
let trainWorker: Worker | null = $state(null);
export const workerReady = $state({ current: false });
export const workerVersion = $state({ current: 0 });
export const trainingState = $state<{ current: 'training' | 'paused' | 'stopped' }>({
	current: 'stopped'
});

// UA memory measurement state (main thread only)
let uaMemoryInterval: ReturnType<typeof setInterval> | null = null;
let lastUAMemoryBytes: number | null = null;

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
					case 'ready':
						console.log('[Main] Worker is ready');
						resolve();
						workerReady.current = true;
						workerVersion.current += 1;
						break;

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
						break;

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

export function workerStartTraining() {
	if (!trainWorker) {
		throw new Error('Worker not initialized');
	}

	const run = currentRun.current;

	if (!run) {
		throw new Error('No current run');
	}

	trainWorker.postMessage({
		type: 'start',
		data: JSON.parse(JSON.stringify(run))
	});

	trainingState.current = 'training';
}

export async function workerStopTraining() {
	if (!trainWorker || trainingState.current === 'stopped') return;

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
let modelInspectionRequestId = $state<string | null>(null);
let isInspectingModel = $state(false);
let modelInspectionWorker: Worker | null = $state(null);

export function getParameterCount() {
	return parameterCount;
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
				requestId: modelInspectionRequestId
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
}) {
	if (data.requestId === modelInspectionRequestId) {
		parameterCount = data.parameterCount;
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
}
