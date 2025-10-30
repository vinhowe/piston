import { currentRun, log } from './runs.svelte';
import { triggerLowDiversityDatasetError } from './ui.svelte';

// Train state
let trainWorker: Worker | null = $state(null);
export const workerReady = $state({ current: false });
export const workerVersion = $state({ current: 0 });
export const trainingState = $state<{ current: 'training' | 'stopped' }>({
	current: 'stopped'
});

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
						log(data.runId, combinedMetrics, { step });
						break;
					}
					case 'complete':
						console.log(`[Main] Training completed for run ${data.runId}`);
						trainingState.current = 'stopped';
						currentRun.current = null;
						break;

					case 'error':
						if (data.name === 'LowDiversityDatasetError') {
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

export function cleanupWorker() {
	if (trainWorker) {
		trainWorker.terminate();
		trainWorker = null;
	}
}
