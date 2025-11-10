import type { Config } from '$lib/workspace/config';
import type { Tensor } from '@piston-ml/piston-web';

import * as piston from '@piston-ml/piston-web';

import type { WorkerCommand, WorkerEvent } from './protocol';

import { TrainingSession } from './session';
import { type CheckpointExtra, splitLoadedState } from './utils/checkpoint';
import { inspectModel } from './utils/model';

let session: TrainingSession | undefined;
let pendingVisualizerCanvas: { canvas: OffscreenCanvas; labelPaddingCssPx: number } | null = null;

// Console Interception
const originalConsole = {
	log: console.log.bind(console),
	error: console.error.bind(console),
	warn: console.warn.bind(console),
	info: console.info.bind(console),
	debug: console.debug.bind(console)
};

function formatArgs(args: unknown[]) {
	return args
		.map((arg) => {
			if (typeof arg === 'object' && arg !== null) {
				try {
					return JSON.stringify(arg);
				} catch (_: unknown) {
					return '[Unserializable Object]';
				}
			}
			return String(arg);
		})
		.join(' ');
}

interface LogInfo {
	level: string;
	message: string;
	source: string;
	lineno?: number;
	colno?: number;
}

function sendLog(level: string, message: string, source: string = '[Worker]') {
	// Check if this is a WASM log and parse it
	const wasmLogRegex = /^\[WASM ([^:]+):(\d+)(?::(\d+))?\] (.*)$/;
	const match = message.match(wasmLogRegex);

	if (level === 'error' && message.startsWith('panicked at')) {
		const lines = message.split('\n');
		if (lines[1].startsWith('VRAM limit exceeded')) {
			self.postMessage({ type: 'log', level: 'error', message: lines[1] });
			self.postMessage({
				type: 'error',
				runId: session?.runId,
				message: 'VRAM limit exceeded',
				name: 'VRAMLimitExceededError'
			});
			return;
		} else {
			self.postMessage({ type: 'error', runId: session?.runId, message });
		}
	}

	if (match) {
		// Handle WASM logs with parsed source info
		const [, filepath, lineno, colno, actualMessage] = match;
		const logInfo: LogInfo = {
			level,
			message: actualMessage,
			source: `[WASM] ${filepath}`,
			lineno: parseInt(lineno, 10),
			...(colno && { colno: parseInt(colno, 10) })
		};
		self.postMessage({ type: 'log', ...logInfo });
	} else {
		// Handle regular logs
		const logInfo: LogInfo = {
			level,
			message,
			source
		};
		self.postMessage({ type: 'log', ...logInfo });
	}
}

// Wrap console methods before importing Piston to catch its logs
Object.keys(originalConsole).forEach((level) => {
	(console as unknown as Record<string, (...args: unknown[]) => void>)[level] = (
		...args: unknown[]
	) => {
		const message = formatArgs(args);

		// Use sendLog which will handle WASM log parsing internally
		sendLog(level, message, currentExecutionSource);

		// Also call original console for debugging
		originalConsole[level as keyof typeof originalConsole](...args);
	};
});

// Global error handler - catches unhandled errors
self.addEventListener('error', (event) => {
	const errorMessage = `Uncaught Error: ${event.message} at ${event.filename}:${event.lineno}:${event.colno}`;
	sendLog('error', errorMessage);
	if (event.error?.stack) {
		sendLog('error', `${event.error.stack}`);
	}
});

// Unhandled promise rejection handler - catches unhandled promise rejections
self.addEventListener('unhandledrejection', (event) => {
	const errorMessage = `Unhandled Promise Rejection: ${event.reason}`;
	sendLog('error', errorMessage);
	if (event.reason?.stack) {
		sendLog('error', `${event.reason.stack}`);
	}
	// Prevent the default browser behavior (logging to console)
	event.preventDefault();
});

// Intercept and override the default error reporting
const originalOnError = self.onerror;
self.onerror = (message, source, lineno, colno, error) => {
	const errorMessage = `Global Error: ${message} at ${source}:${lineno}:${colno}`;
	sendLog('error', errorMessage);
	if (error?.stack) {
		sendLog('error', `${error.stack}`);
	}
	// Call original handler if it exists
	if (originalOnError) {
		return originalOnError(message, source, lineno, colno, error);
	}
	// Prevent default browser error handling
	return true;
};

//
// End Console Interception
//

// Track current execution context for logging (will be reassigned during execution)
let currentExecutionSource = '[Worker]';

function postEvent(e: WorkerEvent) {
	self.postMessage(e);
}

function startTraining() {
	if (!session) return;
	const runId = session.runId;
	session.start().catch((error: unknown) => {
		console.error('Training error:', error);
		self.postMessage({
			type: 'error',
			runId,
			message: error instanceof Error ? error.message : String(error),
			name: error instanceof Error ? error.name : undefined,
			stack: error instanceof Error ? error.stack : undefined
		});
	});
}

// Message handler for worker
self.addEventListener('message', async (event) => {
	const raw = event.data as WorkerCommand | { type: string; data?: unknown };
	const type: string = (raw as { type: string }).type;
	const data: unknown = (raw as { data?: unknown }).data;

	switch (type) {
		case 'save': {
			session?.save();
			break;
		}
		case 'pause': {
			session?.pause();
			break;
		}
		case 'resume': {
			session?.resume();
			startTraining();
			break;
		}
		case 'step': {
			if (!session) break;
			await session.pause();
			await session.step({ manual: true });
			break;
		}
		case 'visualizer.updateScript': {
			try {
				const { script, example } = data as { script: string; example: string };
				session?.setVisualizationScript(example, script ?? null);
				self.postMessage({ type: 'visualizer.ready' });
			} catch (e) {
				console.error('Failed to update visualizer script', e);
				self.postMessage({ type: 'visualizer.error', message: String(e) });
			}
			break;
		}
		case 'visualizer.canvas': {
			try {
				const payload = data as { canvas: OffscreenCanvas; labelPaddingCssPx?: number };
				const labelPaddingCssPx = payload.labelPaddingCssPx ?? 0;
				pendingVisualizerCanvas = { canvas: payload.canvas, labelPaddingCssPx };
				if (session) {
					session.initVisualizerCanvas(payload.canvas, labelPaddingCssPx);
				}
				self.postMessage({ type: 'visualizer.ready' });
			} catch (e) {
				console.error('Visualizer init failed', e);
				self.postMessage({ type: 'visualizer.error', message: String(e) });
			}
			break;
		}
		case 'visualizer.resize': {
			try {
				const { width } = data as { width: number };
				session?.resizeVisualizer(width);
			} catch (e) {
				console.error('Visualizer resize failed', e);
			}
			break;
		}
		case 'visualizer.setTarget': {
			try {
				const { target } = data as { target: 'train' | 'validation' };
				session?.setVisualizationTarget(target);
			} catch (e) {
				console.error('Visualizer set target failed', e);
			}
			break;
		}
		case 'visualizer.setSelectedValidation': {
			try {
				const { exampleIndex, tokenIndex } = data as { exampleIndex: number; tokenIndex: number };
				session?.setVisualizationSelectedValidation({ exampleIndex, tokenIndex });
			} catch (e) {
				console.error('Visualizer set selected validation failed', e);
			}
			break;
		}
		case 'start':
			try {
				const {
					runId: runIdFromData,
					config,
					resumeFrom,
					gpuPowerPreference
				} = data as {
					runId: string;
					config: Config;
					resumeFrom?: Uint8Array<ArrayBufferLike>;
					gpuPowerPreference?: 'high-performance' | 'low-power';
				};
				currentExecutionSource = `[Training:${runIdFromData}]`;

				// Apply GPU power preference before any GPU initialization
				if (gpuPowerPreference) {
					await piston.applyGpuPowerPreference(gpuPowerPreference);
				}

				console.info(`Starting training for run ${runIdFromData}`);
				session = new TrainingSession(runIdFromData, config, postEvent, resumeFrom);
				if (pendingVisualizerCanvas) {
					try {
						session.initVisualizerCanvas(
							pendingVisualizerCanvas.canvas,
							pendingVisualizerCanvas.labelPaddingCssPx
						);
						self.postMessage({ type: 'visualizer.ready' });
					} catch (e) {
						console.error('Visualizer init (deferred) failed', e);
						self.postMessage({ type: 'visualizer.error', message: String(e) });
					}
				}
				startTraining();
			} catch (error: unknown) {
				console.error('Training error:', error);
				self.postMessage({
					type: 'error',
					runId: (data as { runId?: string })?.runId,
					message: error instanceof Error ? error.message : String(error),
					name: error instanceof Error ? error.name : undefined,
					stack: error instanceof Error ? error.stack : undefined
				});
			}
			break;
		case 'checkpoint.peekConfig': {
			const { requestId, buffer } = data as {
				requestId: string;
				buffer: Uint8Array<ArrayBufferLike>;
			};
			const loaded = piston.load(buffer, piston.gpu);
			const split = splitLoadedState(
				loaded as { state: Record<string, Tensor>; extra?: CheckpointExtra }
			);
			self.postMessage({ type: 'checkpoint.config', requestId, config: split.config });
			break;
		}
		case 'inspectModel':
			try {
				const { config, requestId, gpuPowerPreference } = data as {
					config: Config;
					requestId: string;
					gpuPowerPreference?: 'high-performance' | 'low-power';
				};
				currentExecutionSource = '[ModelInspection]';

				// Apply GPU power preference before any GPU usage
				if (gpuPowerPreference) {
					await piston.applyGpuPowerPreference(gpuPowerPreference);
					(globalThis as unknown as { piston: typeof piston }).piston = piston;
				}

				console.debug('Inspecting model...');
				const result = inspectModel(config);

				self.postMessage({
					type: 'modelInspection',
					requestId,
					parameterCount: result.parameterCount,
					modelIndex: result.modelIndex,
					vocabSize: result.vocabSize,
					blockSize: result.blockSize
				});
			} catch (error: unknown) {
				console.error('Model inspection error:', error);
				self.postMessage({
					type: 'modelInspectionError',
					requestId: (data as { requestId?: string })?.requestId ?? '',
					message: error instanceof Error ? error.message : String(error)
				});
			}
			break;

		default:
			console.warn(`Unknown message type: ${type}`);
			break;
	}
});

// Initialize Piston, then signal that the worker is ready
piston
	.init()
	.then(() => {
		console.info('Piston initialized');
		self.postMessage({ type: 'ready' });
	})
	.catch((error: unknown) => {
		console.error('Error initializing Piston:', error);
		self.postMessage({
			type: 'error',
			message: error instanceof Error ? error.message : String(error)
		});
	});
