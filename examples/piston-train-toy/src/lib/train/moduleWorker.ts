import * as piston from '@piston-ml/piston-web';

import type { WorkerCommand, WorkerEvent } from './protocol';

import { TrainingSession } from './session';
import { inspectModel } from './utils/model';

let session: TrainingSession | undefined;

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

import type { Config } from '$lib/workspace/config';

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
		case 'start':
			try {
				const { runId: runIdFromData, config } = data as {
					runId: string;
					config: Config;
				};
				currentExecutionSource = `[Training:${runIdFromData}]`;

				console.info(`Starting training for run ${runIdFromData}`);
				session = new TrainingSession(runIdFromData, config, postEvent);
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
		case 'inspectModel':
			try {
				const { config, requestId } = data as { config: Config; requestId: string };
				currentExecutionSource = '[ModelInspection]';

				console.debug('Inspecting model...');
				const result = inspectModel(config);

				self.postMessage({
					type: 'modelInspection',
					requestId,
					parameterCount: result.parameterCount,
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
