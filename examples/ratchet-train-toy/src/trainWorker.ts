import { Trainer } from '@ratchet-ml/ratchet-web-train';
import {
	evalExampleGenerators,
	trainBatchGenerators,
	type EvalConfig,
	type TaskConfigMap
} from './tasks';

let trainer: Trainer;
let sessionCounter = 0;
let currentSession = 0;
const trainingSessions: Record<number, boolean> = {};

export interface TrainerConfig {
	vocab_size: number;
	n_embd: number;
	n_layer: number;
	n_head: number;
	block_size: number;
	batch_size: number;
	dataset: keyof typeof trainBatchGenerators;
	task_parameters: TaskConfigMap[keyof TaskConfigMap];
	activation: string;
	attention_only: boolean;
	position_encoding: string;
	seed?: number;
	layernorm_position: string;
	optimizer: {
		optimizer_type: string;
		lr: number;
		beta1: number;
		beta2: number;
		eps: number;
		weight_decay: number;
		momentum: number;
		scheduler_type: string;
		scheduler_factor: number;
		scheduler_steps: number;
		scheduler_eta_min: number;
	};
}

function markStopTraining() {
	Object.keys(trainingSessions).forEach((key) => {
		trainingSessions[Number(key)] = false;
	});
}

async function initializeTrainer(config: TrainerConfig) {
	// Stop all existing training sessions
	markStopTraining();

	trainer = await new Trainer(config);
	self.postMessage({ type: 'modelReady' });
	currentSession = sessionCounter++;
	trainingSessions[currentSession] = true;
	trainingLoop(currentSession, config);
}

async function trainingLoop(sessionId: number, config: TrainerConfig) {
	if (!trainer) {
		console.error('Trainer not initialized');
		return;
	}

	// Get the appropriate task generator
	const taskGenerator = trainBatchGenerators[config.dataset];
	const evalConfig = evalExampleGenerators[config.dataset] as EvalConfig<typeof config.dataset>;
	if (!taskGenerator) {
		console.error(`Unknown dataset type: ${config.dataset}`);
		return;
	}

	let step = 0;
	while (trainingSessions[sessionId]) {
		// Skip if this isn't the current session anymore
		if (sessionId !== currentSession) {
			break;
		}

		try {
			// Generate a new batch using the selected task generator
			const [input, target] = generateTask(config);

			// Train on the batch
			const result = await trainer.train_on_batch(input, target);
			const logits = result.get('logits') as Map<string, Uint8Array | number[]>;
			const loss = result.get('loss') as Map<string, number>;
			const attn_masks = result.get('attn_masks') as Map<string, Uint8Array | number[]>;
			const usage_bytes = trainer.usage_bytes();

			// Only send message if this is still the current session
			if (sessionId === currentSession) {
				self.postMessage({
					type: 'step',
					input: input,
					target: target,
					loss: {
						total: loss.get('total'),
						tokens: loss.get('tokens')
					},
					learning_rate: result.get('learning_rate'),
					usage_bytes,
					attn_masks: {
						data: attn_masks.get('data'),
						shape: attn_masks.get('shape')
					},
					logits: {
						data: logits.get('data'),
						shape: logits.get('shape')
					}
				});
			}
		} catch (error: Error | unknown) {
			console.error('Training error in worker:', error);
			if (sessionId === currentSession) {
				self.postMessage({
					type: 'error',
					error: error instanceof Error ? error.message : String(error)
				});
			}
			break;
		}
		// Yield to keep the worker responsive
		await new Promise((resolve) => setTimeout(resolve, 0));
	}
}

function generateTask(config: TrainerConfig): [number[][], number[][]] {
	const taskGenerator = trainBatchGenerators[config.dataset];
	if (!taskGenerator) {
		throw new Error(`Unknown dataset: ${config.dataset}`);
	}

	// Combine batch size with task-specific parameters
	const taskConfig = {
		batchSize: config.batch_size,
		...config.task_parameters
	};

	return taskGenerator(taskConfig);
}

self.onmessage = async (e: MessageEvent) => {
	if (e.data.type === 'stop') {
		markStopTraining();
		return;
	}
	// If we receive a configuration object, initialize a new trainer
	if (typeof e.data === 'object') {
		await initializeTrainer(e.data);
	}
};

self.postMessage({ type: 'ready' });
