import { type TaskConfigMap, tasks, type TaskSpec, type SimpleTokenizer } from '$lib/tasks';
import { Trainer, init } from '@piston-ml/piston-web';

let trainer: Trainer;
let sessionCounter = 0;
let currentSession = 0;
const trainingSessions: Record<number, boolean> = {};
let saveCheckpointPath: string | null = null;

export interface TrainerConfig {
	vocab_size: number;
	n_embd: number;
	n_layer: number;
	n_head: number;
	block_size: number;
	batch_size: number;
	dataset: keyof typeof tasks;
	task_parameters: TaskConfigMap[keyof TaskConfigMap];
	activation: string;
	attention_only: boolean;
	position_encoding: string;
	seed?: number;
	layernorm_position: string;
	label_smoothing: number;
	caching_enabled: boolean;
	inplace_support: boolean;
	debug_selection: string;
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
	await init();
	// Stop all existing training sessions
	markStopTraining();

	const task = tasks[config.dataset] as TaskSpec<TaskConfigMap[keyof TaskConfigMap]>;
	const tokenizer = task.createTokenizer(config.task_parameters);

	// Get the appropriate task generator
	if (!task) {
		console.error(`Unknown dataset type: ${config.dataset}`);
		return;
	}

	trainer = await new Trainer({
		...config,
		// Add 1 to the vocab size to account for the end-of-sequence token
		vocab_size: tokenizer.lastToken
	});
	self.postMessage({ type: 'modelReady' });
	currentSession = sessionCounter++;
	trainingSessions[currentSession] = true;
	trainingLoop(currentSession, config, task, tokenizer);
}

async function trainingLoop(
	sessionId: number,
	config: TrainerConfig,
	task: TaskSpec<TaskConfigMap[keyof TaskConfigMap]>,
	tokenizer: SimpleTokenizer
) {
	if (!trainer) {
		console.error('Trainer not initialized');
		return;
	}

	let step = 0;
	while (trainingSessions[sessionId]) {
		// Skip if this isn't the current session anymore
		if (sessionId !== currentSession) {
			break;
		}

		// Check if we need to save a checkpoint
		if (saveCheckpointPath !== null) {
			try {
				// Get the URL from the save_checkpoint method instead of handling the download directly
				const url = await trainer.save_checkpoint();
				self.postMessage({
					type: 'checkpoint_saved',
					success: true,
					url: url,
					filename: saveCheckpointPath
				});
			} catch (error) {
				self.postMessage({
					type: 'checkpoint_saved',
					success: false,
					error: error instanceof Error ? error.message : String(error)
				});
			}
			// Reset the path after saving
			saveCheckpointPath = null;
		}

		try {
			// Combine batch size with task-specific parameters before invoking the generator
			const taskConfig = {
				batchSize: config.batch_size,
				...config.task_parameters
			};
			const [input, target] = task.trainBatch(taskConfig);

			// Train on the batch
			const result = await trainer.train_on_batch(input, target);

			const logOutput = await trainer.take_step_log();
			const usingCache = logOutput?.cached ?? false;
			const totalElapsed = 0;
			const averageElapsed = 0;
			const logits = result.get('logits') as Map<string, Uint8Array | number[]>;
			const loss = result.get('loss') as Map<string, number>;
			const attn_masks = result.get('attn_masks') as Map<string, Uint8Array | number[]>;
			const usage_bytes = trainer.usage_bytes();

			let winCount = null;
			const EVAL_TRIAL_COUNT = 5;

			const { example: evalExample, metric: evalMetric } = task.eval(config.task_parameters);

			// After every n steps, evaluate the model
			if (step % 50 === 0) {
				const [streamingSequence, streamingTarget] = evalExample();
				let streamingExampleCompletion: number[] | null = null;
				const streamingExampleLogits: number[] = [];
				let streamingExampleMetric: boolean | null = null;

				// For the first eval, we'll stream it to the frontend
				await trainer.generate(
					streamingSequence,
					/* max_tokens= */ streamingTarget.length + 1,
					(tokens: number[], logitsObj: { shape: number[]; data: number[] }) => {
						// For the first call (prompt), just store logits
						if (streamingExampleCompletion === null) {
							streamingExampleCompletion = [];
							streamingExampleLogits.push(...logitsObj.data);
							return;
						}

						// For each subsequent token, append it and evaluate
						streamingExampleCompletion.push(...tokens);
						streamingExampleLogits.push(...logitsObj.data);

						// Run the metric on the current completion
						const evalResult = evalMetric(
							streamingExampleCompletion,
							streamingTarget,
							streamingSequence
						);

						streamingExampleMetric = evalResult.every((result) => result);

						// Send the streaming eval result to the frontend
						if (sessionId === currentSession) {
							self.postMessage({
								type: 'evalStreaming',
								sequence: streamingSequence.map((t) => tokenizer.ids[t]),
								completion: streamingExampleCompletion.map((t) => tokenizer.ids[t]),
								target: streamingTarget.map((t) => tokenizer.ids[t]),
								evalResult,
								logits: {
									data: streamingExampleLogits,
									shape: logitsObj.shape
								}
							});
						}
					}
				);

				winCount = streamingExampleMetric === true ? 1 : 0;

				// We'll do best of 5 evals, so we'll do the rest non-streaming
				for (let i = 0; i < EVAL_TRIAL_COUNT - 1; i++) {
					const [sequence, target] = evalExample();
					const result = await trainer.generate(sequence, target.length + 1);
					const tokens = result.get('tokens') as number[];
					const evalResult = evalMetric(tokens, target, sequence);
					if (evalResult.every((result) => result)) {
						winCount++;
					}
				}
			}

			// Only send message if this is still the current session
			if (sessionId === currentSession) {
				self.postMessage({
					type: 'step',
					totalElapsed,
					averageElapsed,
					usingCache,
					input: input.map((x) => x.map((t) => tokenizer.ids[t])),
					target: target.map((t) => t.map((t) => tokenizer.ids[t])),
					accuracy: winCount === null ? null : winCount / EVAL_TRIAL_COUNT,
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
		step++;
		// Yield to keep the worker responsive
		await new Promise((resolve) => setTimeout(resolve, 0));
	}
}

self.onmessage = async (e: MessageEvent) => {
	if (e.data.type === 'stop') {
		markStopTraining();
		return;
	}
	if (e.data.type === 'save_checkpoint') {
		saveCheckpointPath = e.data.path;
		return;
	}
	// If we receive a configuration object, initialize a new trainer
	if (typeof e.data === 'object') {
		await initializeTrainer(e.data);
	}
};

self.postMessage({ type: 'ready' });
