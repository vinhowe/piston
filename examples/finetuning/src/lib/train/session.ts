import type { Config } from '$lib/workspace/config';
import type { StepData } from '$lib/workspace/runs.svelte';

import {
	CosineAnnealingLR,
	ExponentialLR,
	LinearLR,
	type LRScheduler,
	SequentialLR,
	StepLR,
	type Tensor
} from '@piston-ml/piston-web';
import * as piston from '@piston-ml/piston-web';

import type { NaturalLanguageDataset } from './data/natural';
import type { BuiltData } from './data/pipeline';
import type { RunWorkerEvent, RunWorkerEventWithoutRunId, WorkerEvent } from './protocol';
import type { GeneratableModel, NaturalCollateFnType } from './types';

import { buildDataset, tensorWrap } from './data';
import { filterDatasetByHeldoutSamples } from './data/filter';
import { buildDataPipeline } from './data/pipeline';
import { GPT, GPT2_BLOCK_SIZE, GPT2_VOCAB_SIZE } from './model/gpt';
import {
	type AnySchedulerState,
	buildCheckpoint,
	type CheckpointDataState,
	type CheckpointExtra,
	splitLoadedState
} from './utils/checkpoint';
// import { initTransformerParameters } from './utils/init';
import { calculateParameterSum, createCollateFn, createModel } from './utils/model';
import { MarkStepModeIfEnabled, WeakModeIfEnabled } from './utils/modes';
import { configureOptimizers } from './utils/optim';
import {
	buildValidationExamplesSubset,
	buildValidationLog,
	computeLikelihoodMetrics,
	computeNaturalValidationMetrics,
	type NaturalValidationExamples,
	prepareNaturalValidationExamples,
	type ValidationStep
} from './validation';

// @ts-expect-error polyfill
Symbol.dispose ||= Symbol.for('Symbol.dispose');

export class TrainingSession {
	readonly runId: string;
	private config: Config;
	private readonly post: (e: RunWorkerEventWithoutRunId) => void;
	private readonly resumeFrom?: Uint8Array<ArrayBufferLike>;

	private paused = false;
	private resolvePause: (() => void) | null = null;

	private model!: GeneratableModel;
	private optimizer!: piston.Optimizer;
	private scheduler: LRScheduler<unknown> | undefined;
	private trainDataset!: NaturalLanguageDataset;
	private blockSize!: number;

	private isSetup: boolean = false;

	private startTimeMs: number | null = null;
	private lastLogTime: number | null = null;
	private lastLogStep: number | null = null;
	private stepCount: number = 0;

	private dataPipeline!: BuiltData;

	private validationExamples: NaturalValidationExamples | null = null;
	private validationCollateFn: NaturalCollateFnType<Tensor> | null = null;
	private validationDataset: NaturalLanguageDataset | null = null;
	// This is a little bit gross, but it's a straightforward way to make sure we have valid targets
	// when we resume from a checkpoint. If I ever do another pass over this code, this will be first
	// to go.
	private includeTargetsOnNextValidation: boolean = false;

	constructor(
		runId: string,
		config: Config,
		post: (e: WorkerEvent) => void,
		resumeFrom?: Uint8Array<ArrayBufferLike>
	) {
		this.runId = runId;
		this.config = config;
		this.post = (e: RunWorkerEventWithoutRunId) =>
			// We only post the subset of events that have runId in the payload
			(post as (e: RunWorkerEvent) => void)({ ...e, runId: this.runId });
		this.resumeFrom = resumeFrom;
		if (resumeFrom) {
			this.includeTargetsOnNextValidation = true;
		}
	}

	async pause() {
		if (this.paused) return;
		this.paused = true;
		await new Promise<void>((resolve) => {
			this.resolvePause = resolve;
		});
		this.resolvePause = null;
		this.post({ type: 'paused' });
	}

	resume() {
		this.paused = false;
		this.post({ type: 'resumed' });
	}

	async save() {
		try {
			if (this.paused) {
				try {
					if (!this.model) {
						// Defer save until model is ready
						this.post({ type: 'log', level: 'info', message: 'Save requested before model ready' });
						return;
					}
					await piston.gpu.markStep();
					const buffer = await this.saveLatestCheckpoint();
					this.post({ type: 'checkpoint', buffer });
				} catch (e) {
					this.post({
						type: 'error',
						message: String(e)
					});
				}
			} else {
				throw new Error('Saving during training is not supported');
			}
		} catch (e) {
			this.post({ type: 'error', message: String(e) });
		}
	}

	async saveLatestCheckpoint(): Promise<Uint8Array<ArrayBufferLike>> {
		if (!this.model) throw new Error('No model available to save');
		await piston.gpu.markStep();
		// Derive dataset state if available
		const dataState = {
			blockSize: this.blockSize,
			...this.trainDataset.exportState()
		};
		const { tensors, extra } = buildCheckpoint(
			this.model,
			this.optimizer!,
			this.stepCount,
			this.config ? JSON.parse(JSON.stringify(this.config)) : null,
			this.scheduler,
			dataState,
			this.startTimeMs ?? undefined
		);
		return piston.save(tensors, extra);
	}

	private logMetrics(
		data: { [metricName: string]: Omit<StepData, 'step'> },
		metadata?: { step?: number }
	) {
		this.post({ type: 'metrics', data, metadata });
	}

	private async setup() {
		if (this.isSetup) {
			return;
		}

		// Log initial memory
		const initialMemoryMB = Number(piston.gpu.usageBytes()) / (1024 * 1024);
		console.debug(`Initial memory: ${initialMemoryMB} MB`);

		// If resuming from a checkpoint, parse and use checkpoint config
		let resumePayload: {
			modelState: Record<string, piston.Tensor>;
			optimizerPacked?: { state: Record<number, unknown>; paramGroups: piston.ParamGroupConfig[] };
			schedulerState?: unknown;
			numSteps: number;
			config: Config;
			dataState?: CheckpointDataState;
			startTimeMs?: number;
		} | null = null;

		if (this.resumeFrom) {
			const loaded = piston.load(this.resumeFrom, piston.gpu);
			const split = splitLoadedState(
				loaded as { state: Record<string, Tensor>; extra?: CheckpointExtra }
			);
			resumePayload = {
				modelState: split.modelState,
				optimizerPacked: split.optimizerState as unknown as {
					state: Record<number, unknown>;
					paramGroups: piston.ParamGroupConfig[];
				},
				schedulerState: split.schedulerState,
				numSteps: split.numSteps,
				config: split.config,
				dataState: split.dataState,
				startTimeMs: split.startTimeMs
			};
			if (resumePayload.config) {
				this.config = resumePayload.config as Config;
			}
			// If blockSize present in extras, prefer it
			if (split.dataState && split.dataState.blockSize !== undefined) {
				this.blockSize = split.dataState.blockSize;
			}
		}

		if (this.config.training.vramLimitMb.present) {
			piston.gpu.setVRAMLimit(BigInt(this.config.training.vramLimitMb.value * 1024 * 1024));
		}

		// Ensure shared-object allocation is enabled so buffer handles are stable across steps
		piston.gpu.setSharedObjectAllocationEnabled(this.config.training.sharedObjectAllocation);
		piston.gpu.setCachingEnabled(this.config.training.cachingEnabled);
		piston.gpu.setInplaceSupport(this.config.training.inplaceSupport);

		const trainDataset: NaturalLanguageDataset = buildDataset(this.config, 'train');
		// Restore dataset state if present
		if (resumePayload && resumePayload.dataState) {
			const dsState = resumePayload.dataState;
			await trainDataset.importState(dsState);
		}
		this.trainDataset = trainDataset;

		const validationDisabled =
			('disableValidation' in this.trainDataset && this.trainDataset.disableValidation) || false;

		this.validationExamples = null;
		this.validationCollateFn = null;
		this.validationDataset = null;

		if (this.config.training.validation.present && !validationDisabled) {
			this.validationDataset = buildDataset(this.config, 'val');
			this.validationCollateFn = createCollateFn(tensorWrap);
			this.validationExamples = await prepareNaturalValidationExamples(
				this.config,
				this.validationDataset!
			);
			// Filter training dataset against holdout examples without duplication
			const validationSequences: number[][] = this.validationExamples.naturalSequences;
			this.trainDataset = filterDatasetByHeldoutSamples(
				this.trainDataset,
				this.config.data.dataset,
				validationSequences
			);
			console.debug(
				`Prepared ${validationSequences.length} validation examples for batch generation`
			);
		}

		if (validationDisabled) {
			console.debug('Validation disabled by dataset; skipping validation and holdout filtering.');
		}

		const vocabSize = GPT2_VOCAB_SIZE;
		const blockSize = GPT2_BLOCK_SIZE;
		this.blockSize = blockSize;

		console.debug(
			`Created dataset ${this.trainDataset.name} with vocab size ${vocabSize} and block size ${blockSize}`
		);

		// Create model
		this.model = createModel(this.config);

		// If starting from scratch, initialize model parameters
		if (!resumePayload) {
			// initTransformerParameters(this.model, this.config);

			// We need to flatten down initialization to the constant tensors they're on top of
			await piston.gpu.markStep();

			const parameterSum = new BigUint64Array(
				new Float64Array([await (await calculateParameterSum(this.model).to('cpu')).item()]).buffer
			);
			console.debug(`Initialization parameter sum: ${parameterSum}`);
		}

		// Build and store the training data pipeline (iterator bound to current dataset/collate)
		this.dataPipeline = await buildDataPipeline(this.config, this.trainDataset);

		// If resuming, load model state BEFORE creating the optimizer so param identities match
		let startStep = 0;
		if (resumePayload) {
			this.model.loadStateDict(resumePayload.modelState, { strict: false });
			startStep = (resumePayload.numSteps ?? 0) + 1;
			this.stepCount = startStep;
			// If checkpoint carried a startTimeMs, use it for wall-clock continuity
			if (typeof resumePayload.startTimeMs === 'number') {
				this.startTimeMs = resumePayload.startTimeMs;
			}
		}

		// Create optimizer based on model type, using the (possibly restored) model parameters
		const optimizer = configureOptimizers(
			this.model,
			['transformer.h'],
			'lm_head',
			this.config.optimizer,
			piston.gpu
		);
		this.optimizer = optimizer;

		// If resuming, load optimizer state NOW that groups refer to current model parameters
		if (resumePayload && resumePayload.optimizerPacked) {
			optimizer.loadStateDict(resumePayload.optimizerPacked as piston.StateDict);
		}

		// Create learning rate scheduler if configured
		if (this.config.optimizer.lrScheduler.present) {
			const lrConfig = this.config.optimizer.lrScheduler;
			switch (lrConfig.type) {
				case 'step':
					this.scheduler = new StepLR(
						this.optimizer,
						lrConfig.stepSchedule.stepSize,
						lrConfig.stepSchedule.gamma
					);
					break;
				case 'cosine':
					this.scheduler = new CosineAnnealingLR(
						this.optimizer,
						lrConfig.cosineAnnealingSchedule.tMax,
						lrConfig.cosineAnnealingSchedule.etaMin
					);
					break;
				case 'exponential':
					this.scheduler = new ExponentialLR(this.optimizer, lrConfig.exponentialSchedule.gamma);
					break;
				case 'linear':
					this.scheduler = new LinearLR(
						this.optimizer,
						lrConfig.linearSchedule.startFactor,
						lrConfig.linearSchedule.endFactor,
						lrConfig.linearSchedule.totalIters
					);
					break;
				default:
					throw new Error(`Unknown scheduler type: ${lrConfig.type}`);
			}

			if (this.scheduler && this.config.optimizer.warmupSteps.present) {
				const n = this.config.optimizer.warmupSteps.value;
				if (n > 0) {
					const warmup = new LinearLR(optimizer, 1e-8, 1.0, n);
					this.scheduler = new SequentialLR(optimizer, [warmup, this.scheduler], [n]);
				}
			}
		} else if (this.config.optimizer.warmupSteps.present) {
			const n = this.config.optimizer.warmupSteps.value;
			if (n > 0) {
				this.scheduler = new LinearLR(optimizer, 1e-8, 1.0, n);
			}
		}

		// If resuming, load scheduler state after it is created
		if (resumePayload && this.scheduler && resumePayload.schedulerState) {
			this.scheduler.loadStateDict(resumePayload.schedulerState as AnySchedulerState);
		}

		this.model.train();

		this.isSetup = true;
	}

	async step({ manual = false }: { manual?: boolean } = {}): Promise<
		IteratorResult<void, 'completed' | 'restarted'>
	> {
		if (this.startTimeMs == null) {
			this.startTimeMs = Date.now();
		}
		if (this.lastLogStep == null) {
			this.lastLogStep = this.stepCount;
		}
		try {
			const iterNext = await this.dataPipeline.train.iterator.next();
			if (iterNext.done) {
				return { done: true, value: 'completed' };
			}
			const batch = iterNext.value;
			performance.mark('stepStart');
			// Reset peak GPU memory tracking at the start of the step
			piston.gpu.markUsageBytesStep();

			let isLastStep = false;
			if (
				this.config.training.limitTraining.present &&
				this.stepCount + 1 >= this.config.training.limitTraining.steps
			) {
				console.log(
					`Stopping training at step ${this.stepCount} because it reached the limit of ${this.config.training.limitTraining.steps} steps`
				);
				isLastStep = true;
			}

			const loggingStep =
				manual || isLastStep || this.stepCount % this.config.training.logSteps === 0;

			const weakModeUntilAfterBackward = new WeakModeIfEnabled(
				this.config.training.useWeakTensorReferences,
				{
					label: 'train/forward_through_backward'
				}
			);

			let loss: Tensor;
			try {
				// For GPT: batch contains [inputs, targets]
				const { tensors } = batch;
				const [inputs, gptTargets] = tensors;
				const [, computedLoss] = (this.model as GPT).forward(await inputs.to('gpu'), {
					targets: await gptTargets.to('gpu')
				});

				if (!computedLoss) {
					throw new Error('No loss tensor returned from decoder-only model');
				}

				loss = computedLoss;

				weakModeUntilAfterBackward.pin(loss);

				loss.backward();
			} finally {
				weakModeUntilAfterBackward[Symbol.dispose]();
			}

			const weakModeForOptimizerStep = new WeakModeIfEnabled(
				this.config.training.useWeakTensorReferences,
				{
					label: 'train/optimizer_step'
				}
			);

			let gradNorm: Tensor | undefined;
			try {
				const weakMarkStepMode = new MarkStepModeIfEnabled(
					this.config.training.useWeakTensorReferences
				);
				weakModeForOptimizerStep.pin(loss);

				if (this.config.training.gradNorm.track) {
					if (this.config.training.clipGradNorm.present) {
						gradNorm = weakModeForOptimizerStep.pin(
							piston.clipGradNorm_(this.model.parameters(), this.config.training.clipGradNorm.value)
						);
					} else if (loggingStep) {
						// If we're not clipping gradients, we can just get the total gradient norm
						gradNorm = weakModeForOptimizerStep.pin(
							piston.getTotalGradNorm(this.model.parameters())
						);
					}
				}

				try {
					// await this.optimizer.step();
					await piston.gpu.markStep();
				} finally {
					weakMarkStepMode[Symbol.dispose]();
				}
			} finally {
				// TODO: decide if it's okay that we're disposing the mode twice here
				weakModeForOptimizerStep[Symbol.dispose]();
			}

			const finalWeakModeForStep = new WeakModeIfEnabled(
				this.config.training.useWeakTensorReferences,
				{
					label: 'train/final'
				}
			);

			try {
				// We've kept loss strong; we'll want to make sure we get rid of it
				// Batch tensors are created outside of weak mode, so we manually mark them as weak
				finalWeakModeForStep.markWeak([loss, gradNorm, batch.tensors]);

				this.optimizer.zeroGrad(true);

				// Step learning rate scheduler if present
				if (this.scheduler) {
					this.scheduler.step();
				}

				if (
					this.config.training.validation.present &&
					(this.stepCount % this.config.training.validation.valSteps === 0 || isLastStep) &&
					this.validationExamples &&
					this.validationDataset &&
					this.validationCollateFn
				) {
					try {
						let valLoss = Number.NaN;
						let perplexity = Number.NaN;
						let validationLog: Record<string, number | Omit<ValidationStep, 'step'>> = {};

						if (this.validationExamples) {
							if (this.config.training.validation.completions.present) {
								let validationExamplesSubset: NaturalValidationExamples | null = null;
								if (this.config.training.validation.completions.amount === 'subset') {
									validationExamplesSubset = buildValidationExamplesSubset(
										this.validationExamples,
										this.config.training.validation.completions.subsetSize
									);
								} else {
									validationExamplesSubset = this.validationExamples;
								}
								const validationStepData = await computeNaturalValidationMetrics(
									this.model,
									this.validationDataset,
									validationExamplesSubset as NaturalValidationExamples,
									this.config.training.validation
								);
								validationLog = buildValidationLog(validationStepData);
								if (this.includeTargetsOnNextValidation) {
									this.includeTargetsOnNextValidation = false;
								}
							}

							const result = await computeLikelihoodMetrics(
								this.model,
								this.validationExamples!,
								this.validationCollateFn!
							);

							valLoss = result.valLoss;
							perplexity = result.perplexity;

							const logData: Record<string, number | Omit<ValidationStep, 'step'>> = {
								...validationLog,
								'validation/loss': valLoss,
								'validation/perplexity': perplexity
							};
							this.logMetrics(logData, { step: this.stepCount });
						}
					} catch (error) {
						console.error('Error during batch validation:', error);
					}
				}

				if (loggingStep) {
					const currentTime = Date.now();
					const totalElapsedSeconds = (currentTime - this.startTimeMs!) / 1000;

					// Calculate delta time and steps since last log
					const deltaTime = (currentTime - this.lastLogTime!) / 1000;
					const deltaSteps = this.stepCount - this.lastLogStep!;

					// Calculate steps per second and words per second based on delta
					const stepsPerSecond = deltaSteps > 0 ? deltaSteps / deltaTime : 0;

					// Calculate words per second (tokens per second)
					// Get sequence length from the first tensor in the batch
					let sequenceLength = 0;

					// Encoder-decoder will have three tensors in its batch, but we can just use the first one
					const [inputs] = batch.tensors;
					sequenceLength = inputs.shape[1]; // [batch_size, seq_len]

					const tokensPerStep = this.config.training.batchSize * sequenceLength;
					const tokensPerSecond = deltaSteps > 0 ? (deltaSteps * tokensPerStep) / deltaTime : 0;

					const activeMap = piston.__pistonActiveTensors();
					const activeTensors = Array.from(activeMap.values()).reduce((s, v) => s + v.length, 0);

					let lossItem: number | null = null;

					const lossCpu = await loss.to('cpu');
					lossItem = await lossCpu.item();

					if (lossItem === null) {
						throw new Error('Loss item is null?');
					}

					const peakUsageMb = Number(piston.gpu.peakUsageBytes()) / (1024 * 1024);

					const logData: Record<string, number> = {
						'train/loss': lossItem,
						'allocation/active_tensor_count': activeTensors,
						'allocation/gpu_memory_mb': peakUsageMb,
						'speed/steps_per_second': stepsPerSecond,
						'speed/step': this.stepCount,
						'speed/tokens_per_second': tokensPerSecond,
						'speed/wall_clock_seconds': totalElapsedSeconds
					};

					if (gradNorm) {
						const gradNormCpu = await gradNorm.to('cpu');
						const gradNormItem = await gradNormCpu.item();
						if (this.config.training.gradNorm.errorIfNonfinite && !isFinite(gradNormItem)) {
							throw new Error(`Gradient norm was nonfinite, so it cannot be clipped.`);
						}
						logData['train/grad_norm'] = gradNormItem;
					}

					// Log current learning rate if scheduler is present
					const currentLr = this.optimizer.paramGroups[0].lr;
					if (currentLr) {
						logData['optimizer/learning_rate'] = currentLr;
					}

					this.logMetrics(logData, { step: this.stepCount });

					// Update last log time and step
					this.lastLogTime = currentTime;
					this.lastLogStep = this.stepCount;
				}

				// Trigger periodic checkpoint save (non-restart) if configured
				if (this.config.training.checkpointEverySteps.present) {
					const checkpointEvery = this.config.training.checkpointEverySteps.value;
					if (checkpointEvery > 0 && (this.stepCount + 1) % checkpointEvery === 0) {
						try {
							const bytes = await this.saveLatestCheckpoint();
							this.post({ type: 'checkpoint', buffer: bytes });
						} catch (e) {
							// Non-fatal; continue training
							this.post({ type: 'log', level: 'warn', message: String(e) });
						}
					}
				}

				// Trigger periodic restart if configured
				const restartEvery = this.config.training.restartEverySteps ?? 0;
				const willRestart = restartEvery > 0 && (this.stepCount + 1) % restartEvery === 0;
				if (willRestart) {
					console.debug(`Routine restart at step ${this.stepCount}`);
					await piston.gpu.markStep();
					const bytes = await this.saveLatestCheckpoint();
					this.post({ type: 'restart', buffer: bytes });
					return { done: true, value: 'restarted' };
				}

				if (isLastStep) {
					return { done: true, value: 'completed' };
				}
			} finally {
				finalWeakModeForStep[Symbol.dispose]();
			}

			this.stepCount++;

			performance.mark('stepEnd');
		} catch (error) {
			console.error(`Error during training: ${error}`);
			throw error;
		}
		return { done: false, value: undefined };
	}

	async start(): Promise<void> {
		await this.setup();
		while (true) {
			if (this.paused) {
				if (this.resolvePause) {
					this.resolvePause();
				}
				return;
			}
			const { done, value } = await this.step();
			if (done) {
				if (value === 'completed') {
					this.post({ type: 'complete' });
					break;
				}
				if (value === 'restarted') {
					return;
				}
			}
		}
	}
}
