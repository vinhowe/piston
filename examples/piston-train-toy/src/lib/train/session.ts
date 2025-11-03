import type { Config } from '$lib/workspace/config';
import type { StepData } from '$lib/workspace/runs.svelte';

import { getEffectiveVisualizationScript } from '$lib/workspace/visualizationExamples';
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

import type { CaptureMatch, RemoteCaptureStep } from './capture';
import type { BuiltData } from './data/pipeline';
import type {
	ToyAutoregressiveBatch,
	ToyBidirectionalBatch,
	ToyEncoderDecoderBatch,
	ToySequence
} from './data/toy/dataset';
import type { RNNEncoder, RNNEncoderDecoder } from './model/rnn';
import type { RunWorkerEvent, RunWorkerEventWithoutRunId, WorkerEvent } from './protocol';
import type { GeneratableModel, PistonCollateFnType, PistonDatasetType } from './types';

import { CaptureManager, makeCaptureMatchRemote, runValidationExampleForCapture } from './capture';
import { buildDataset, tensorWrap } from './data';
import { filterDatasetByHeldoutSamples } from './data/filter';
import { NaturalLanguageDataset } from './data/natural';
import { buildDataPipeline } from './data/pipeline';
import ToyDataset from './data/toy/dataset';
import {
	DecoderTransformer,
	EncoderDecoderTransformer,
	EncoderTransformer
} from './model/transformer';
import {
	type AnySchedulerState,
	buildCheckpoint,
	type CheckpointDataState,
	type CheckpointExtra,
	splitLoadedState
} from './utils/checkpoint';
import {
	calculateBlockSize,
	calculateParameterSum,
	calculateVocabSize,
	createCollateFn,
	createDataloader,
	createModel,
	initializeModel,
	seedPiston
} from './utils/model';
import { MarkStepModeIfEnabled, WeakModeIfEnabled } from './utils/modes';
import { configureOptimizerForModel } from './utils/optim';
import { forkRandom, seededRandom } from './utils/random';
import {
	buildValidationExamplesSubset,
	buildValidationLog,
	computeLikelihoodMetrics,
	computeNaturalValidationMetrics,
	computeToyValidationMetrics,
	type NaturalValidationExamples,
	prepareValidationExamples,
	type ToyValidationExamples,
	type ValidationExamples,
	type ValidationStep
} from './validation';
import { Visualizer } from './visualizer';

// @ts-expect-error polyfill
Symbol.dispose ||= Symbol.for('Symbol.dispose');

export class TrainingSession {
	readonly runId: string;
	private config: Config;
	private readonly post: (e: RunWorkerEventWithoutRunId) => void;
	private readonly resumeFrom?: Uint8Array<ArrayBufferLike>;

	private paused = false;
	private resolvePause: (() => void) | null = null;

	private visualizer: Visualizer | null = null;
	private visualizerDevice: GPUDevice | null = null;
	private visualizerCanvas: OffscreenCanvas | null = null;
	private visualizerLabelPaddingCssPx: number = 0;
	private readonly currentVisualizationSelectedValidation: {
		exampleIndex: number;
		tokenIndex: number;
	} = {
		exampleIndex: 0,
		tokenIndex: 0
	};

	private model!: GeneratableModel;
	private optimizer!: piston.Optimizer;
	private scheduler: LRScheduler<unknown> | undefined;
	private trainDataset!: PistonDatasetType;
	private blockSize!: number | { source: number; target: number };

	private isSetup: boolean = false;

	private startTimeMs: number | null = null;
	private lastLogTime: number | null = null;
	private lastLogStep: number | null = null;
	private stepCount: number = 0;

	private captureManager: CaptureManager | null = null;

	private dataPipeline!: BuiltData;

	private validationExamples: ValidationExamples | null = null;
	private validationCollateFn: PistonCollateFnType<Tensor> | null = null;
	private validationDataset: PistonDatasetType | null = null;

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

	setVisualizationScript(example: string, script: string | null) {
		// If a model is already active, rebuild the plan immediately; otherwise setup() will build it
		if (this.model) {
			const newCaptureManager = new CaptureManager();
			newCaptureManager.build(this.model, {
				enabled: this.config.training.enableVisualization,
				script: script ?? undefined
			});
			if (newCaptureManager.plan) {
				this.captureManager = newCaptureManager;
				this.config.visualization.example = example;
				this.config.visualization.script = script;
			}
		}
	}

	setVisualizationTarget(target: 'train' | 'validation') {
		this.config.visualization.target = target;
	}

	setVisualizationSelectedValidation({
		exampleIndex,
		tokenIndex
	}: {
		exampleIndex: number;
		tokenIndex: number;
	}) {
		this.currentVisualizationSelectedValidation.exampleIndex = exampleIndex;
		this.currentVisualizationSelectedValidation.tokenIndex = tokenIndex;
	}

	initVisualizerCanvas(canvas: OffscreenCanvas, labelPaddingCssPx: number = 0) {
		// Dispose old visualizer if any
		if (this.visualizer) {
			this.visualizer.dispose();
			this.visualizer = null;
		}

		this.visualizerCanvas = canvas;
		this.visualizerLabelPaddingCssPx = labelPaddingCssPx;
		const pistonDevice = piston.gpu.asWebGPUDevice();
		if (!pistonDevice) {
			throw new Error('Failed to get WebGPU device from piston.gpu');
		}
		this.visualizerDevice = pistonDevice;
		this.visualizer = new Visualizer(this.visualizerDevice);
		this.visualizer.init(this.visualizerCanvas);
		this.visualizer.setCssLabelPadding(this.visualizerLabelPaddingCssPx);
	}

	resizeVisualizer(width: number) {
		this.visualizer?.resize(width);
	}

	private async onCaptureMatches(step: number, matches: CaptureMatch[]) {
		try {
			await piston.gpu.markStep();
			const captureStep: Omit<RemoteCaptureStep, 'step'> = {
				type: 'capture',
				matches: matches.map(makeCaptureMatchRemote)
			};
			this.logMetrics({ 'visualization/matches': captureStep }, { step });
			const result = this.visualizer
				? await this.visualizer.renderCapture(matches)
				: { boxes: [], statsById: {}, width: 1, height: 1 };

			this.post({
				type: 'capture',
				queries: this.captureManager?.queries ?? [],
				step,
				boxes: result.boxes.map((box) => ({ ...box, match: makeCaptureMatchRemote(box.match) })),
				statsById: result.statsById,
				width: result.width,
				height: result.height
			});
		} catch (err) {
			// Fall back to logging the error and continue
			this.post({
				type: 'log',
				level: 'warn',
				message: `Visualizer render failed: ${String(err)}`
			});
		}
	}

	async saveLatestCheckpoint(): Promise<Uint8Array<ArrayBufferLike>> {
		if (!this.model) throw new Error('No model available to save');
		await piston.gpu.markStep();
		// Derive dataset state if available
		let dataState: CheckpointDataState | undefined = undefined;
		if (this.trainDataset) {
			if (this.trainDataset instanceof NaturalLanguageDataset) {
				dataState = {
					blockSize: this.blockSize,
					natural: this.trainDataset.exportState()
				};
			} else if (this.trainDataset instanceof ToyDataset) {
				dataState = {
					blockSize: this.blockSize,
					toy: {
						cursor: this.trainDataset.cursor,
						baseSeed: this.trainDataset.baseSeed,
						datasetName: this.config.data.dataset
					}
				};
			}
		}
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

		// Determine model type and create appropriate dataloader/model
		const isEncoderOnly = this.config.model.topology === 'encoder';
		const isDecoderOnly = this.config.model.topology === 'decoder';
		const isEncoderDecoder = this.config.model.topology === 'encoder-decoder';

		const seed = seedPiston(this.config);

		if (this.config.training.vramLimitMb.present) {
			piston.gpu.setVRAMLimit(BigInt(this.config.training.vramLimitMb.value * 1024 * 1024));
		}

		// Ensure shared-object allocation is enabled so buffer handles are stable across steps
		piston.gpu.setSharedObjectAllocationEnabled(this.config.training.sharedObjectAllocation);
		piston.gpu.setCachingEnabled(this.config.training.cachingEnabled);
		piston.gpu.setInplaceSupport(this.config.training.inplaceSupport);

		// Set up dataset; we use two generators so we change validation parameters without affecting
		// training
		const trainGenerator = seededRandom(seed);
		const maskGenerator = isEncoderOnly ? forkRandom(trainGenerator) : null;

		const trainDataset: PistonDatasetType = buildDataset(this.config, trainGenerator, 'train');
		// Restore dataset state if present
		if (resumePayload && resumePayload.dataState) {
			const dsState = resumePayload.dataState;
			if (trainDataset instanceof NaturalLanguageDataset && dsState.natural) {
				await trainDataset.importState(dsState.natural);
			} else if (trainDataset instanceof ToyDataset && dsState.toy) {
				// Restore cursor and baseSeed for toy datasets
				trainDataset.cursor = dsState.toy.cursor | 0;
				if (typeof dsState.toy.baseSeed === 'number') {
					trainDataset.baseSeed = dsState.toy.baseSeed;
				}
			}
		}
		this.trainDataset = trainDataset;

		const validationDisabled =
			('disableValidation' in this.trainDataset && this.trainDataset.disableValidation) || false;

		this.validationExamples = null;
		this.validationCollateFn = null;
		this.validationDataset = null;

		if (this.config.training.validation.present && !validationDisabled) {
			const validationGenerator = forkRandom(trainGenerator);
			this.validationDataset = buildDataset(this.config, validationGenerator, 'val');
			this.validationCollateFn = createCollateFn(
				this.config,
				this.validationDataset,
				maskGenerator,
				tensorWrap
			);
			this.validationExamples = await prepareValidationExamples(
				this.config,
				this.validationDataset,
				{
					isDecoderOnly,
					isEncoderDecoder
				}
			);
			// Filter training dataset against holdout examples without duplication
			let validationSequences: ToySequence[] | number[][];
			if ('toySequences' in this.validationExamples) {
				validationSequences = this.validationExamples.toySequences;
				this.trainDataset = filterDatasetByHeldoutSamples(
					this.trainDataset,
					this.config.data.dataset,
					validationSequences
				);
			} else if ('naturalSequences' in this.validationExamples) {
				validationSequences = this.validationExamples.naturalSequences;
				this.trainDataset = filterDatasetByHeldoutSamples(
					this.trainDataset,
					this.config.data.dataset,
					validationSequences
				);
			} else {
				throw new Error('Unsupported validation dataset');
			}
			console.debug(
				`Prepared ${validationSequences.length} validation examples for batch generation`
			);
		}

		if (validationDisabled) {
			console.debug('Validation disabled by dataset; skipping validation and holdout filtering.');
		}

		// Calculate vocab size using shared utility
		const vocabSize = calculateVocabSize(this.config, this.trainDataset);
		const [trainDataloaderForSizing] = createDataloader(
			this.config,
			this.trainDataset,
			trainGenerator,
			tensorWrap
		);
		const blockSize =
			this.blockSize !== undefined
				? this.blockSize
				: calculateBlockSize(this.config, trainDataloaderForSizing);
		this.blockSize = blockSize;

		const datasetName = this.config.data.dataset;

		console.debug(
			`Created dataset ${datasetName} with vocab size ${vocabSize} and block size ${blockSize}`
		);

		if (!isEncoderOnly && !isDecoderOnly && !isEncoderDecoder) {
			throw new Error(
				`Unsupported model type: ${this.config.model.topology}. Only 'encoder', 'decoder', and` +
					` 'encoder-decoder' are currently supported.`
			);
		}

		// Create model
		this.model = createModel(this.config, vocabSize, blockSize);

		// If starting from scratch, initialize model parameters
		if (!resumePayload) {
			initializeModel(this.config, this.model);

			// We need to flatten down initialization to the constant tensors they're on top of
			await piston.gpu.markStep();

			const parameterSum = new BigUint64Array(
				new Float64Array([await (await calculateParameterSum(this.model).to('cpu')).item()]).buffer
			);
			console.debug(`Initialization parameter sum: ${parameterSum}`);
		}

		// Build or refresh capture plan using CaptureManager
		this.captureManager = new CaptureManager();
		const scriptToUse = this.config.training.enableVisualization
			? getEffectiveVisualizationScript(
					this.config.visualization.example,
					this.config.visualization.script
				)
			: null;
		this.captureManager.build(this.model, {
			enabled: this.config.training.enableVisualization,
			script: scriptToUse
		});

		// Build and store the training data pipeline (iterator bound to current dataset/collate)
		this.dataPipeline = await buildDataPipeline(
			this.config,
			trainGenerator,
			maskGenerator,
			this.trainDataset
		);

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
		const optimizer = configureOptimizerForModel(
			this.model,
			isEncoderOnly,
			isEncoderDecoder,
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

			let captureSession: piston.CaptureSession | null = null;
			if (loggingStep && this.captureManager && this.config.visualization.target === 'train') {
				// Create session on top of weak mode for this step to capture forward ops
				captureSession = this.captureManager.createSession();
			}

			const weakModeUntilAfterBackward = new WeakModeIfEnabled(
				this.config.training.useWeakTensorReferences,
				{
					label: 'train/forward_through_backward'
				}
			);

			let loss: Tensor;
			try {
				if (this.config.model.topology === 'encoder') {
					// For BERT: batch contains [inputIds, labels, attentionMask]
					const { tensors } = batch as ToyBidirectionalBatch<Tensor>;
					const [inputIds, bertLabels, attentionMask] = tensors;

					let computedLoss: Tensor | null = null;
					if (this.model instanceof EncoderTransformer) {
						[, , , computedLoss] = (this.model as EncoderTransformer).forward(
							await inputIds.to('gpu'),
							{
								attentionMask: await attentionMask.to('gpu'),
								targets: await bertLabels.to('gpu')
							}
						);
					} else {
						[, , computedLoss] = (this.model as RNNEncoder).forward(await inputIds.to('gpu'), {
							targets: await bertLabels.to('gpu')
						});
					}

					if (!computedLoss) {
						throw new Error('No loss tensor returned from encoder-only model');
					}

					loss = computedLoss;
				} else if (this.config.model.topology === 'encoder-decoder') {
					// For Transformer or RNN seq2seq: batch contains [encoderInputs, decoderInputs, decoderTargets]
					const { tensors } = batch as ToyEncoderDecoderBatch<Tensor>;
					const [encoderInputs, decoderInputs, decoderTargets] = tensors;
					let computedLoss: Tensor | null;
					if (this.model instanceof EncoderDecoderTransformer) {
						[, computedLoss] = (this.model as EncoderDecoderTransformer).forward(
							await encoderInputs.to('gpu'),
							await decoderInputs.to('gpu'),
							{ targets: await decoderTargets.to('gpu') }
						);
					} else {
						[, computedLoss] = (this.model as RNNEncoderDecoder).forward(
							await encoderInputs.to('gpu'),
							await decoderInputs.to('gpu'),
							{ targets: await decoderTargets.to('gpu') }
						);
					}

					if (!computedLoss) {
						throw new Error('No loss tensor returned from encoder-decoder model');
					}

					loss = computedLoss;
				} else {
					// For GPT: batch contains [inputs, targets]
					const { tensors } = batch as ToyAutoregressiveBatch<Tensor>;
					const [inputs, gptTargets] = tensors;
					const [, computedLoss] = (this.model as DecoderTransformer).forward(
						await inputs.to('gpu'),
						{
							targets: await gptTargets.to('gpu')
						}
					);

					if (!computedLoss) {
						throw new Error('No loss tensor returned from decoder-only model');
					}

					loss = computedLoss;
				}

				weakModeUntilAfterBackward.pin(loss);

				loss.backward();

				if (captureSession && this.onCaptureMatches) {
					try {
						const matches = this.captureManager!.finalize(captureSession, 0);
						await this.onCaptureMatches(this.stepCount, matches);
					} finally {
						captureSession[Symbol.dispose]();
					}
					captureSession = null;
				}
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
					await this.optimizer.step();
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
								let validationExamplesSubset: ValidationExamples | null = null;
								if (this.config.training.validation.completions.amount === 'subset') {
									validationExamplesSubset = buildValidationExamplesSubset(
										this.validationExamples,
										this.config.training.validation.completions.subsetSize
									);
								} else {
									validationExamplesSubset = this.validationExamples;
								}
								if (this.validationDataset instanceof ToyDataset) {
									const validationStepData = await computeToyValidationMetrics(
										this.model,
										this.validationDataset,
										validationExamplesSubset as ToyValidationExamples,
										this.config.training.validation,
										{
											isDecoderOnly: this.config.model.topology === 'decoder',
											isEncoderDecoder: this.config.model.topology === 'encoder-decoder',
											includeTargets:
												this.stepCount === 0 && (this.validationDataset.hasCanonicalTargets ?? true)
										}
									);
									validationLog = buildValidationLog(validationStepData);
								} else if (this.validationDataset instanceof NaturalLanguageDataset) {
									const validationStepData = await computeNaturalValidationMetrics(
										this.model,
										this.validationDataset,
										validationExamplesSubset as NaturalValidationExamples,
										this.config.training.validation,
										{
											isDecoderOnly: this.config.model.topology === 'decoder',
											includeTargets: this.stepCount === 0,
											maskRatio: this.config.data.maskRatio
										}
									);
									validationLog = buildValidationLog(validationStepData);
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

					if (
						loggingStep &&
						this.captureManager &&
						this.validationExamples &&
						this.validationCollateFn &&
						this.onCaptureMatches &&
						this.config.visualization.target === 'validation'
					) {
						// Create session on top of weak mode for this step to capture forward ops
						captureSession = this.captureManager.createSession();

						await runValidationExampleForCapture(
							this.model,
							this.validationExamples,
							this.validationCollateFn,
							this.currentVisualizationSelectedValidation.exampleIndex
						);

						// runValidationExampleForCapture actually ends up… training on the validation set,
						// so we need to zero out the gradients here
						this.optimizer.zeroGrad(true);

						try {
							const matches = this.captureManager!.finalize(
								captureSession!,
								this.currentVisualizationSelectedValidation.exampleIndex
							);
							await this.onCaptureMatches(this.stepCount, matches);
						} finally {
							captureSession![Symbol.dispose]();
						}
						captureSession = null;
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
					continue;
				}
			}
		}
	}
}
