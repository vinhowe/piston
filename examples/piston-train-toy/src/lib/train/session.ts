import type { Config } from '$lib/workspace/config';

import {
	CosineAnnealingLR,
	ExponentialLR,
	LinearLR,
	type LRScheduler,
	StepLR,
	type Tensor
} from '@piston-ml/piston-web';
import * as piston from '@piston-ml/piston-web';

import type { BuiltData } from './data/pipeline';
import type {
	ToyAutoregressiveBatch,
	ToyBidirectionalBatch,
	ToyDatasetLike,
	ToyEncoderDecoderBatch
} from './data/toy/dataset';
import type { RunWorkerEvent, RunWorkerEventWithoutRunId, WorkerEvent } from './protocol';
import type { GeneratableModel } from './types';

import { tensorWrap } from './data';
import { buildDataPipeline } from './data/pipeline';
import { buildToyDataset } from './data/toy';
import {
	DecoderTransformer,
	EncoderDecoderTransformer,
	EncoderTransformer
} from './model/transformer';
import {
	calculateBlockSize,
	calculateVocabSize,
	createDataloader,
	createModel
} from './utils/model';
import { configureOptimizerForModel } from './utils/optim';
import { seededRandom } from './utils/random';

// @ts-expect-error polyfill
Symbol.dispose ||= Symbol.for('Symbol.dispose');

export class TrainingSession {
	readonly runId: string;
	private config: Config;
	private readonly post: (e: RunWorkerEventWithoutRunId) => void;

	private model!: GeneratableModel;
	private optimizer!: piston.Optimizer;
	private scheduler: LRScheduler<unknown> | undefined;
	private trainDataset!: ToyDatasetLike;
	private blockSize!: number | { source: number; target: number };

	private isSetup: boolean = false;

	private startTimeMs: number | null = null;
	private lastLogTime: number | null = null;
	private lastLogStep: number | null = null;
	private stepCount: number = 0;

	private dataPipeline!: BuiltData;

	constructor(runId: string, config: Config, post: (e: WorkerEvent) => void) {
		this.runId = runId;
		this.config = config;
		this.post = (e: RunWorkerEventWithoutRunId) =>
			// We only post the subset of events that have runId in the payload
			(post as (e: RunWorkerEvent) => void)({ ...e, runId: this.runId });
	}

	private logMetrics(data: { [metricName: string]: number }, metadata?: { step?: number }) {
		this.post({ type: 'metrics', data, metadata });
	}

	private async setup() {
		if (this.isSetup) {
			return;
		}

		// Determine model type and create appropriate dataloader/model
		const isEncoderOnly = this.config.model.topology === 'encoder';
		const isDecoderOnly = this.config.model.topology === 'decoder';
		const isEncoderDecoder = this.config.model.topology === 'encoder-decoder';

		// Ensure shared-object allocation is enabled so buffer handles are stable across steps
		piston.gpu.setSharedObjectAllocationEnabled(this.config.training.sharedObjectAllocation);
		piston.gpu.setCachingEnabled(this.config.training.cachingEnabled);
		piston.gpu.setInplaceSupport(this.config.training.inplaceSupport);

		// Set up dataset; we use two generators so we change validation parameters without affecting
		// training
		const trainGenerator = seededRandom();

		this.trainDataset = buildToyDataset(this.config, trainGenerator);

		// Calculate vocab size using shared utility
		const vocabSize = calculateVocabSize(this.trainDataset);
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

		// Build and store the training data pipeline (iterator bound to current dataset/collate)
		this.dataPipeline = await buildDataPipeline(this.config, trainGenerator, this.trainDataset);
		// Create optimizer based on model type, using the (possibly restored) model parameters
		this.optimizer = configureOptimizerForModel(
			this.model,
			isEncoderOnly,
			isEncoderDecoder,
			this.config.optimizer,
			piston.gpu
		);

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
			const iterNext = await this.dataPipeline.iterator.next();
			if (iterNext.done) {
				return { done: true, value: 'completed' };
			}
			const batch = iterNext.value;
			performance.mark('stepStart');
			// Reset peak GPU memory tracking at the start of the step
			piston.gpu.markUsageBytesStep();

			const loggingStep = manual || this.stepCount % this.config.training.logSteps === 0;

			let loss: Tensor;
			if (this.config.model.topology === 'encoder') {
				// For BERT: batch contains [inputIds, labels, attentionMask]
				const { tensors } = batch as ToyBidirectionalBatch<Tensor>;
				const [inputIds, bertLabels, attentionMask] = tensors;

				let computedLoss: Tensor | null = null;
				[, , , computedLoss] = (this.model as EncoderTransformer).forward(
					await inputIds.to('gpu'),
					{
						attentionMask: await attentionMask.to('gpu'),
						targets: await bertLabels.to('gpu')
					}
				);

				if (!computedLoss) {
					throw new Error('No loss tensor returned from encoder-only model');
				}

				loss = computedLoss;
			} else if (this.config.model.topology === 'encoder-decoder') {
				// For Transformer: batch contains [encoderInputs, decoderInputs, decoderTargets]
				const { tensors } = batch as ToyEncoderDecoderBatch<Tensor>;
				const [encoderInputs, decoderInputs, decoderTargets] = tensors;
				const [, computedLoss] = (this.model as EncoderDecoderTransformer).forward(
					await encoderInputs.to('gpu'),
					await decoderInputs.to('gpu'),
					{ targets: await decoderTargets.to('gpu') }
				);

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

			loss.backward();

			await this.optimizer.step();

			this.optimizer.zeroGrad(true);

			// Step learning rate scheduler if present
			if (this.scheduler) {
				this.scheduler.step();
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

				let lossItem: number | null = null;

				const lossCpu = await loss.to('cpu');
				lossItem = await lossCpu.item();

				if (lossItem === null) {
					throw new Error('Loss item is null?');
				}

				const logData: Record<string, number> = {
					'train/loss': lossItem,
					'speed/steps_per_second': stepsPerSecond,
					'speed/step': this.stepCount,
					'speed/tokens_per_second': tokensPerSecond,
					'speed/wall_clock_seconds': totalElapsedSeconds
				};
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
