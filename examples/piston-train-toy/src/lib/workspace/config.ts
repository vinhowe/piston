import type {
	ConstantConfig,
	CosineAnnealingConfig,
	ExponentialConfig,
	LinearConfig,
	StepConfig
} from '@piston-ml/piston-web';

export type {
	AdditionConfig,
	ModularAdditionConfig,
	RepeatConfig,
	SlapjackConfig,
	SortConfig,
	TwoSumConfig,
	ZerosConfig
} from '../train/data/toy/config';

import type { DATASET_CONFIG_DEFAULTS } from '$lib/train/data';

import type { TOY_DATASET_CONFIG_DEFAULTS } from '../train/data/toy/config';

export interface DropoutConfig {
	present: boolean;
	embedding: number;
	transformer: {
		attention: number;
		residual: number;
	};
	rnn: {
		interLayer: number;
	};
}

export type TransformerDropoutConfig = Omit<DropoutConfig, 'rnn'>;
export type RNNDropoutConfig = Omit<DropoutConfig, 'transformer'>;

export interface ValidationCompletionsConfig {
	present: boolean;
	decodingBatchSize: number;
	amount: 'all' | 'subset';
	subsetSize: number;
}

export interface ValidationConfig {
	present: boolean;
	valSteps: number;
	batchSize: number;
	temperature: number;
	completions: ValidationCompletionsConfig;
	useKvCache: boolean;
}

export interface TrainingConfig {
	logSteps: number;
	limitTraining: {
		present: boolean;
		steps: number;
	};
	batchSize: number;
	dropout: DropoutConfig;
	validation: ValidationConfig;
	labelSmoothing: {
		present: boolean;
		value: number;
	};
	randomSeed: {
		present: boolean;
		value: string;
	};
	vramLimitMb: {
		present: boolean;
		value: number;
	};
	gradNorm: {
		track: boolean;
		errorIfNonfinite: boolean;
	};
	clipGradNorm: {
		present: boolean;
		value: number;
	};
	useWeakTensorReferences: boolean;
	sharedObjectAllocation: boolean;
	cachingEnabled: boolean;
	inplaceSupport: boolean;
	enableVisualization: boolean;
	restartEverySteps: number;
}

export interface TransformerAttentionConfig {
	present: boolean;
	nKeyValueHeads: number;
	groupedQueryAttention: {
		present: boolean;
		queryHeadsPerKeyValueHead: number;
	};
	gating: AttentionGatingConfig;
	sinks: {
		present: boolean;
	};
}

export interface DataConfig {
	dataset: keyof typeof DATASET_CONFIG_DEFAULTS;
	trainOnPrompt: boolean;
	datasets: typeof TOY_DATASET_CONFIG_DEFAULTS;
	maskRatio: number;
	specialTokens: {
		includeEos: boolean;
	};
	natural: {
		contextSize: number;
		vocabSize: 'char' | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 | 65536;
	};
}

export interface AlibiConfig {
	maxBias: number;
}

export interface RoPEConfig {
	base: number;
}

export interface PositionEncodingConfig {
	present: boolean;
	type: 'sinusoidal' | 'learned' | 'rope' | 'alibi';
	alibi: AlibiConfig;
	rope: RoPEConfig;
}

export type LayerNormPosition = 'pre' | 'post';
export type NormalizationType = 'layernorm' | 'rmsnorm';

export interface LayerNormalizationConfig {
	type: NormalizationType;
	eps: number;
	transformer: {
		present: boolean;
		position: LayerNormPosition;
	};
	rnn: {
		withinCell: boolean;
		betweenLayers: boolean;
	};
}

export interface QKNormConfig {
	present: boolean;
	type: NormalizationType;
	eps: number;
}

export interface SoftcapConfig {
	attention: {
		present: boolean;
		value: number;
	};
	logits: {
		present: boolean;
		value: number;
	};
}

export interface TransformerNormalizationConfig {
	qkNorm: QKNormConfig;
	softcap: SoftcapConfig;
}

export type Activation = 'relu' | 'relu2' | 'gelu' | 'silu' | 'sigmoid' | 'swiglu' | 'tanh';

export interface AttentionGatingSitesConfig {
	afterSdpaOutput: boolean;
	afterValueProjection: boolean;
	afterKeyProjection: boolean;
	afterQueryProjection: boolean;
	afterFinalOutputProjection: boolean;
}

export interface AttentionGatingConfig {
	present: boolean;
	activation: Activation;
	sites: AttentionGatingSitesConfig;
}

export interface MLPConfig {
	present: boolean;
	activation: Activation;
	hiddenExpansionFactor: number;
	variant: 'standard' | 'gated';
}

export type ModelType = 'decoder' | 'encoder' | 'encoder-decoder';
export type ModelFamily = 'transformer' | 'rnn';

export interface MultiplicativeRNNAttentionConfig {
	scaleByInverseSqrtHiddenSize: boolean;
}

export interface RNNAttentionConfig {
	present: boolean;
	type: 'additive' | 'multiplicative';
	inputFeedingProjection: boolean;
	multiplicative: MultiplicativeRNNAttentionConfig;
}

export interface RNNHiddenStateProjectionConfig {
	present: boolean;
	size: number;
}

export interface RNNEmbeddingConfig {
	type: 'learned' | 'one-hot';
	learned: {
		size: number;
	};
}

export interface RNNConfig {
	cellType: 'lstm' | 'gru' | 'rnn';
	// RNN token embedding configuration
	embedding: RNNEmbeddingConfig;
	separateHiddenSize: {
		present: boolean;
		value: number;
	};
	initialization: RNNInitializationConfig;
	hiddenStateProjection: RNNHiddenStateProjectionConfig;
	encoder: {
		bidirectional: boolean;
	};
	encoderDecoderAttention: RNNAttentionConfig;
}

export type ProjectionInitializationStrategy = 'layer-scaled' | 'zero';
export interface ProjectionInitializationConfig {
	present: boolean;
	strategy: ProjectionInitializationStrategy;
}

export interface TransformerInitializationConfig {
	present: boolean;
	std: number;
	projections: {
		attention: ProjectionInitializationConfig;
		mlp: ProjectionInitializationConfig;
		lmHead: ProjectionInitializationConfig;
	};
}

export interface LSTMInitializationConfig {
	forgetGateBias: {
		present: boolean;
		value: number;
	};
}

export type XavierInitializationDistribution = 'uniform' | 'normal';

export interface RNNInitializationConfig {
	present: boolean;
	xavierInputColumns: {
		present: boolean;
		distribution: XavierInitializationDistribution;
	};
	orthogonalRecurrentColumns: boolean;
	perGateOrthogonalBlocks: boolean;
	zeroBiases: boolean;
	gru: {
		updateGateBias: {
			present: boolean;
			value: number;
		};
	};
	lstm: {
		forgetGateBias: {
			present: boolean;
			value: number;
		};
	};
}

export interface TransformerConfig {
	headDim: number;
	attention: TransformerAttentionConfig;
	positionalEncoding: PositionEncodingConfig;
	initialization: TransformerInitializationConfig;
	normalization: TransformerNormalizationConfig;
	mlp: MLPConfig;
}

export interface ModelConfig {
	family: ModelFamily;
	topology: ModelType;
	layers: number;
	tieEmbeddingsAndLmHead: boolean;
	roundVocabSizeToNearestMultiple: {
		present: boolean;
		value: number;
	};
	encoderDecoder: {
		decoderLayers: number;
		encoderLayers: number;
	};
	layerNormalization: LayerNormalizationConfig;
	transformer: TransformerConfig;
	rnn: RNNConfig;
}

export interface OptimizerConfig {
	type: 'AdamW' | 'Adam' | 'SGD' | 'Muon';
	lr: number;
	weightDecay: {
		present: boolean;
		value: number;
		useWeightDecayGroups: boolean;
	};
	warmupSteps: { present: boolean; value: number };
	lrScheduler: {
		present: boolean;
		type: string;
		stepSchedule: StepConfig;
		constantSchedule: ConstantConfig;
		cosineAnnealingSchedule: CosineAnnealingConfig;
		exponentialSchedule: ExponentialConfig;
		linearSchedule: LinearConfig;
	};
	adam: {
		beta1: number;
		beta2: number;
		eps: number;
		amsgrad: boolean;
	};
	sgd: {
		dampening: number;
		momentum: number;
		nesterov: boolean;
	};
	muon: {
		nsSteps: number;
		momentum: number;
		nesterov: boolean;
	};
}

export interface VisualizationConfig {
	// If null, the effective script comes from the selected example
	script: string | null;
	// Selected example id or "custom"
	example: string;
	// current training step or the selected validation example
	target: 'train' | 'validation';
	selectedValidation: {
		exampleIndex: number;
		tokenIndex: number;
	};
}

export interface Config {
	version: number;
	// Currently selected layered preset id; null means no preset
	preset: string | null;
	training: TrainingConfig;
	data: DataConfig;
	model: ModelConfig;
	optimizer: OptimizerConfig;
	visualization: VisualizationConfig;
}
