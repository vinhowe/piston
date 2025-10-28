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
}

export interface ValidationConfig {
	present: boolean;
	valSteps: number;
	batchSize: number;
	temperature: number;
	useKvCache: boolean;
}

export interface TrainingConfig {
	logSteps: number;
	batchSize: number;
	dropout: DropoutConfig;
	validation: ValidationConfig;
	sharedObjectAllocation: boolean;
	cachingEnabled: boolean;
	inplaceSupport: boolean;
}

export interface TransformerAttentionConfig {
	present: boolean;
	nKeyValueHeads: number;
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

export interface PositionEncodingConfig {
	present: boolean;
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
}

export type Activation = 'relu' | 'relu2' | 'gelu' | 'silu' | 'sigmoid' | 'swiglu' | 'tanh';

export interface MLPConfig {
	present: boolean;
	activation: Activation;
	hiddenExpansionFactor: number;
}

export type ModelType = 'decoder' | 'encoder' | 'encoder-decoder';

export interface TransformerConfig {
	headDim: number;
	attention: TransformerAttentionConfig;
	positionalEncoding: PositionEncodingConfig;
	mlp: MLPConfig;
}

export interface ModelConfig {
	topology: ModelType;
	layers: number;
	encoderDecoder: {
		decoderLayers: number;
		encoderLayers: number;
	};
	layerNormalization: LayerNormalizationConfig;
	transformer: TransformerConfig;
}

export interface OptimizerConfig {
	type: 'AdamW' | 'Adam' | 'SGD' | 'Muon';
	lr: number;
	weightDecay: {
		present: boolean;
		value: number;
		useWeightDecayGroups: boolean;
	};
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

export interface Config {
	version: number;
	training: TrainingConfig;
	data: DataConfig;
	model: ModelConfig;
	optimizer: OptimizerConfig;
}
