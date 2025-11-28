import type { DATASET_CONFIG_DEFAULTS } from '$lib/train/data';
import type {
	ConstantConfig,
	CosineAnnealingConfig,
	ExponentialConfig,
	LinearConfig,
	StepConfig
} from '@piston-ml/piston-web';

export interface TransformerDropoutConfig {
	present: boolean;
	embedding: number;
	transformer: {
		attention: number;
		residual: number;
	};
}

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
	checkpointEverySteps: {
		present: boolean;
		value: number;
	};
	batchSize: number;
	dropout: TransformerDropoutConfig;
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
	restartEverySteps: number;
}

export interface DataConfig {
	dataset: keyof typeof DATASET_CONFIG_DEFAULTS;
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

export type GPT2ModelType = 'distilgpt2' | 'gpt2' | 'gpt2-medium' | 'gpt2-large' | 'gpt2-xl';

export interface ModelConfig {
	type: GPT2ModelType;
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

export interface Config {
	version: number;
	training: TrainingConfig;
	data: DataConfig;
	model: ModelConfig;
	optimizer: OptimizerConfig;
}

export type ConfigItemDescription =
	| {
			shortName: string;
	  }
	| string
	| [string, number]
	| null;

type ReplaceValues<T, V> = T extends object ? { [K in keyof T]: ReplaceValues<T[K], V> } : V;

export type ConfigValues = ReplaceValues<Config, ConfigItemDescription>;

export const CONFIG_DESCRIPTIONS: ConfigValues = {
	training: {
		logSteps: 'log steps',
		batchSize: 'batch',
		clipGradNorm: {
			present: 'clip grad norm',
			value: 'clip grad norm'
		},
		validation: {
			present: 'val',
			valSteps: 'val steps',
			batchSize: 'val size',
			temperature: 'val temp',
			useKvCache: 'val kv cache',
			completions: {
				present: 'completions',
				decodingBatchSize: 'completions batch',
				amount: 'completions amount strategy',
				subsetSize: 'completions subset'
			}
		},
		limitTraining: {
			present: 'limit train',
			steps: 'max steps'
		},
		labelSmoothing: {
			present: 'smoothing',
			value: 'smoothing'
		},
		dropout: {
			present: 'dropout',
			embedding: 'dropout emb',
			transformer: {
				attention: 'dropout attn',
				residual: 'dropout resid'
			}
		},
		randomSeed: {
			present: 'seed',
			value: 'seed'
		},
		gradNorm: {
			track: 'track grad norm',
			errorIfNonfinite: 'error nonfinite'
		},
		useWeakTensorReferences: 'weak tensor refs',
		sharedObjectAllocation: 'shared objs',
		cachingEnabled: 'caching',
		inplaceSupport: 'inplace',
		vramLimitMb: {
			present: 'vram lim',
			value: 'vram lim'
		},
		checkpointEverySteps: {
			present: 'checkpointing',
			value: 'checkpoint steps'
		},
		restartEverySteps: 'restart steps'
	},
	data: {
		dataset: 'dataset'
	},
	model: {
		type: 'model'
	},
	optimizer: {
		type: 'optim',
		lr: 'lr',
		weightDecay: {
			present: 'decay',
			value: 'decay',
			useWeightDecayGroups: 'decay groups'
		},
		warmupSteps: {
			present: 'warmup',
			value: 'warmup steps'
		},
		lrScheduler: {
			present: 'lr sched',
			type: 'lr sched',
			stepSchedule: {
				stepSize: 'sched step',
				gamma: 'sched gamma'
			},
			constantSchedule: {
				factor: 'const lr factor',
				totalIters: 'const lr total'
			},
			cosineAnnealingSchedule: {
				tMax: 'cos lr tmax',
				etaMin: 'cos lr eta min'
			},
			exponentialSchedule: {
				gamma: 'exp lr gamma'
			},
			linearSchedule: {
				startFactor: 'lin lr start',
				endFactor: 'lin lr end',
				totalIters: 'lin lr total'
			}
		},
		adam: {
			beta1: 'adam beta1',
			beta2: 'adam beta2',
			eps: 'adam eps',
			amsgrad: 'adam ams'
		},
		sgd: {
			momentum: 'sgd moment',
			dampening: 'sgd damp',
			nesterov: 'sgd nester'
		},
		muon: {
			momentum: 'muon moment',
			nsSteps: 'muon nssteps',
			nesterov: 'muon nester'
		}
	},
	version: null
};
