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

import { ADDITION_SHORT_DESCRIPTIONS } from '$lib/train/data/toy/addition';
import { COPY_MEMORY_SHORT_DESCRIPTIONS } from '$lib/train/data/toy/copyMemory';
import { DYCK_SHORT_DESCRIPTIONS } from '$lib/train/data/toy/dyck';
import { MARKED_ADDITION_SHORT_DESCRIPTIONS } from '$lib/train/data/toy/markedAddition';
import { MODULAR_ADDITION_SHORT_DESCRIPTIONS } from '$lib/train/data/toy/modularAddition';
import { PARITY_SHORT_DESCRIPTIONS } from '$lib/train/data/toy/parity';
import { RANDOM_SHORT_DESCRIPTIONS } from '$lib/train/data/toy/random';
import { REPEAT_SHORT_DESCRIPTIONS } from '$lib/train/data/toy/repeat';
import { REVERSE_SHORT_DESCRIPTIONS } from '$lib/train/data/toy/reverse';
import { SLAPJACK_SHORT_DESCRIPTIONS } from '$lib/train/data/toy/slapjack';
import { SORT_SHORT_DESCRIPTIONS } from '$lib/train/data/toy/sort';
import { TEMPORAL_ORDER_SHORT_DESCRIPTIONS } from '$lib/train/data/toy/temporalOrder';
import { TWO_SUM_SHORT_DESCRIPTIONS } from '$lib/train/data/toy/twoSum';
import { ZEROS_SHORT_DESCRIPTIONS } from '$lib/train/data/toy/zeros';

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
	checkpointEverySteps: {
		present: boolean;
		value: number;
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
	// No default preset; shown as a top-level selector only
	preset: null,
	training: {
		logSteps: 'log steps',
		checkpointEverySteps: 'checkpoint steps',
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
			},
			rnn: {
				interLayer: 'dropout rnn'
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
		enableVisualization: 'viz',
		vramLimitMb: {
			present: 'vram lim',
			value: 'vram lim'
		},
		restartEverySteps: 'restart steps'
	},
	data: {
		dataset: 'dataset',
		trainOnPrompt: 'train prompt',
		maskRatio: 'mask ratio',
		specialTokens: {
			includeEos: 'eos'
		},
		datasets: {
			addition: ADDITION_SHORT_DESCRIPTIONS,
			'copy-memory': COPY_MEMORY_SHORT_DESCRIPTIONS,
			dyck: DYCK_SHORT_DESCRIPTIONS,
			elman: {},
			'marked-addition': MARKED_ADDITION_SHORT_DESCRIPTIONS,
			'modular-addition': MODULAR_ADDITION_SHORT_DESCRIPTIONS,
			parity: PARITY_SHORT_DESCRIPTIONS,
			random: RANDOM_SHORT_DESCRIPTIONS,
			repeat: REPEAT_SHORT_DESCRIPTIONS,
			reverse: REVERSE_SHORT_DESCRIPTIONS,
			slapjack: SLAPJACK_SHORT_DESCRIPTIONS,
			sort: SORT_SHORT_DESCRIPTIONS,
			'temporal-order': TEMPORAL_ORDER_SHORT_DESCRIPTIONS,
			'two-sum': TWO_SUM_SHORT_DESCRIPTIONS,
			zeros: ZEROS_SHORT_DESCRIPTIONS
		},
		// datasets: TOY_DATASET_CONFIG_DEFAULTS,
		natural: {
			contextSize: 'ctx',
			vocabSize: 'vocab'
		}
	},
	model: {
		family: 'family',
		topology: 'topology',
		layers: 'layers',
		tieEmbeddingsAndLmHead: 'tie emb + lm head',
		roundVocabSizeToNearestMultiple: {
			present: 'vocab mult',
			value: 'vocab mult'
		},
		encoderDecoder: {
			encoderLayers: 'n enc',
			decoderLayers: 'n dec'
		},
		layerNormalization: {
			type: 'ln',
			eps: 'eps',
			transformer: {
				present: 'ln',
				position: 'ln pos'
			},
			rnn: {
				withinCell: 'ln in cell',
				betweenLayers: 'ln between'
			}
		},
		transformer: {
			headDim: 'head dim',
			initialization: {
				present: 'init',
				std: 'init std',
				projections: {
					attention: {
						present: 'init attn',
						strategy: 'init attn'
					},
					mlp: {
						present: 'init mlp',
						strategy: 'init mlp'
					},
					lmHead: {
						present: 'init lmhead',
						strategy: 'init lmhead'
					}
				}
			},
			attention: {
				present: 'attn',
				nKeyValueHeads: 'n attn kv',
				groupedQueryAttention: {
					present: 'gqa',
					queryHeadsPerKeyValueHead: 'n attn q/kv'
				},
				gating: {
					present: 'attn gate',
					activation: 'attn gate act',
					sites: {
						afterSdpaOutput: 'gate@sdpa',
						afterValueProjection: 'gate@value',
						afterKeyProjection: 'gate@key',
						afterQueryProjection: 'gate@query',
						afterFinalOutputProjection: 'gate@proj'
					}
				},
				sinks: {
					present: 'attn sinks'
				}
			},
			positionalEncoding: {
				present: 'pos enc',
				type: 'pos type',
				alibi: {
					maxBias: 'alibi max bias'
				},
				rope: {
					base: 'rope base'
				}
			},
			normalization: {
				qkNorm: {
					present: 'qk norm',
					type: 'qk norm type',
					eps: 'qk norm eps'
				},
				softcap: {
					attention: {
						present: 'softcap attn',
						value: 'softcap attn'
					},
					logits: {
						present: 'softcap logits',
						value: 'softcap logits'
					}
				}
			},
			mlp: {
				present: 'mlp',
				activation: 'mlp act',
				hiddenExpansionFactor: 'mlp factor',
				variant: 'mlp mode'
			}
		},
		rnn: {
			cellType: 'cell',
			embedding: {
				type: 'cell emb',
				learned: {
					size: 'cell emb'
				}
			},
			separateHiddenSize: {
				present: 'cell hid.',
				value: 'cell hid.'
			},
			initialization: {
				present: 'init',
				xavierInputColumns: {
					present: 'xavier in. cols',
					distribution: 'xavier in. dist'
				},
				orthogonalRecurrentColumns: 'ortho cols',
				perGateOrthogonalBlocks: 'ortho blocks',
				zeroBiases: 'zero biases',
				gru: {
					updateGateBias: {
						present: 'gru update gate bias',
						value: 'gru update gate bias'
					}
				},
				lstm: {
					forgetGateBias: {
						present: 'forget gate bias',
						value: 'forget gate bias'
					}
				}
			},
			hiddenStateProjection: {
				present: 'hid. proj',
				size: 'hid. proj'
			},
			encoder: {
				bidirectional: 'enc bidir'
			},
			encoderDecoderAttention: {
				present: 'enc-dec attn',
				type: 'enc-dec attn',
				inputFeedingProjection: 'enc-dec proj',
				multiplicative: {
					scaleByInverseSqrtHiddenSize: 'x 1/sqrt(hid.)'
				}
			}
		}
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
	visualization: {
		script: null,
		example: null,
		target: null,
		selectedValidation: {
			exampleIndex: null,
			tokenIndex: null
		}
	},
	version: null
};
