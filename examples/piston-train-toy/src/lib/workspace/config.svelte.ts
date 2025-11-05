import type { Config, ModelType } from '$lib/workspace/config';

import { buildDataset, DATASET_CONFIG_METADATA } from '$lib/train/data';
import { getCollatedSampleData } from '$lib/train/data/collate';
import { TOY_DATASET_CONFIG_DEFAULTS } from '$lib/train/data/toy/config';
import { calculateBlockSize, calculateVocabSize, createDataloader } from '$lib/train/utils/model';
import { seededRandom } from '$lib/train/utils/random';
import { SvelteURL, SvelteURLSearchParams } from 'svelte/reactivity';

import { getPresetLayers } from './presets';
import { getCurrentRun, getLatestRun } from './runs.svelte';
import { getVisualizationExampleOptions } from './visualizationExamples';

export const MODEL_TYPES = [
	'decoder',
	'encoder',
	'encoder-decoder'
] as const satisfies readonly ModelType[];

const CONFIG_DEFAULTS: Config = {
	preset: null,
	training: {
		logSteps: 5,
		batchSize: 32,
		validation: {
			present: true,
			valSteps: 100,
			batchSize: 8,
			temperature: 0.0,
			useKvCache: false,
			completions: {
				present: true,
				decodingBatchSize: 1,
				amount: 'subset',
				subsetSize: 4
			}
		},
		limitTraining: {
			present: false,
			steps: 50_000
		},
		labelSmoothing: {
			present: false,
			value: 1e-4
		},
		dropout: {
			present: false,
			embedding: 0.1,
			transformer: {
				attention: 0.1,
				residual: 0.1
			},
			rnn: {
				interLayer: 0.1
			}
		},
		randomSeed: {
			present: true,
			value: 'sequence toy'
		},
		gradNorm: {
			track: true,
			errorIfNonfinite: true
		},
		clipGradNorm: {
			present: false,
			value: 1.0
		},
		useWeakTensorReferences: true,
		sharedObjectAllocation: false,
		cachingEnabled: false,
		inplaceSupport: true,
		enableVisualization: true,
		vramLimitMb: {
			present: true,
			value: 4096
		},
		checkpointEverySteps: {
			present: true,
			value: 200
		},
		restartEverySteps: 1000
	},
	data: {
		dataset: 'sort',
		trainOnPrompt: false,
		maskRatio: 0.15,
		specialTokens: {
			includeEos: true
		},
		datasets: TOY_DATASET_CONFIG_DEFAULTS,
		natural: {
			contextSize: 32,
			vocabSize: 1024
		}
	},
	model: {
		family: 'transformer',
		topology: 'decoder',
		layers: 1,
		tieEmbeddingsAndLmHead: false,
		roundVocabSizeToNearestMultiple: {
			present: false,
			value: 64
		},
		encoderDecoder: {
			encoderLayers: 1,
			decoderLayers: 1
		},
		layerNormalization: {
			type: 'rmsnorm',
			eps: 1e-5,
			transformer: {
				present: true,
				position: 'pre'
			},
			rnn: {
				withinCell: true,
				betweenLayers: false
			}
		},
		transformer: {
			headDim: 16,
			initialization: {
				present: true,
				std: 0.02,
				projections: {
					attention: {
						present: true,
						strategy: 'zero'
					},
					mlp: {
						present: true,
						strategy: 'zero'
					},
					lmHead: {
						present: true,
						strategy: 'layer-scaled'
					}
				}
			},
			attention: {
				present: true,
				nKeyValueHeads: 4,
				groupedQueryAttention: {
					present: false,
					queryHeadsPerKeyValueHead: 2
				},
				gating: {
					present: true,
					activation: 'sigmoid',
					sites: {
						afterSdpaOutput: true,
						afterValueProjection: false,
						afterKeyProjection: false,
						afterQueryProjection: false,
						afterFinalOutputProjection: false
					}
				},
				sinks: {
					present: false
				}
			},
			positionalEncoding: {
				present: true,
				type: 'learned',
				alibi: {
					maxBias: 8.0
				},
				rope: {
					base: 10000.0
				}
			},
			normalization: {
				qkNorm: {
					present: true,
					type: 'rmsnorm',
					eps: 1e-5
				},
				softcap: {
					attention: {
						present: false,
						value: 30
					},
					logits: {
						present: false,
						value: 30
					}
				}
			},
			mlp: {
				present: true,
				activation: 'relu2',
				hiddenExpansionFactor: 4,
				variant: 'gated'
			}
		},
		rnn: {
			cellType: 'lstm',
			embedding: {
				type: 'learned',
				learned: {
					size: 16
				}
			},
			separateHiddenSize: {
				present: false,
				value: 16
			},
			initialization: {
				present: true,
				xavierInputColumns: {
					present: true,
					distribution: 'uniform'
				},
				orthogonalRecurrentColumns: true,
				perGateOrthogonalBlocks: true,
				zeroBiases: true,
				gru: {
					updateGateBias: {
						present: false,
						value: 0.0
					}
				},
				lstm: {
					forgetGateBias: {
						present: true,
						value: 1.0
					}
				}
			},
			hiddenStateProjection: {
				present: false,
				size: 16
			},
			encoder: {
				bidirectional: false
			},
			encoderDecoderAttention: {
				present: false,
				type: 'additive',
				inputFeedingProjection: true,
				multiplicative: {
					scaleByInverseSqrtHiddenSize: true
				}
			}
		}
	},
	optimizer: {
		type: 'Muon',
		lr: 1e-3,
		weightDecay: {
			present: true,
			value: 1e-2,
			useWeightDecayGroups: true
		},
		warmupSteps: {
			present: true,
			value: 100
		},
		lrScheduler: {
			present: true,
			type: 'cosine',
			stepSchedule: {
				stepSize: 100,
				gamma: 0.8
			},
			constantSchedule: {
				factor: 1 / 3,
				totalIters: 100
			},
			cosineAnnealingSchedule: {
				tMax: 500,
				etaMin: 1e-4
			},
			exponentialSchedule: {
				gamma: 0.999
			},
			linearSchedule: {
				startFactor: 1.0,
				endFactor: 1 / 3,
				totalIters: 1000
			}
		},
		adam: {
			beta1: 0.9,
			beta2: 0.999,
			eps: 1e-8,
			amsgrad: false
		},
		sgd: {
			momentum: 0.9,
			dampening: 0,
			nesterov: false
		},
		muon: {
			momentum: 0.95,
			nsSteps: 5,
			nesterov: true
		}
	},
	visualization: {
		script: null,
		example: 'attention-activations',
		target: 'validation',
		selectedValidation: {
			exampleIndex: 0,
			tokenIndex: 0
		}
	},
	version: 1
};

function computeEffectiveDefaults(presetId: string | null | undefined): Config {
	// Start from root defaults
	const base = JSON.parse(JSON.stringify(CONFIG_DEFAULTS)) as Config;
	if (presetId) {
		const layers = getPresetLayers(presetId);
		for (const layer of layers) {
			mergeDeep(
				base as unknown as Record<string, unknown>,
				layer as unknown as Record<string, unknown>
			);
		}
		base.preset = presetId;
	}
	return base;
}

/**
 * Parses a value based on the type of the default value in the config. This is not wildly general,
 * but it seems to work for the current config.
 * @param valueStr - The value to parse.
 * @param defaultValue - The default value.
 * @returns The parsed value.
 */
function parseValueBasedOnDefault(valueStr: string, defaultValue: unknown): unknown {
	if (typeof defaultValue === 'boolean') {
		return valueStr.toLowerCase() === 'true';
	}
	if (typeof defaultValue === 'number') {
		const num = parseFloat(valueStr);
		return isNaN(num) ? defaultValue : num;
	}
	return valueStr; // Default to string if type is not boolean or number
}

/**
 * Builds a config from URL search params.
 * @param params - The URL search params.
 * @param defaults - The defaults to use if no URL search params are present.
 * @returns The config.
 */
function buildConfigFromUrlParams(params: URLSearchParams, defaults: Config): Partial<Config> {
	const configFromUrl: Record<string, unknown> = {};

	for (const [path, valueStr] of params) {
		const keys = path.split('.');
		let currentLevel = configFromUrl;
		let currentDefaultsLevel: unknown = defaults;

		try {
			for (let i = 0; i < keys.length; i++) {
				const key = keys[i];
				if (
					currentDefaultsLevel === undefined ||
					typeof currentDefaultsLevel !== 'object' ||
					currentDefaultsLevel === null
				) {
					throw new Error(`Invalid config path from URL: ${path}`);
				}
				currentDefaultsLevel = (currentDefaultsLevel as Record<string, unknown>)[key];

				if (i < keys.length - 1) {
					if (
						!(currentLevel as Record<string, unknown>)[key] ||
						typeof (currentLevel as Record<string, unknown>)[key] !== 'object'
					) {
						(currentLevel as Record<string, unknown>)[key] = {};
					}
					currentLevel = (currentLevel as Record<string, unknown>)[key] as Record<string, unknown>;
				} else {
					(currentLevel as Record<string, unknown>)[key] = parseValueBasedOnDefault(
						valueStr,
						currentDefaultsLevel
					);
				}
			}
		} catch (e) {
			console.warn((e as Error).message);
			continue; // Skip this parameter if path is invalid or type mismatch
		}
	}
	return configFromUrl as Partial<Config>;
}

function mergeDeep(target: Record<string, unknown>, source: Record<string, unknown>) {
	for (const key in source) {
		if (Object.prototype.hasOwnProperty.call(source, key)) {
			const sourceVal = source[key];
			let targetKeyAsObject = target[key] as Record<string, unknown>;

			if (sourceVal && typeof sourceVal === 'object' && !Array.isArray(sourceVal)) {
				if (
					!targetKeyAsObject ||
					typeof targetKeyAsObject !== 'object' ||
					Array.isArray(targetKeyAsObject)
				) {
					targetKeyAsObject = {};
					target[key] = targetKeyAsObject;
				}
				mergeDeep(targetKeyAsObject, sourceVal as Record<string, unknown>);
			} else if (sourceVal !== undefined) {
				target[key] = sourceVal;
			}
		}
	}
}

/**
 * Gets the initial config from the URL search params, or the defaults if no URL search params are
 * present.
 * @returns The initial config.
 */
function getInitialConfig(): Config {
	// Start with effective defaults, possibly from URL 'preset'
	let base: Config = JSON.parse(JSON.stringify(CONFIG_DEFAULTS));
	if (typeof window !== 'undefined' && window.location && window.URLSearchParams) {
		try {
			const params = new URLSearchParams(window.location.search);
			const presetFromUrl = params.get('preset');
			base = computeEffectiveDefaults(presetFromUrl);
			const configOverrides = buildConfigFromUrlParams(params, base);
			const initial = JSON.parse(JSON.stringify(base));
			mergeDeep(initial, configOverrides);
			return initial;
		} catch (e) {
			console.error('Error processing config from URL, using defaults:', e);
			return JSON.parse(JSON.stringify(CONFIG_DEFAULTS));
		}
	}
	return base;
}

export const config = $state(getInitialConfig());
const configDefaults = $derived(computeEffectiveDefaults(config.preset));

/**
 * Resets one or more config values to their defaults using dot-separated paths.
 */
export function resetConfigToDefaults(paths: string | string[]) {
	const pathList = Array.isArray(paths) ? paths : [paths];

	for (const path of pathList) {
		const defaultValue = getValueAtPath(configDefaults as unknown as Record<string, unknown>, path);
		if (defaultValue === undefined) {
			console.warn(`resetConfigToDefaults: Unknown config path "${path}"`);
			continue;
		}
		// Deep clone to avoid mutating the CONFIG_DEFAULTS reference
		const cloned = deepClone(defaultValue);
		const ok = setValueAtPath(config as unknown as Record<string, unknown>, path, cloned);
		if (!ok) {
			console.warn(`resetConfigToDefaults: Failed to set value for path "${path}"`);
		}
	}
}

function deepClone<T>(value: T): T {
	return JSON.parse(JSON.stringify(value)) as T;
}

export function getConfigDefaultValue(path: string): unknown {
	const val = getValueAtPath(configDefaults as unknown as Record<string, unknown>, path);
	return deepClone(val);
}

export function equalsConfigDefault(path: string): boolean {
	const current = getValueAtPath(config as unknown as Record<string, unknown>, path);
	const def = getValueAtPath(configDefaults as unknown as Record<string, unknown>, path);
	return valuesDeepEqual(current, def);
}

function valuesDeepEqual(a: unknown, b: unknown): boolean {
	try {
		return JSON.stringify(a) === JSON.stringify(b);
	} catch {
		return a === b;
	}
}

function getValueAtPath(obj: Record<string, unknown>, path: string): unknown {
	const keys = path.split('.');
	let current: unknown = obj;
	for (const key of keys) {
		if (
			current === null ||
			current === undefined ||
			typeof current !== 'object' ||
			!(key in (current as Record<string, unknown>))
		) {
			return undefined;
		}
		current = (current as Record<string, unknown>)[key];
	}
	return current;
}

function setValueAtPath(target: Record<string, unknown>, path: string, value: unknown): boolean {
	const keys = path.split('.');
	let current: Record<string, unknown> = target;
	for (let i = 0; i < keys.length - 1; i++) {
		const key = keys[i];
		const next = current[key];
		if (next === undefined || next === null || typeof next !== 'object' || Array.isArray(next)) {
			// Only create an object if we are not overwriting a non-object path
			current[key] = {};
		}
		current = current[key] as Record<string, unknown>;
	}
	const lastKey = keys[keys.length - 1];
	current[lastKey] = value as unknown;
	return true;
}

/**
 * Flattens only the non-default values from an object by comparing against a defaults object.
 * Returns a map of dot-separated paths to stringified values.
 */
function flattenNonDefault(
	obj: Record<string, unknown>,
	defaults: Record<string, unknown>,
	prefix: string = ''
): Record<string, string> {
	const params: Record<string, string> = {};
	for (const key in obj) {
		if (!Object.prototype.hasOwnProperty.call(obj, key)) continue;
		const newPrefix = prefix ? `${prefix}.${key}` : key;
		const value = obj[key];
		const defaultValue = (defaults ?? {})[key];

		if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
			const defaultChild =
				defaultValue !== null && typeof defaultValue === 'object' && !Array.isArray(defaultValue)
					? (defaultValue as Record<string, unknown>)
					: ({} as Record<string, unknown>);
			const nested = flattenNonDefault(value as Record<string, unknown>, defaultChild, newPrefix);
			Object.assign(params, nested);
		} else if (value !== undefined) {
			if (defaultValue === undefined || !valuesDeepEqual(value, defaultValue)) {
				params[newPrefix] = String(value);
			}
		}
	}
	return params;
}

export function initSharedConfigUrlSync() {
	if (typeof window !== 'undefined' && window.history && window.URL) {
		$effect(() => {
			const configSnapshot = $state.snapshot(config);
			const flatParams = flattenNonDefault(
				configSnapshot,
				configDefaults as unknown as Record<string, unknown>
			);
			// Always include preset when set, so shared URLs preserve selection
			if (configSnapshot.preset) flatParams['preset'] = String(configSnapshot.preset);
			// If any parameters are present, also include the current config version
			if (Object.keys(flatParams).length > 0) {
				flatParams['version'] = String(configSnapshot.version);
			}
			const searchParamsString = new SvelteURLSearchParams(flatParams).toString();

			const currentUrl = new SvelteURL(window.location.href);
			currentUrl.search = searchParamsString; // This replaces the entire search string

			// Only call replaceState if the URL actually changed to avoid flooding history
			if (window.location.href !== currentUrl.href) {
				window.history.replaceState({}, '', currentUrl.toString());
			}
		});
	}
}

export function setPreset(presetId: string) {
	if (config.preset === presetId) return;
	const next = computeEffectiveDefaults(presetId);
	// Replace config fields in-place because we've made config a const and I don't want to figure
	// that out right now.
	config['preset'] = next.preset;
	config['training'] = deepClone(next.training);
	config['data'] = deepClone(next.data);
	config['model'] = deepClone(next.model);
	config['optimizer'] = deepClone(next.optimizer);
	config['visualization'] = deepClone(next.visualization);
	config['version'] = next.version;
	validateConfig();
}

export function replaceConfig(next: Config) {
	config['preset'] = next.preset;
	config['training'] = deepClone(next.training);
	config['data'] = deepClone(next.data);
	config['model'] = deepClone(next.model);
	config['optimizer'] = deepClone(next.optimizer);
	config['visualization'] = deepClone(next.visualization);
	config['version'] = next.version;
	validateConfig();
}

function ensureDatasetSupportsModelType() {
	const datasetKey = config.data.dataset as keyof typeof DATASET_CONFIG_METADATA;
	const meta = DATASET_CONFIG_METADATA[datasetKey];
	const allModels = MODEL_TYPES;
	const supported = [
		...('supportsModelTypes' in meta ? meta.supportsModelTypes : allModels)
	].toSorted();
	const isModelSupported = [...supported].includes(config.model.topology);
	if (!isModelSupported) {
		console.debug(`ensureDatasetSupportsModelType: setting model topology to ${supported[0]}`);
		config.model.topology = supported[0] as ModelType;
	}
}

function datasetFromConfig(config: Config) {
	ensureDatasetSupportsModelType();
	const generator = seededRandom(0);
	const dataset = buildDataset(config, generator, 'train');
	const [dataloader, collateFn] = createDataloader(config, dataset, generator, null);
	const blockSize = calculateBlockSize(config, dataloader);
	const vocabSize = calculateVocabSize(config, dataset);

	const collatedData = getCollatedSampleData(dataset, collateFn, 4);

	return {
		dataset,
		vocabSize,
		blockSize,
		tokenizer: dataset.tokenizer,
		sampleData: collatedData.then((data) => {
			const firstSample = data.collated[0];
			return {
				hasPrompt: 'prompt' in firstSample && (firstSample.prompt?.length ?? 0) > 0,
				samples: data.samples,
				collated: data.collated
			};
		})
	};
}

const currentDataset = $derived(datasetFromConfig(config));

export function getCurrentDataset() {
	return currentDataset;
}

const currentRunDataset = $derived.by(() => {
	const currentRun = getCurrentRun();
	return currentRun?.config ? datasetFromConfig(currentRun.config) : null;
});

export function getCurrentRunDataset() {
	return currentRunDataset;
}

const latestRunDataset = $derived.by(() => {
	const latestRun = getLatestRun();
	return latestRun?.config ? datasetFromConfig(latestRun.config) : null;
});

export function getLatestRunDataset() {
	return latestRunDataset;
}

const rnnHiddenSize = $derived.by(() => {
	const { rnn } = config.model;
	if (rnn.hiddenStateProjection.present) {
		return rnn.hiddenStateProjection.size;
	}
	if (rnn.separateHiddenSize.present) {
		return rnn.separateHiddenSize.value;
	}
	if (rnn.embedding.type === 'learned') {
		return rnn.embedding.learned.size;
	}
	return currentDataset.vocabSize;
});

const transformerHiddenSize = $derived(
	config.model.transformer.headDim *
		(config.model.transformer.attention.groupedQueryAttention.present
			? config.model.transformer.attention.groupedQueryAttention.queryHeadsPerKeyValueHead
			: 1) *
		config.model.transformer.attention.nKeyValueHeads
);

const mlpIntermediateSize = $derived(
	transformerHiddenSize * config.model.transformer.mlp.hiddenExpansionFactor
);

export function getHiddenSize() {
	if (config.model.family === 'rnn') {
		return rnnHiddenSize;
	}
	return transformerHiddenSize;
}

export function getMlpIntermediateSize() {
	return mlpIntermediateSize;
}

export function validateConfig() {
	// There are a few things that can still slip through the cracks, so we deal with those here.

	if (
		config.visualization.selectedValidation.exampleIndex >= config.training.validation.batchSize
	) {
		config.visualization.selectedValidation.exampleIndex = 0;
	}

	if (
		config.visualization.example !== 'custom' &&
		!getVisualizationExampleOptions(config).some((e) => e.value === config.visualization.example)
	) {
		config.visualization.example = 'all-activations';
	}

	if (
		config.training.validation.completions.present &&
		config.training.validation.completions.amount === 'subset' &&
		config.training.validation.completions.subsetSize > config.training.validation.batchSize
	) {
		config.training.validation.completions.subsetSize = config.training.validation.batchSize;
	}

	ensureDatasetSupportsModelType();
}
