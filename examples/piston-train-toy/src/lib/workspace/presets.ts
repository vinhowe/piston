import type { Config } from './config';

type DeepPartial<T> =
	// eslint-disable-next-line @typescript-eslint/no-unsafe-function-type
	T extends Function
		? T
		: T extends Array<infer U>
			? Array<DeepPartial<U>>
			: T extends object
				? { [K in keyof T]?: DeepPartial<T[K]> }
				: T;

export type PresetDefinition = {
	label: string;
	// Layers are merged in order on top of root defaults
	layers: Array<DeepPartial<Config>>;
	// If true, this preset is not shown in the preset selector but can be used as a layer
	hidden?: boolean;
};

export const PRESET_DEFINITIONS: Record<string, PresetDefinition> = {
	'transformer-toy-base': {
		label: 'Transformer Toy (base)',
		hidden: true,
		layers: [
			{
				model: {
					family: 'transformer',
					topology: 'encoder-decoder'
				},
				training: {
					validation: {
						batchSize: 6,
						completions: {
							present: true,
							decodingBatchSize: 6,
							amount: 'all'
						}
					}
				},
				optimizer: {
					type: 'Muon',
					lr: 1e-3
				}
			}
		]
	},
	'encoder-toy-base': {
		label: 'Encoder Toy (base)',
		hidden: true,
		layers: [
			{
				model: {
					family: 'transformer',
					topology: 'encoder',
					layers: 2
				},
				training: {
					validation: {
						batchSize: 6,
						completions: {
							present: true,
							decodingBatchSize: 6,
							amount: 'all'
						}
					}
				},
				optimizer: {
					lr: 1e-5,
					lrScheduler: {
						present: false,
						cosineAnnealingSchedule: {
							etaMin: 5e-6
						}
					}
				},
				data: {
					trainOnPrompt: false
				}
			}
		]
	},

	'sort-characters': {
		label: 'Toy: Sort Characters',
		layers: [{ preset: 'transformer-toy-base' }, { data: { dataset: 'sort' } }]
	},
	'reverse-sequence': {
		label: 'Toy: Reverse Sequence',
		layers: [{ preset: 'transformer-toy-base' }, { data: { dataset: 'reverse' } }]
	},
	'two-sum': {
		label: 'Toy: Two Sum',
		layers: [{ preset: 'transformer-toy-base' }, { data: { dataset: 'two-sum' } }]
	},
	'dyck-encoder': {
		label: 'Toy: Dyck (Encoder)',
		layers: [{ preset: 'encoder-toy-base' }, { data: { dataset: 'dyck' } }]
	},

	tinystories: {
		label: 'TinyStories with ~15M parameters',
		layers: [
			{
				preset: 'transformer-toy-base',
				model: {
					family: 'transformer',
					topology: 'decoder',
					layers: 6,
					transformer: {
						headDim: 32,
						attention: {
							nKeyValueHeads: 10
						}
					}
				},
				data: {
					trainOnPrompt: false,
					natural: {
						vocabSize: 8192
					}
				},
				optimizer: {
					lr: 1e-4,
					lrScheduler: {
						type: 'cosine',
						cosineAnnealingSchedule: {
							etaMin: 1e-5
						}
					}
				},
				training: {
					logSteps: 20,
					validation: {
						batchSize: 16,
						valSteps: 500,
						completions: {
							present: true,
							decodingBatchSize: 1,
							amount: 'subset',
							subsetSize: 1
						}
					},
					// This is based on how long it takes to train the model on my own machine
					checkpointEverySteps: {
						present: true,
						value: 40
					}
				}
			},
			{
				preset: 'tinystories',
				data: {
					dataset: 'tinystories'
				}
			}
		]
	},

	fineweb: {
		label: 'FineWeb with ~GPT-2 sized model ⚠️',
		layers: [
			{
				preset: 'fineweb',
				data: {
					dataset: 'fineweb',
					natural: {
						vocabSize: 32_768,
						contextSize: 64
					}
				},
				model: {
					family: 'transformer',
					topology: 'decoder',
					layers: 12,
					transformer: {
						headDim: 64,
						attention: {
							nKeyValueHeads: 12,
							gating: {
								present: false
							}
						},
						mlp: {
							present: true,
							activation: 'gelu',
							hiddenExpansionFactor: 4,
							variant: 'standard'
						},
						normalization: {
							qkNorm: {
								present: false
							}
						},
						initialization: {
							present: true,
							std: 0.02,
							projections: {
								attention: {
									strategy: 'layer-scaled'
								},
								mlp: {
									strategy: 'layer-scaled'
								},
								lmHead: {
									strategy: 'layer-scaled'
								}
							}
						}
					},
					layerNormalization: {
						type: 'layernorm',
						eps: 1e-5,
						transformer: {
							present: true,
							position: 'pre'
						}
					}
				},
				optimizer: {
					lr: 1e-4,
					lrScheduler: {
						type: 'cosine',
						cosineAnnealingSchedule: {
							etaMin: 1e-5
						}
					}
				},
				training: {
					enableVisualization: false,
					logSteps: 20,
					batchSize: 32,
					dropout: {
						present: true,
						embedding: 0.1,
						transformer: {
							attention: 0.1,
							residual: 0.1
						}
					},
					vramLimitMb: { present: true, value: 32_768 },
					validation: {
						valSteps: 500,
						batchSize: 16,
						temperature: 0.0,
						completions: {
							present: false
						}
					},
					// This is purely theoretical: I have no idea how long it would take to train this model
					// because I've never managed to get it training on my own machine.
					checkpointEverySteps: {
						present: true,
						value: 40
					}
				}
			}
		]
	}
};

export function getPresetOptions(): Array<{ value: string; text: string }> {
	return Object.entries(PRESET_DEFINITIONS)
		.filter(([, def]) => !def.hidden)
		.map(([id, def]) => ({ value: id, text: def.label }));
}

function expandPresetLayers(
	presetId: string,
	visiting: Set<string> = new Set()
): Array<DeepPartial<Config>> {
	const def = PRESET_DEFINITIONS[presetId];
	if (!def) return [];
	if (visiting.has(presetId)) return [];
	visiting.add(presetId);

	const expanded = [];
	for (const layer of def.layers) {
		const maybePreset = layer.preset;
		if (typeof maybePreset === 'string' && PRESET_DEFINITIONS[maybePreset]) {
			if (!visiting.has(maybePreset)) {
				expanded.push(...expandPresetLayers(maybePreset, visiting));
			}
			const { preset: _omit, ...rest } = layer;
			if (Object.keys(rest).length > 0) {
				expanded.push(rest);
			}
		} else {
			expanded.push(layer);
		}
	}
	visiting.delete(presetId);
	return expanded;
}

export function getPresetLayers(presetId: string | null | undefined): Array<DeepPartial<Config>> {
	if (!presetId) return [];
	const def = PRESET_DEFINITIONS[presetId];
	return def ? expandPresetLayers(presetId) : [];
}
