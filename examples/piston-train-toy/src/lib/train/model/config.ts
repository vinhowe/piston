import type {
	Config,
	DropoutConfig,
	InitializationConfig,
	LayerNormalizationConfig,
	MLPConfig,
	PositionEncodingConfig,
	TransformerAttentionConfig
} from '$lib/workspace/config';

export interface TransformerModuleConfig {
	vocabSize: number;
	embeddingSize: number;
	attention: TransformerAttentionConfig;
	layerNormalization: LayerNormalizationConfig;
	mlp: MLPConfig;
	positionalEncoding: PositionEncodingConfig;
	dropout: DropoutConfig;
	initialization: InitializationConfig;
}

export function buildTransformerConfigCommon(
	config: Config,
	vocabSize: number
): TransformerModuleConfig {
	const effectiveHeads = config.model.transformer.attention.present
		? config.model.transformer.attention.nKeyValueHeads *
			(config.model.transformer.attention.groupedQueryAttention.present
				? config.model.transformer.attention.groupedQueryAttention.queryHeadsPerKeyValueHead
				: 1)
		: 1;

	return {
		vocabSize: vocabSize,
		embeddingSize: config.model.transformer.headDim * effectiveHeads,
		attention: config.model.transformer.attention,
		mlp: config.model.transformer.mlp,
		positionalEncoding: config.model.transformer.positionalEncoding,
		layerNormalization: config.model.layerNormalization,
		initialization: config.model.transformer.initialization,
		dropout: (({ present, embedding: embd, transformer }: DropoutConfig) => {
			return {
				present,
				embedding: present ? embd : 0,
				transformer: {
					attention: present ? transformer.attention : 0,
					residual: present ? transformer.residual : 0
				}
			};
		})(config.training.dropout)
	};
}
