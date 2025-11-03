import type {
	Config,
	LayerNormalizationConfig,
	MLPConfig,
	PositionEncodingConfig,
	RNNDropoutConfig,
	RNNEmbeddingConfig,
	RNNHiddenStateProjectionConfig,
	RNNInitializationConfig,
	TransformerAttentionConfig,
	TransformerDropoutConfig,
	TransformerInitializationConfig,
	TransformerNormalizationConfig
} from '$lib/workspace/config';

export interface TransformerModuleConfig {
	vocabSize: number;
	embeddingSize: number;
	attention: TransformerAttentionConfig;
	layerNormalization: LayerNormalizationConfig;
	normalization: TransformerNormalizationConfig;
	mlp: MLPConfig;
	positionalEncoding: PositionEncodingConfig;
	dropout: TransformerDropoutConfig;
	initialization: TransformerInitializationConfig;
}

export interface RNNModuleConfig {
	cellType: 'gru' | 'lstm' | 'rnn';
	embeddingSize: number;
	vocabSize: number;
	hiddenSize: number;
	baseHiddenSize: number;
	projectionSize?: number;
	embedding: RNNEmbeddingConfig;
	dropout: RNNDropoutConfig;
	layerNormalization: LayerNormalizationConfig;
	initialization: RNNInitializationConfig;
	hiddenStateProjection: RNNHiddenStateProjectionConfig;
	tieEmbeddingsAndLmHead: boolean;
}

export function buildRNNConfigCommon(config: Config, vocabSize: number): RNNModuleConfig {
	const effectiveEmbeddingSize =
		config.model.rnn.embedding.type === 'learned'
			? config.model.rnn.embedding.learned.size
			: vocabSize;
	const rawHiddenSize = config.model.rnn.separateHiddenSize.present
		? config.model.rnn.separateHiddenSize.value
		: effectiveEmbeddingSize;
	const projectionSize = config.model.rnn.hiddenStateProjection.present
		? config.model.rnn.hiddenStateProjection.size
		: undefined;
	const baseHiddenSize = projectionSize ?? rawHiddenSize;

	return {
		vocabSize: vocabSize,
		cellType: config.model.rnn.cellType,
		embeddingSize: effectiveEmbeddingSize,
		hiddenSize: rawHiddenSize,
		baseHiddenSize,
		projectionSize,
		embedding: config.model.rnn.embedding,
		layerNormalization: config.model.layerNormalization,
		hiddenStateProjection: config.model.rnn.hiddenStateProjection,
		initialization: config.model.rnn.initialization,
		tieEmbeddingsAndLmHead: config.model.tieEmbeddingsAndLmHead,
		dropout: (({ present, embedding: embd, rnn }: RNNDropoutConfig) => {
			return {
				present,
				embedding: present ? embd : 0,
				rnn: {
					interLayer: present ? rnn.interLayer : 0
				}
			};
		})(config.training.dropout)
	};
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
		normalization: config.model.transformer.normalization,
		initialization: config.model.transformer.initialization,
		dropout: (({ present, embedding: embd, transformer }: TransformerDropoutConfig) => {
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
