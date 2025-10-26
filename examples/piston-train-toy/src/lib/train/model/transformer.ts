/**
 * @fileoverview Implementation of generic encoder-decoder, encoder-only, and decoder-only
 * transformer models
 */

import type { Config } from '$lib/workspace/config';
import type { CrossEntropyLoss, Tensor } from '@piston-ml/piston-web';

import { nn } from '@piston-ml/piston-web';
import * as piston from '@piston-ml/piston-web';

import type { DecoderKVCache } from './cache';

import { MLMHead, Pooler } from './bidirectional';
import { createEmptyDecoderKVCache } from './cache';
import { buildTransformerConfigCommon, type TransformerModuleConfig } from './config';
import { DecoderLayer } from './modules/decoder';
import { EncoderLayer } from './modules/encoder';
import {
	addPositionalEncodingToEmbeddings as addPositionalEncodingToEmbedding,
	buildLmHead,
	computeAutoregressiveCrossEntropyLoss,
	createCrossEntropyCriterion,
	maybeCreateFinalLayerNorm,
	maybeCreateLearnedPositionEmbedding
} from './utils';

export type EncoderDecoderTransformerConfig = TransformerModuleConfig & {
	nEncoderLayer: number;
	nDecoderLayer: number;
	blockSizeSrc: number;
	blockSizeTgt: number;
};

export type EncoderForEncoderDecoderDict = {
	drop?: nn.Dropout;
	wordEmbedding: nn.Embedding;
	positionEmbedding?: nn.Embedding;
	layer: nn.ModuleList;
	layerNorm?: nn.Module<[Tensor], Tensor>;
};

export type DecoderForEncoderDecoderDict = {
	drop?: nn.Dropout;
	wordEmbedding: nn.Embedding;
	positionEmbedding?: nn.Embedding;
	layer: nn.ModuleList;
	layerNorm?: nn.Module<[Tensor], Tensor>;
};

export type EncoderDecoderDict = {
	encoder: nn.ModuleDict<EncoderForEncoderDecoderDict>;
	decoder: nn.ModuleDict<DecoderForEncoderDecoderDict>;
};

/**
 * Encoder-Decoder Transformer model implementation
 */
export type EncoderDecoderForwardOptions = {
	targets?: Tensor | null;
	srcPaddingMask?: Tensor | null;
	tgtPaddingMask?: Tensor | null;
	encoderHiddenStates?: Tensor | null;
	kvCache?: DecoderKVCache | null;
};

export class EncoderDecoderTransformer extends nn.Module {
	public config: EncoderDecoderTransformerConfig;
	readonly lmHead: nn.Linear;
	public encoder: nn.ModuleDict<EncoderForEncoderDecoderDict>;
	public decoder: nn.ModuleDict<DecoderForEncoderDecoderDict>;
	private readonly criterion: CrossEntropyLoss;

	constructor(config: Config, vocabSize: number, blockSizeSrc: number, blockSizeTgt: number) {
		super();

		this.config = {
			...buildTransformerConfigCommon(config, vocabSize),
			nEncoderLayer: config.model.encoderDecoder.encoderLayers,
			nDecoderLayer: config.model.encoderDecoder.decoderLayers,
			blockSizeSrc,
			blockSizeTgt
		};

		const encoderDict: EncoderForEncoderDecoderDict = {
			drop: this.config.dropout.present ? new nn.Dropout(this.config.dropout.embedding) : undefined,
			wordEmbedding: new nn.Embedding(this.config.vocabSize, this.config.embeddingSize),
			layer: new nn.ModuleList(
				Array.from({ length: this.config.nEncoderLayer }).map(() => new EncoderLayer(this.config))
			)
		};

		const decoderDict: DecoderForEncoderDecoderDict = {
			drop: this.config.dropout.present ? new nn.Dropout(this.config.dropout.embedding) : undefined,
			wordEmbedding: new nn.Embedding(this.config.vocabSize, this.config.embeddingSize),
			layer: new nn.ModuleList(
				Array.from({ length: this.config.nDecoderLayer }).map(
					// With cross-attention to consume encoder hidden states
					() => new DecoderLayer(this.config, true)
				)
			)
		};

		// Learned positional embeddings (transformer Embedding variant)
		encoderDict.positionEmbedding = maybeCreateLearnedPositionEmbedding(
			this.config.positionalEncoding,
			this.config.blockSizeSrc,
			this.config.embeddingSize
		);
		decoderDict.positionEmbedding = maybeCreateLearnedPositionEmbedding(
			this.config.positionalEncoding,
			this.config.blockSizeTgt,
			this.config.embeddingSize
		);

		// Final layer norms if configured
		encoderDict.layerNorm = maybeCreateFinalLayerNorm(
			this.config.embeddingSize,
			this.config.layerNormalization
		);
		decoderDict.layerNorm = maybeCreateFinalLayerNorm(
			this.config.embeddingSize,
			this.config.layerNormalization
		);

		this.encoder = new nn.ModuleDict(encoderDict);
		this.decoder = new nn.ModuleDict(decoderDict);

		this.lmHead = buildLmHead(this.config.embeddingSize, this.config.vocabSize);

		this.criterion = createCrossEntropyCriterion();
	}

	forward(
		inputIdsSrc: Tensor | null,
		inputIdsTgt: Tensor,
		options: EncoderDecoderForwardOptions = {}
	): [Tensor, Tensor | null] {
		const targets = options.targets ?? null;
		const srcPaddingMask = options.srcPaddingMask ?? null;
		const tgtPaddingMask = options.tgtPaddingMask ?? null;
		const explicitEncStates = options.encoderHiddenStates ?? null;
		const kvCache = options.kvCache ?? null;
		// Encode source sequence if not provided
		let finalEncoderHiddenStates: Tensor;
		if (explicitEncStates) {
			finalEncoderHiddenStates = explicitEncStates;
		} else {
			if (!inputIdsSrc) {
				throw new Error('Either inputIdsSrc or encoderHiddenStates must be provided');
			}
			finalEncoderHiddenStates = this.encode(inputIdsSrc, { srcPaddingMask });
		}

		// Decode target sequence
		const [_batchSize, tgtSeqLen] = inputIdsTgt.size();

		if (!tgtSeqLen) {
			throw new Error(
				'Target input tensor has no sequence length (did you forget to pass input as batches?)'
			);
		}

		// Get target embeddings
		let targetWordEmbeddings = this.decoder.dict.wordEmbedding.forward(inputIdsTgt);

		// Use cache length (if any) as position offset for absolute encodings during incremental decoding
		const posOffsetDec = kvCache?.layers?.[0]?.self?.length ?? 0;

		targetWordEmbeddings = addPositionalEncodingToEmbedding(
			targetWordEmbeddings,
			this.config.positionalEncoding,
			this.decoder.dict.positionEmbedding,
			this.decoder.dict.drop,
			posOffsetDec
		);

		// Pass through each decoder layer
		let hiddenStates = targetWordEmbeddings;

		const useCache = kvCache !== null;
		const cacheObj = useCache ? kvCache! : createEmptyDecoderKVCache(this.config.nDecoderLayer);
		for (let i = 0; i < this.config.nDecoderLayer; i++) {
			const layerModule = this.decoder.dict.layer[i] as DecoderLayer;
			const result = layerModule.forward(hiddenStates, {
				encoderHiddenStates: finalEncoderHiddenStates,
				srcPaddingMask,
				tgtPaddingMask,
				cache: cacheObj.layers[i]
			});
			if (useCache) {
				cacheObj.layers[i] = result.pastKeyValues!;
				hiddenStates = result.output;
			} else {
				hiddenStates = result.output;
			}
		}

		// Apply final layer normalization
		if (this.decoder.dict.layerNorm) {
			hiddenStates = this.decoder.dict.layerNorm.forward(hiddenStates);
		}

		// Project to vocabulary
		const logits = this.lmHead.forward(hiddenStates);

		const loss = computeAutoregressiveCrossEntropyLoss(logits, targets, this.criterion);

		return [logits, loss];
	}

	/**
	 * Encode source sequence
	 * @param inputIdsSrc - Source input tensor
	 * @param srcPaddingMask - Source padding mask
	 * @returns Encoder hidden states
	 */
	encode(
		inputIdsSrc: Tensor,
		{ srcPaddingMask }: { srcPaddingMask: Tensor | null } = { srcPaddingMask: null }
	): Tensor {
		const [_batchSize, srcSeqLen] = inputIdsSrc.size();

		if (!srcSeqLen) {
			throw new Error(
				'Source input tensor has no sequence length (did you forget to pass input as batches?)'
			);
		}

		// Get source embeddings
		let sourceWordEmbeddings = this.encoder.dict.wordEmbedding.forward(inputIdsSrc);

		sourceWordEmbeddings = addPositionalEncodingToEmbedding(
			sourceWordEmbeddings,
			this.config.positionalEncoding,
			this.encoder.dict.positionEmbedding,
			this.encoder.dict.drop
		);

		// Pass through each encoder layer
		let hiddenStates = sourceWordEmbeddings;

		for (let i = 0; i < this.config.nEncoderLayer; i++) {
			const layerModule = this.encoder.dict.layer[i] as EncoderLayer;
			const layerOutput = layerModule.forward(hiddenStates, srcPaddingMask);
			hiddenStates = layerOutput;
		}

		// Apply final layer normalization
		if (this.encoder.dict.layerNorm) {
			hiddenStates = this.encoder.dict.layerNorm.forward(hiddenStates) as Tensor;
		}

		return hiddenStates;
	}
}

type DecoderTransformerConfig = TransformerModuleConfig & {
	nLayers: number;
	blockSize: number;
};

type DecoderTransformerDict = {
	drop?: nn.Dropout;
	wordEmbedding: nn.Embedding;
	positionEmbedding?: nn.Embedding;
	layer: nn.ModuleList;
	lnF?: nn.Module<[Tensor], Tensor>;
};

export class DecoderTransformer extends nn.Module {
	public config: DecoderTransformerConfig;
	public decoder: nn.ModuleDict<DecoderTransformerDict>;
	readonly lmHead: nn.Linear;
	private readonly criterion: CrossEntropyLoss;

	constructor(config: Config, vocabSize: number, blockSize: number) {
		super();

		this.config = {
			...buildTransformerConfigCommon(config, vocabSize),
			blockSize,
			nLayers: config.model.layers
		};

		const transformerDict: DecoderTransformerDict = {
			drop: this.config.dropout.present ? new nn.Dropout(this.config.dropout.embedding) : undefined,
			wordEmbedding: new nn.Embedding(this.config.vocabSize, this.config.embeddingSize),
			layer: new nn.ModuleList(
				Array.from({ length: this.config.nLayers }).map(() => new DecoderLayer(this.config, false))
			)
		};

		// Only create positional embedding layer for learned positional encoding
		transformerDict.positionEmbedding = maybeCreateLearnedPositionEmbedding(
			this.config.positionalEncoding,
			this.config.blockSize,
			this.config.embeddingSize
		);

		transformerDict.lnF = maybeCreateFinalLayerNorm(
			this.config.embeddingSize,
			this.config.layerNormalization
		);

		this.decoder = new nn.ModuleDict(transformerDict);

		// Output projection with optional weight tying to token embeddings
		this.lmHead = buildLmHead(this.config.embeddingSize, this.config.vocabSize);

		this.criterion = createCrossEntropyCriterion();
	}

	/**
	 * @param input - Input tensor of token IDs [batch_size, seq_len]
	 * @param targets - Target tensor of token IDs [batch_size,
	 * seq_len]
	 * @returns [logits, loss]
	 */
	forward(
		input: Tensor,
		options: { targets?: Tensor | null; kvCache?: DecoderKVCache | null } = {}
	): [Tensor, Tensor | null] {
		const targets = options.targets ?? null;
		const kvCache = options.kvCache ?? null;
		const [_batchSize, seqLen] = input.size();

		if (!seqLen) {
			throw new Error(
				'Input tensor has no sequence length (did you forget to pass input as batches?)'
			);
		}

		// Get token embeddings
		let wordEmbeddings = this.decoder.dict.wordEmbedding.forward(input);

		// Use cache length (if any) as position offset for absolute encodings during incremental decoding
		const posOffset = kvCache?.layers?.[0]?.self?.length ?? 0;

		wordEmbeddings = addPositionalEncodingToEmbedding(
			wordEmbeddings,
			this.config.positionalEncoding,
			this.decoder.dict.positionEmbedding,
			this.decoder.dict.drop,
			posOffset
		);

		// Pass through each transformer layer
		let hiddenStates = wordEmbeddings;

		const useCache = kvCache !== null;
		const cacheObj = useCache ? kvCache! : createEmptyDecoderKVCache(this.config.nLayers);
		for (let i = 0; i < this.config.nLayers; i++) {
			const layerModule = this.decoder.dict.layer[i] as DecoderLayer;
			const result = layerModule.forward(hiddenStates, { cache: cacheObj.layers[i] });
			if (useCache) {
				cacheObj.layers[i] = result.pastKeyValues!;
				hiddenStates = result.output;
			} else {
				hiddenStates = result.output;
			}
		}

		// Apply final layer normalization
		if (this.decoder.dict.lnF) {
			hiddenStates = this.decoder.dict.lnF.forward(hiddenStates) as Tensor;
		}

		// Project to vocabulary
		const logits = this.lmHead.forward(hiddenStates);

		const loss = computeAutoregressiveCrossEntropyLoss(logits, targets, this.criterion);

		return [logits, loss ?? null];
	}
}

export type EncoderTransformerConfig = TransformerModuleConfig & {
	typeVocabSize: number;
	nLayers: number;
	blockSize: number;
	attentionMasking: {
		padMask: boolean;
	};
	pooling: {
		present: boolean;
	};
	mlmHead: {
		present: boolean;
	};
};

type EncoderTransformerDict = {
	dropout?: nn.Dropout;
	wordEmbedding: nn.Embedding;
	positionEmbedding?: nn.Embedding;
	tokenTypeEmbedding: nn.Embedding;
	layer: nn.ModuleList;
	layerNorm?: nn.Module<[Tensor], Tensor>;
};

export class EncoderTransformer extends nn.Module {
	public config: EncoderTransformerConfig;
	public encoder: nn.ModuleDict<EncoderTransformerDict>;
	readonly mlmHead?: MLMHead;
	readonly pooler?: Pooler;
	private readonly criterion: CrossEntropyLoss;

	constructor(config: Config, vocabSize: number, blockSize: number, typeVocabSize: number = 2) {
		super();

		this.config = {
			...buildTransformerConfigCommon(config, vocabSize),
			typeVocabSize: typeVocabSize,
			nLayers: config.model.layers,
			blockSize,
			// We hardcode these for now, as all of our tasks focus on MLM right now.
			attentionMasking: {
				padMask: true
			},
			pooling: {
				present: false
			},
			mlmHead: {
				present: true,
			}
		};

		const encoderDict: EncoderTransformerDict = {
			dropout: this.config.dropout.present
				? new nn.Dropout(this.config.dropout.embedding)
				: undefined,
			wordEmbedding: new nn.Embedding(this.config.vocabSize, this.config.embeddingSize),
			tokenTypeEmbedding: new nn.Embedding(this.config.typeVocabSize, this.config.embeddingSize),
			layer: new nn.ModuleList(
				Array.from({ length: this.config.nLayers }).map(() => new EncoderLayer(this.config))
			)
		};

		// Only create positional embedding layer for learned positional encoding
		if (this.config.positionalEncoding.present) {
			encoderDict.positionEmbedding = new nn.Embedding(
				this.config.blockSize,
				this.config.embeddingSize
			);
		}

		encoderDict.layerNorm = maybeCreateFinalLayerNorm(
			this.config.embeddingSize,
			this.config.layerNormalization
		) as nn.Module<[Tensor], Tensor> | undefined;

		this.encoder = new nn.ModuleDict(encoderDict);

		// MLM head for masked language modeling
		if (this.config.mlmHead.present) {
			this.mlmHead = new MLMHead(
				this.config.embeddingSize,
				this.config.vocabSize,
				this.config.layerNormalization,
			);
		}

		// Pooler for classification tasks
		if (this.config.pooling.present) {
			this.pooler = new Pooler(this.config.embeddingSize);
		}

		this.criterion = createCrossEntropyCriterion();
	}

	/**
	 * @param inputIds - Input tensor of token IDs [batch_size, seq_len]
	 * @param tokenTypeIds - Token type/segment IDs [batch_size, seq_len]
	 * @param attentionMask - Attention mask [batch_size, seq_len] (1 for real tokens, 0 for padding)
	 * @param targets - Target tensor for MLM [batch_size, seq_len] (-100 for non-masked positions)
	 * @returns [lastHiddenState, pooledOutput?, mlmLogits?, loss?]
	 */
	forward(
		inputIds: Tensor,
		{
			tokenTypeIds,
			attentionMask,
			targets
		}: { tokenTypeIds?: Tensor | null; attentionMask?: Tensor | null; targets?: Tensor | null } = {
			tokenTypeIds: null,
			attentionMask: null,
			targets: null
		}
	): [Tensor, Tensor | null, Tensor | null, Tensor | null] {
		const [batchSize, seqLen] = inputIds.size();

		if (!seqLen) {
			throw new Error(
				'Input tensor has no sequence length (did you forget to pass input as batches?)'
			);
		}

		// Get token embeddings
		let wordEmbedding = this.encoder.dict.wordEmbedding.forward(inputIds);

		// Add segment/token type embeddings
		if (tokenTypeIds == null) {
			tokenTypeIds = piston.zeros([batchSize, seqLen], {
				device: inputIds.device,
				dtype: piston.int32
			});
		}
		const typeEmbeddings = this.encoder.dict.tokenTypeEmbedding.forward(tokenTypeIds!);
		wordEmbedding = wordEmbedding.add(typeEmbeddings);

		wordEmbedding = addPositionalEncodingToEmbedding(
			wordEmbedding,
			this.config.positionalEncoding,
			this.encoder.dict.positionEmbedding,
			this.encoder.dict.dropout
		);

		// Pass through each encoder layer
		let hiddenStates = wordEmbedding;

		for (let i = 0; i < this.config.nLayers; i++) {
			const layerModule = this.encoder.dict.layer[i] as EncoderLayer;
			const layerOutput = layerModule.forward(hiddenStates, attentionMask);
			hiddenStates = layerOutput;
		}

		// Apply final layer normalization
		if (this.encoder.dict.layerNorm) {
			hiddenStates = this.encoder.dict.layerNorm.forward(hiddenStates) as Tensor;
		}

		// Pooled output for classification tasks
		let pooledOutput: Tensor | null = null;
		if (this.pooler) {
			pooledOutput = this.pooler.forward(hiddenStates);
		}

		// MLM logits
		let mlmLogits: Tensor | null = null;
		if (this.mlmHead) {
			mlmLogits = this.mlmHead.forward(hiddenStates);
		}

		// Calculate MLM loss if targets are provided
		let loss: Tensor | null = null;
		if (targets && mlmLogits) {
			// Only compute loss on masked positions (where targets != -100)
			const flatLogits = mlmLogits.view([-1, mlmLogits.size(-1)]);
			const flatTargets = targets.view(-1);
			loss = this.criterion.forward(flatLogits, flatTargets);
		}

		return [hiddenStates, pooledOutput, mlmLogits, loss];
	}
}
