/**
 * @fileoverview Implementation of generic encoder-decoder, encoder-only, and decoder-only
 * transformer models
 */

import type { Config } from '$lib/workspace/config';

import { cat, CrossEntropyLoss, nn, Parameter, type Tensor } from '@piston-ml/piston-web';

import type { DecoderLayerCache, SelfAttentionCache } from './cache';
import type { DecoderKVCache } from './cache';

import { createEmptyDecoderKVCache } from './cache';
import { type GPT2Config } from './config';
import { createCausalMask, createPositionIds, maskedFill } from './utils';

export const GPT2_VOCAB_SIZE = 1024;
export const GPT2_BLOCK_SIZE = 1024;

export class CausalSelfAttention extends nn.Module {
	private readonly nHeads: number;
	private readonly nKvHeads: number;
	private readonly embeddingSize: number;
	private readonly headDim: number;
	private readonly config: GPT2Config;

	private readonly c_attn: nn.Linear;
	private readonly c_proj: nn.Linear;

	private attn_dropout?: nn.Dropout;
	private resid_dropout?: nn.Dropout;

	constructor(config: GPT2Config) {
		super();

		this.nHeads = config.nHead;
		this.nKvHeads = config.nHead;
		this.embeddingSize = config.nEmbd;
		this.headDim = config.nEmbd / this.nHeads;
		this.config = config;

		const kvDim = this.headDim * this.nKvHeads;
		const qkvOutDim = this.embeddingSize + 2 * kvDim;

		this.c_attn = new nn.Linear(this.embeddingSize, qkvOutDim);
		this.c_proj = new nn.Linear(this.embeddingSize, this.embeddingSize);

		this.attn_dropout = new nn.Dropout(config.attnPdrop);
		this.resid_dropout = new nn.Dropout(config.residPdrop);
	}

	private projectQkv(input: Tensor): [Tensor, Tensor, Tensor] {
		const [B, T, _] = input.size();
		const kvDim = this.headDim * this.nKvHeads;
		const keyPos = this.embeddingSize;
		const valuePos = this.embeddingSize + kvDim;
		const qkv = this.c_attn.forward(input);
		let q = qkv.slice([
			[0, B],
			[0, T],
			[0, this.embeddingSize]
		]);
		let k = qkv.slice([
			[0, B],
			[0, T],
			[keyPos, keyPos + kvDim]
		]);
		let v = qkv.slice([
			[0, B],
			[0, T],
			[valuePos, valuePos + kvDim]
		]);
		const qShape = [B, T, this.nHeads, this.headDim];
		const kvShape = [B, T, this.nKvHeads, this.headDim];
		q = q.view(qShape)?.transpose(1, 2);
		k = k.view(kvShape)?.transpose(1, 2);
		v = v.view(kvShape)?.transpose(1, 2);
		[k, v] = this.applyGroupedQueryBroadcast(k, v, B, T);
		return [q, k, v];
	}

	private runAttention(
		q: Tensor,
		k: Tensor,
		v: Tensor,
		options: {
			attentionMask?: Tensor | null;
			cache?: SelfAttentionCache | null;
			ropeOffsets?: { qOffset: number; kOffset: number } | null;
		}
	): [Tensor, SelfAttentionCache | null] {
		const B = q.size(0);
		const T_q = q.size(2);

		// Concatenate with cache if present
		const kCat = options.cache ? cat([options.cache.k, k], { dim: 2 }) : k;
		const vCat = options.cache ? cat([options.cache.v, v], { dim: 2 }) : v;
		const T_kv = kCat.size(2);

		// Compute attention scores
		let att = q.matmul(kCat, { transRhs: true }).div(Math.sqrt(this.headDim));

		att = this.applyCausalMask(att, T_q, T_kv);

		if (options.attentionMask) {
			att = this.applyAttentionMask(att, options.attentionMask);
		}

		// Apply softmax over keys plus optional sink column
		att = att.softmax(3);
		att = this.attn_dropout ? this.attn_dropout.forward(att) : att;

		let y = att.matmul(vCat).transpose(1, 2).view([B, T_q, this.embeddingSize]);
		// Apply output projection
		y = this.c_proj.forward(y);

		if (this.resid_dropout) {
			y = this.resid_dropout.forward(y);
		}

		const newCache: SelfAttentionCache | null = {
			k: kCat,
			v: vCat,
			length: T_kv
		};
		return [y, newCache];
	}

	private applyAttentionMask(att: Tensor, attentionMask: Tensor) {
		// Convert attention mask to appropriate shape [B, 1, 1, T_kv] and broadcast
		const mask = attentionMask.unsqueeze(1).unsqueeze(2).broadcastTo(att.size());
		return maskedFill(att, mask, -1e9);
	}

	private applyCausalMask(att: Tensor, queryLen: number, keyLen: number) {
		// Apply causal mask if needed
		return queryLen <= 1
			? att
			: (() => {
					const mask = createCausalMask(queryLen, keyLen).broadcastTo(att.size());
					return maskedFill(att, mask, -1e9);
				})();
	}

	/**
	 * Apply grouped-query attention broadcasting to key and value tensors
	 * @param k Key tensor [B, nKvHeads, seqLen, headDim]
	 * @param v Value tensor [B, nKvHeads, seqLen, headDim]
	 * @param B Batch size
	 * @param seqLen Sequence length
	 * @returns Broadcasted [k, v] tensors [B, nHeads, seqLen, headDim]
	 */
	private applyGroupedQueryBroadcast(
		k: Tensor,
		v: Tensor,
		B: number,
		seqLen: number
	): [Tensor, Tensor] {
		if (this.nHeads !== this.nKvHeads) {
			const repeatFactor = this.nHeads / this.nKvHeads;
			const th = seqLen * this.headDim;

			k = k
				.view([B, this.nKvHeads, th])
				.unsqueeze(2)
				.broadcastTo([B, this.nKvHeads, repeatFactor, th])
				.view([B, this.nHeads, seqLen, this.headDim]);
			v = v
				.view([B, this.nKvHeads, th])
				.unsqueeze(2)
				.broadcastTo([B, this.nKvHeads, repeatFactor, th])
				.view([B, this.nHeads, seqLen, this.headDim]);
		}
		return [k, v];
	}

	forward(
		input: Tensor,
		options: { attentionMask?: Tensor | null; cache?: SelfAttentionCache | null } = {}
	): { output: Tensor; pastKeyValues?: SelfAttentionCache } {
		const qkv = this.projectQkv(input);
		const [q, k, v] = qkv;
		const pastLen = options.cache?.length ?? 0;
		const [y, newCache] = this.runAttention(q, k, v, {
			attentionMask: options.attentionMask ?? null,
			cache: options.cache ?? null,
			ropeOffsets: { qOffset: pastLen, kOffset: pastLen }
		});
		return { output: y, pastKeyValues: newCache ?? undefined };
	}
}

type GPTDict = {
	drop: nn.Dropout;
	wte: nn.Embedding;
	wpe: nn.Embedding;
	h: nn.ModuleList;
	ln_f: nn.Module<[Tensor], Tensor>;
};

export class GPT extends nn.Module {
	public config: GPT2Config;
	public transformer: nn.ModuleDict<GPTDict>;
	readonly lm_head: nn.Linear;
	private readonly criterion: CrossEntropyLoss;

	constructor(modelConfig: GPT2Config, config: Config) {
		super();

		this.config = modelConfig;
		const transformerDict: GPTDict = {
			drop: new nn.Dropout(this.config.embdPdrop),
			wte: new nn.Embedding(this.config.vocabSize, this.config.nEmbd),
			wpe: new nn.Embedding(this.config.blockSize, this.config.nEmbd),
			h: new nn.ModuleList(
				Array.from({ length: this.config.nLayer }).map(() => new Block(this.config))
			),
			ln_f: new nn.LayerNorm(this.config.nEmbd)
		};

		this.transformer = new nn.ModuleDict(transformerDict);

		// Output projection with optional weight tying to token embeddings
		this.lm_head = new nn.Linear(this.config.nEmbd, this.config.vocabSize, false);
		this.lm_head.weight = new Parameter(this.transformer.dict.wte.weight);

		this.criterion = new CrossEntropyLoss({
			labelSmoothing: config.training.labelSmoothing.present
				? config.training.labelSmoothing.value
				: 0.0,
			ignoreIndex: -100
		});
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
		const [batchSize, seqLen] = input.size();

		if (!seqLen) {
			throw new Error(
				'Input tensor has no sequence length (did you forget to pass input as batches?)'
			);
		}

		// Get token embeddings
		let wordEmbeddings = this.transformer.dict.wte.forward(input);

		// Use cache length (if any) as position offset for absolute encodings during incremental decoding
		const posOffset = kvCache?.layers?.[0]?.self?.length ?? 0;

		// Add positional embeddings
		const positions = createPositionIds(seqLen, batchSize, wordEmbeddings.device, posOffset);
		const positionEmbeddingsOutput = this.transformer.dict.wpe.forward(positions);
		wordEmbeddings = wordEmbeddings.add(positionEmbeddingsOutput);
		// Apply embedding dropout if configured
		wordEmbeddings = this.transformer.dict.drop.forward(wordEmbeddings);

		// Pass through each transformer layer
		let hiddenStates = wordEmbeddings;

		const useCache = kvCache !== null;
		const cacheObj = useCache ? kvCache! : createEmptyDecoderKVCache(this.config.nLayer);
		for (let i = 0; i < this.config.nLayer; i++) {
			const layerModule = this.transformer.dict.h[i] as Block;
			const result = layerModule.forward(hiddenStates, { cache: cacheObj.layers[i] });
			if (useCache) {
				cacheObj.layers[i] = result.pastKeyValues!;
				hiddenStates = result.output;
			} else {
				hiddenStates = result.output;
			}
		}

		// Apply final layer normalization
		if (this.transformer.dict.ln_f) {
			hiddenStates = this.transformer.dict.ln_f.forward(hiddenStates);
		}

		// Project to vocabulary
		const logits = this.lm_head.forward(hiddenStates);

		const loss = targets
			? this.criterion.forward(logits.view([-1, logits.size(-1)]), targets.view(-1))
			: null;

		return [logits, loss];
	}
}

export type DecoderLayerForwardOptions = {
	encoderHiddenStates?: Tensor | null;
	srcPaddingMask?: Tensor | null;
	tgtPaddingMask?: Tensor | null;
	cache?: DecoderLayerCache | null;
};

export type DecoderLayerForwardResult = {
	output: Tensor;
	pastKeyValues?: DecoderLayerCache;
};

export class Block extends nn.Module {
	private readonly lnSelfAttn!: nn.Module<[Tensor], Tensor>;
	private readonly lnMlp: nn.Module<[Tensor], Tensor>;
	private readonly selfAttn: CausalSelfAttention;
	private readonly mlp: MLP;
	private readonly dropout?: nn.Dropout;

	constructor(config: GPT2Config) {
		super();

		this.lnSelfAttn = new nn.LayerNorm(config.nEmbd);
		this.selfAttn = new CausalSelfAttention(config);

		this.lnMlp = new nn.LayerNorm(config.nEmbd);
		this.mlp = new MLP(config.nEmbd);

		if (config.residPdrop > 0) {
			this.dropout = new nn.Dropout(config.residPdrop);
		}
	}

	forward(input: Tensor, options: DecoderLayerForwardOptions = {}): DecoderLayerForwardResult {
		const tgtPaddingMask = options.tgtPaddingMask ?? null;
		const cache = options.cache ?? null;
		let x = input;
		let selfCache = cache?.self ?? null;

		const residual = input;
		x = this.lnSelfAttn.forward(input);
		const selfResult = this.selfAttn.forward(x, {
			attentionMask: tgtPaddingMask ?? null,
			cache: selfCache ?? null
		});
		selfCache = selfResult.pastKeyValues ?? null;
		x = residual.add(selfResult.output);

		const residual3 = x;
		x = this.lnMlp.forward(x);
		x = this.mlp.forward(x);
		if (this.dropout) {
			x = this.dropout.forward(x);
		}
		x = residual3.add(x);

		const result: DecoderLayerForwardResult = { output: x };
		if (cache) {
			result.pastKeyValues = { self: selfCache ?? undefined, cross: cache.cross ?? undefined };
		}
		return result;
	}
}

export class MLP extends nn.Module {
	private readonly upProj: nn.Linear;
	private readonly downProj: nn.Linear;
	private readonly activation: (x: Tensor) => Tensor;

	/**
	 * @param embeddingSize - Embedding size
	 */
	constructor(embeddingSize: number) {
		super();

		const intermediateSize = 4 * embeddingSize;
		this.upProj = new nn.Linear(embeddingSize, intermediateSize);
		this.downProj = new nn.Linear(intermediateSize, embeddingSize);

		this.activation = (x: Tensor): Tensor => x.gelu();
	}

	/**
	 * Forward pass through the MLP
	 * @param input - Input tensor
	 * @returns Output tensor
	 */
	forward(input: Tensor): Tensor {
		let h = this.upProj.forward(input);
		h = this.activation(h);
		return this.downProj.forward(h);
	}
}
