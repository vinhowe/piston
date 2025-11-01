import {
	AlibiEmbedding,
	cat,
	gpu,
	Linear,
	nn,
	RotaryEmbedding,
	Tensor,
	zeros
} from '@piston-ml/piston-web';

import type { SelfAttentionCache } from '../cache';
import type { TransformerModuleConfig } from '../config';

import { createCausalMask, maskedFill } from '../utils';
import { SimpleHadamardGate } from './gate';
import { applySoftcap } from './utils';

abstract class CoreAttention extends nn.Module {
	protected readonly nHeads: number;
	protected readonly nKvHeads: number;
	protected readonly embeddingSize: number;
	protected readonly headDim: number;
	protected abstract cProj: Linear;
	protected attnDropout?: nn.Dropout;
	protected residDropout?: nn.Dropout;
	protected rope?: RotaryEmbedding;
	protected alibi?: AlibiEmbedding;
	protected sinks?: nn.Parameter;
	protected readonly isCausal: boolean;
	protected readonly config: TransformerModuleConfig;
	protected readonly gateAfterSdpa?: SimpleHadamardGate;
	protected readonly gateAfterQ?: SimpleHadamardGate;
	protected readonly gateAfterK?: SimpleHadamardGate;
	protected readonly gateAfterV?: SimpleHadamardGate;
	protected readonly gateAfterFinal?: SimpleHadamardGate;

	constructor(embeddingSize: number, config: TransformerModuleConfig, isCausal: boolean = false) {
		super();

		this.nHeads =
			config.attention.nKeyValueHeads *
			(config.attention.groupedQueryAttention.present
				? config.attention.groupedQueryAttention.queryHeadsPerKeyValueHead
				: 1);
		this.nKvHeads = config.attention.nKeyValueHeads;
		this.embeddingSize = embeddingSize;
		this.headDim = embeddingSize / this.nHeads;
		this.isCausal = isCausal;

		this.config = config;

		if (config.attention.sinks?.present) {
			this.sinks = new nn.Parameter(zeros([this.nHeads], { device: gpu }));
		}

		const gatingConfig = config.attention.gating;
		if (gatingConfig?.present) {
			const controlDim = this.embeddingSize;
			const act = gatingConfig.activation;
			if (gatingConfig.sites.afterSdpaOutput) {
				this.gateAfterSdpa = new SimpleHadamardGate(controlDim, this.embeddingSize, act);
			}
			if (gatingConfig.sites.afterFinalOutputProjection) {
				this.gateAfterFinal = new SimpleHadamardGate(controlDim, this.embeddingSize, act);
			}
			if (gatingConfig.sites.afterQueryProjection) {
				this.gateAfterQ = new SimpleHadamardGate(controlDim, this.headDim, act);
			}
			if (gatingConfig.sites.afterKeyProjection) {
				this.gateAfterK = new SimpleHadamardGate(controlDim, this.headDim, act);
			}
			if (gatingConfig.sites.afterValueProjection) {
				this.gateAfterV = new SimpleHadamardGate(controlDim, this.headDim, act);
			}
		}
	}

	protected applyQkvGating(
		control: Tensor,
		q: Tensor,
		k: Tensor,
		v: Tensor
	): [Tensor, Tensor, Tensor] {
		if (this.gateAfterQ) {
			q = this.gateAfterQ.forward(q, control);
		}
		if (this.gateAfterK) {
			k = this.gateAfterK.forward(k, control);
		}
		if (this.gateAfterV) {
			v = this.gateAfterV.forward(v, control);
		}
		return [q, k, v];
	}

	protected runAttention(
		input: Tensor,
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

		// Optional QK norm and gating
		[q, k] = this.applyQKNorm(q, k);
		[q, k, v] = this.applyQkvGating(input, q, k, v);

		// Optional RoPE with separate offsets for q and k
		if (options.ropeOffsets) {
			[q, k] = this.applyRoPE(q, k, options.ropeOffsets.qOffset, options.ropeOffsets.kOffset);
		}

		// Concatenate with cache if present
		const kCat = options.cache ? cat([options.cache.k, k], { dim: 2 }) : k;
		const vCat = options.cache ? cat([options.cache.v, v], { dim: 2 }) : v;
		const T_kv = kCat.size(2);

		// Compute attention scores
		let att = q.matmul(kCat, { transRhs: true }).div(Math.sqrt(this.headDim));

		if (this.config.normalization.softcap.attention.present) {
			att = applySoftcap(att, this.config.normalization.softcap.attention.value);
		}

		// Apply ALiBi if configured
		if (this.alibi) {
			att = this.alibi.forward(att);
		}

		// Apply causal mask if needed
		if (this.isCausal) {
			att = this.applyCausalMask(att, T_q, T_kv);
		}
		if (options.attentionMask) {
			att = this.applyAttentionMask(att, options.attentionMask);
		}

		// Append sinks bias column if configured: concatenate along last dim before softmax
		if (this.sinks) {
			// Shape to [B, nHeads, 1, 1] then broadcast to [B, nHeads, T_q, 1]
			// TODO: Support proper expand (shouldn't be hard)
			const sinksCol = this.sinks
				.unsqueeze(0)
				.unsqueeze(2)
				.unsqueeze(3)
				.broadcastTo([B, this.nHeads, T_q, 1]);
			att = cat([att, sinksCol], { dim: 3 });
		}

		// Apply softmax over keys plus optional sink column
		att = att.softmax(3);
		att = this.attnDropout ? this.attnDropout.forward(att) : att;

		if (this.sinks) {
			// If sinks were appended, drop the sink column from attention before matmul with V
			att = att.slice([
				[0, B],
				[0, this.nHeads],
				[0, T_q],
				[0, att.size(3) - 1]
			]);
		}

		let y = att.matmul(vCat).transpose(1, 2).view([B, T_q, this.embeddingSize]);
		if (this.gateAfterSdpa) {
			y = this.gateAfterSdpa.forward(y, input);
		}

		// Apply output projection
		y = this.cProj.forward(y);

		// Optional gating after final output projection
		if (this.gateAfterFinal) {
			y = this.gateAfterFinal.forward(y, input);
		}

		if (this.residDropout) {
			y = this.residDropout.forward(y);
		}

		const newCache: SelfAttentionCache | null = {
			k: kCat,
			v: vCat,
			length: T_kv
		};
		return [y, newCache];
	}

	applyRoPE(q: Tensor, k: Tensor, qOffset: number, kOffset?: number) {
		if (this.rope) {
			q = this.rope.forward(q, qOffset);
			k = this.rope.forward(k, kOffset ?? qOffset);
		}
		return [q, k];
	}

	applyAttentionMask(att: Tensor, attentionMask: Tensor) {
		// Convert attention mask to appropriate shape [B, 1, 1, T_kv] and broadcast
		const mask = attentionMask.unsqueeze(1).unsqueeze(2).broadcastTo(att.size());
		return maskedFill(att, mask, -1e9);
	}

	applyCausalMask(att: Tensor, queryLen: number, keyLen: number) {
		// Apply causal mask if needed
		return queryLen <= 1
			? att
			: (() => {
					const mask = createCausalMask(queryLen, keyLen).broadcastTo(att.size());
					return maskedFill(att, mask, -1e9);
				})();
	}

	applyQKNorm(q: Tensor, k: Tensor): [Tensor, Tensor] {
		if (this.config.normalization.qkNorm.present) {
			if (this.config.normalization.qkNorm.type === 'rmsnorm') {
				q = q.rmsNorm({ eps: this.config.normalization.qkNorm.eps });
				k = k.rmsNorm({ eps: this.config.normalization.qkNorm.eps });
			} else if (this.config.normalization.qkNorm.type === 'layernorm') {
				q = q.layerNorm({ eps: this.config.normalization.qkNorm.eps });
				k = k.layerNorm({ eps: this.config.normalization.qkNorm.eps });
			} else {
				throw new Error(`Unknown QKNorm type: ${this.config.normalization.qkNorm.type}`);
			}
		}
		return [q, k];
	}

	/**
	 * Apply grouped-query attention broadcasting to key and value tensors
	 * @param k Key tensor [B, nKvHeads, seqLen, headDim]
	 * @param v Value tensor [B, nKvHeads, seqLen, headDim]
	 * @param B Batch size
	 * @param seqLen Sequence length
	 * @returns Broadcasted [k, v] tensors [B, nHeads, seqLen, headDim]
	 */
	applyGroupedQueryBroadcast(k: Tensor, v: Tensor, B: number, seqLen: number): [Tensor, Tensor] {
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
}

export class SelfAttention extends CoreAttention {
	private readonly cAttn: Linear;
	protected readonly cProj: Linear;

	constructor(embeddingSize: number, config: TransformerModuleConfig, isCausal: boolean = false) {
		super(embeddingSize, config, isCausal);

		const kvDim = this.headDim * this.nKvHeads;
		const qkvOutDim = embeddingSize + 2 * kvDim;
		this.cAttn = new nn.Linear(embeddingSize, qkvOutDim);
		this.cProj = new nn.Linear(embeddingSize, embeddingSize);

		if (config.dropout.transformer.attention > 0) {
			this.attnDropout = new nn.Dropout(config.dropout.transformer.attention);
		}

		if (config.dropout.transformer.residual > 0) {
			this.residDropout = new nn.Dropout(config.dropout.transformer.residual);
		}

		if (config.positionalEncoding.present && config.positionalEncoding.type === 'rope') {
			this.rope = new RotaryEmbedding(this.headDim, config.positionalEncoding.rope.base);
		}

		if (config.positionalEncoding.present && config.positionalEncoding.type === 'alibi') {
			this.alibi = new AlibiEmbedding(config.positionalEncoding.alibi.maxBias);
		}
	}

	private projectQkv(input: Tensor): [Tensor, Tensor, Tensor] {
		const [B, T, _] = input.size();
		const kvDim = this.headDim * this.nKvHeads;
		const keyPos = this.embeddingSize;
		const valuePos = this.embeddingSize + kvDim;
		const qkv = this.cAttn.forward(input);
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

	forward(
		input: Tensor,
		options: { attentionMask?: Tensor | null; cache?: SelfAttentionCache | null } = {}
	): { output: Tensor; pastKeyValues?: SelfAttentionCache } {
		const qkv = this.projectQkv(input);
		const [q, k, v] = qkv;
		const pastLen = options.cache?.length ?? 0;
		const [y, newCache] = this.runAttention(input, q, k, v, {
			attentionMask: options.attentionMask ?? null,
			cache: options.cache ?? null,
			ropeOffsets: { qOffset: pastLen, kOffset: pastLen }
		});
		return { output: y, pastKeyValues: newCache ?? undefined };
	}
}

export class CrossAttention extends CoreAttention {
	private readonly qProj: Linear;
	private readonly kvProj: Linear;
	protected readonly cProj: Linear;

	constructor(embeddingSize: number, config: TransformerModuleConfig) {
		super(embeddingSize, config);
		this.qProj = new nn.Linear(embeddingSize, embeddingSize);
		const kvDim = this.headDim * this.nKvHeads;
		this.kvProj = new nn.Linear(embeddingSize, 2 * kvDim);
		this.cProj = new nn.Linear(embeddingSize, embeddingSize);

		if (config.dropout.transformer.attention > 0) {
			this.attnDropout = new nn.Dropout(config.dropout.transformer.attention);
		}

		if (config.dropout.transformer.residual > 0) {
			this.residDropout = new nn.Dropout(config.dropout.transformer.residual);
		}
	}

	forward(
		query: Tensor,
		keyValue: Tensor,
		options: { attentionMask?: Tensor | null; cache?: SelfAttentionCache | null } = {}
	): { output: Tensor; pastKeyValues?: SelfAttentionCache } {
		const [B, T_q, _q] = query.size();
		let q = this.qProj.forward(query);
		const qShape = [B, T_q, this.nHeads, this.headDim];
		q = q.view(qShape)?.transpose(1, 2);

		let k: Tensor;
		let v: Tensor;
		let returnCache: SelfAttentionCache | null = null;
		if (options.cache) {
			// Reuse precomputed base encoder K/V from cache for cross-attention
			k = options.cache.k;
			v = options.cache.v;
			returnCache = options.cache;
		} else {
			// Compute and reshape encoder K/V once, cache base K/V for reuse across decoding steps
			const kvDim = this.headDim * this.nKvHeads;
			const kv = this.kvProj.forward(keyValue);
			const kProj = kv.slice([
				[0, B],
				[0, keyValue.size(1)],
				[0, kvDim]
			]);
			const vProj = kv.slice([
				[0, B],
				[0, keyValue.size(1)],
				[kvDim, kvDim + kvDim]
			]);
			const kvShape = [B, keyValue.size(1), this.nKvHeads, this.headDim];
			k = kProj.view(kvShape)?.transpose(1, 2);
			v = vProj.view(kvShape)?.transpose(1, 2);
			[k, v] = this.applyGroupedQueryBroadcast(k, v, B, keyValue.size(1));
			returnCache = { k, v, length: keyValue.size(1) };
		}

		const [y, _ignored] = this.runAttention(query, q, k, v, {
			attentionMask: options.attentionMask ?? null,
			// No KV concatenation for cross-attention; K/V are static from encoder
			cache: null,
			ropeOffsets: { qOffset: 0, kOffset: 0 }
		});
		return { output: y, pastKeyValues: returnCache ?? undefined };
	}
}
