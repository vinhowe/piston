import type { LayerNormPosition } from '$lib/workspace/config';

import { nn, Tensor } from '@piston-ml/piston-web';

import type { DecoderLayerCache } from '../cache';
import type { TransformerModuleConfig } from '../config';

import { CrossAttention, SelfAttention } from './attention';
import { MLP } from './mlp';
import { createNorm } from './norm';

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

export class DecoderLayer extends nn.Module {
	private readonly lnSelfAttn?: nn.Module<[Tensor], Tensor>;
	private readonly selfAttn?: SelfAttention;
	private readonly lnCrossAttn?: nn.Module<[Tensor], Tensor>;
	private readonly crossAttn?: CrossAttention;
	private readonly lnMlp?: nn.Module<[Tensor], Tensor>;
	private readonly mlp?: MLP;
	private readonly dropout?: nn.Dropout;
	private readonly layernormPosition: LayerNormPosition;

	constructor(config: TransformerModuleConfig, crossAttention: boolean = false) {
		super();

		if (config.attention.present) {
			const lnPresent = config.layerNormalization.transformer.present;
			const lnPos = config.layerNormalization.transformer.position;
			if (lnPresent && lnPos === 'pre') {
				this.lnSelfAttn = createNorm(config.embeddingSize, config.layerNormalization);
			}
			this.selfAttn = new SelfAttention(config.embeddingSize, config, true);
			if (lnPresent && lnPos === 'post') {
				this.lnSelfAttn = createNorm(config.embeddingSize, config.layerNormalization);
			}
		}

		if (crossAttention && config.attention.present) {
			const lnPresent = config.layerNormalization.transformer.present;
			const lnPos = config.layerNormalization.transformer.position;
			if (lnPresent && lnPos === 'pre') {
				this.lnCrossAttn = createNorm(config.embeddingSize, config.layerNormalization);
			}
			this.crossAttn = new CrossAttention(config.embeddingSize, config);
			if (lnPresent && lnPos === 'post') {
				this.lnCrossAttn = createNorm(config.embeddingSize, config.layerNormalization);
			}
		}

		if (config.mlp.present) {
			const lnPresent = config.layerNormalization.transformer.present;
			const lnPos = config.layerNormalization.transformer.position;
			if (lnPresent && lnPos === 'pre') {
				this.lnMlp = createNorm(config.embeddingSize, config.layerNormalization);
			}
			this.mlp = new MLP(config.embeddingSize, config.mlp);
			if (lnPresent && lnPos === 'post') {
				this.lnMlp = createNorm(config.embeddingSize, config.layerNormalization);
			}
		}

		if (config.dropout.transformer.residual > 0) {
			this.dropout = new nn.Dropout(config.dropout.transformer.residual);
		}

		this.layernormPosition = config.layerNormalization.transformer.position;
	}

	forward(input: Tensor, options: DecoderLayerForwardOptions = {}): DecoderLayerForwardResult {
		const encoderHiddenStates = options.encoderHiddenStates ?? null;
		const srcPaddingMask = options.srcPaddingMask ?? null;
		const tgtPaddingMask = options.tgtPaddingMask ?? null;
		const cache = options.cache ?? null;
		let x = input;
		let selfCache = cache?.self ?? null;

		if (this.selfAttn) {
			const residual = input;
			if (this.lnSelfAttn && this.layernormPosition === 'pre') {
				x = this.lnSelfAttn.forward(input);
			}
			const selfResult = this.selfAttn.forward(x, {
				attentionMask: tgtPaddingMask ?? null,
				cache: selfCache ?? null
			});
			selfCache = selfResult.pastKeyValues ?? null;
			x = residual.add(selfResult.output);
			if (this.lnSelfAttn && this.layernormPosition === 'post') {
				x = this.lnSelfAttn.forward(x);
			}
		}

		if (this.crossAttn) {
			if (!encoderHiddenStates) {
				throw new Error('Encoder hidden states are required for cross-attention');
			}
			const residual2 = x;
			if (this.lnCrossAttn && this.layernormPosition === 'pre') {
				x = this.lnCrossAttn.forward(x);
			}
			const crossResult = this.crossAttn.forward(x, encoderHiddenStates, {
				attentionMask: srcPaddingMask ?? null,
				cache: cache?.cross ?? null
			});
			// If caching is enabled, store cross-attn K/V once
			if (cache) {
				if (!cache.cross && crossResult.pastKeyValues) {
					cache.cross = crossResult.pastKeyValues;
				}
			}
			x = residual2.add(crossResult.output);
			if (this.lnCrossAttn && this.layernormPosition === 'post') {
				x = this.lnCrossAttn.forward(x);
			}
		}

		if (this.mlp) {
			const residual3 = x;
			if (this.lnMlp && this.layernormPosition === 'pre') {
				x = this.lnMlp.forward(x);
			}
			x = this.mlp.forward(x);
			if (this.dropout) {
				x = this.dropout.forward(x);
			}
			x = residual3.add(x);
			if (this.lnMlp && this.layernormPosition === 'post') {
				x = this.lnMlp.forward(x);
			}
		}

		const result: DecoderLayerForwardResult = { output: x };
		if (cache) {
			result.pastKeyValues = { self: selfCache ?? undefined, cross: cache.cross ?? undefined };
		}
		return result;
	}
}
