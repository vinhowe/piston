import type { LayerNormPosition } from '$lib/workspace/config';

import { type Module, nn, Tensor } from '@piston-ml/piston-web';

import type { TransformerModuleConfig } from '../config';

import { SelfAttention } from './attention';
import { MLP } from './mlp';
import { createNorm } from './norm';

export class EncoderLayer extends nn.Module {
	private readonly lnAttn?: Module<[Tensor], Tensor>;
	private readonly attn?: SelfAttention;
	private readonly lnMlp?: Module<[Tensor], Tensor>;
	private readonly mlp?: MLP;
	private readonly dropout?: nn.Dropout;
	private readonly layernormPosition: LayerNormPosition;

	constructor(config: TransformerModuleConfig) {
		super();

		if (config.attention.present) {
			this.attn = new SelfAttention(config.embeddingSize, config, false);
			if (config.layerNormalization.transformer.present) {
				this.lnAttn = createNorm(config.embeddingSize, config.layerNormalization);
			}
		}

		if (config.mlp.present) {
			this.mlp = new MLP(config.embeddingSize, config.mlp);
			if (config.layerNormalization.transformer.present) {
				this.lnMlp = createNorm(config.embeddingSize, config.layerNormalization);
			}
		}

		if (config.dropout.transformer.residual > 0) {
			this.dropout = new nn.Dropout(config.dropout.transformer.residual);
		}

		this.layernormPosition = config.layerNormalization.transformer.position;
	}

	forward(input: Tensor, attentionMask: Tensor | null = null): Tensor {
		let x = input;
		if (this.attn) {
			const residual = input;
			if (this.lnAttn && this.layernormPosition === 'pre') {
				x = this.lnAttn.forward(input) as Tensor;
			}
			const attnOutput = this.attn.forward(x, { attentionMask });
			x = residual.add(attnOutput.output);
			if (this.lnAttn && this.layernormPosition === 'post') {
				x = this.lnAttn.forward(x) as Tensor;
			}
		}

		if (this.mlp) {
			const residual2 = x;
			if (this.lnMlp && this.layernormPosition === 'pre') {
				x = this.lnMlp.forward(x) as Tensor;
			}
			x = this.mlp.forward(x);
			if (this.dropout) {
				x = this.dropout.forward(x);
			}
			x = residual2.add(x);
			if (this.lnMlp && this.layernormPosition === 'post') {
				x = this.lnMlp.forward(x) as Tensor;
			}
		}

		return x;
	}
}
