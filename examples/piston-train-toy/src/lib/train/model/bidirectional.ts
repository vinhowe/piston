import type { LayerNormalizationConfig } from '$lib/workspace/config';

import { nn, Tensor } from '@piston-ml/piston-web';

import { createNorm } from './modules/norm';

/**
 * MLM Head for masked language modeling
 */
export class MLMHead extends nn.Module {
	private readonly transform: nn.Linear;
	private readonly layernorm?: nn.Module<[Tensor], Tensor>;
	private readonly decoder: nn.Linear;

	/**
	 * @param hiddenSize - Hidden dimension
	 * @param vocabSize - Vocabulary size
	 * @param layerNormalization - Layer normalization configuration
	 * @param embeddings - Optional embedding layer to share weights with
	 */
	constructor(
		hiddenSize: number,
		vocabSize: number,
		layerNormalization: LayerNormalizationConfig,
		embeddings?: nn.Embedding
	) {
		super();

		this.transform = new nn.Linear(hiddenSize, hiddenSize);
		if (layerNormalization.transformer.present) {
			this.layernorm = createNorm(hiddenSize, layerNormalization);
		}

		if (embeddings) {
			// Share weights with embedding layer
			this.decoder = new nn.Linear(hiddenSize, vocabSize, false);
			// Tie weights
			this.decoder.weight = new nn.Parameter(embeddings.weight);
		} else {
			this.decoder = new nn.Linear(hiddenSize, vocabSize, false);
		}
	}

	/**
	 * @param hiddenStates - Hidden states from encoder
	 * @returns MLM logits
	 */
	forward(hiddenStates: Tensor): Tensor {
		let x = this.transform.forward(hiddenStates);
		x = x.gelu(); // Standard activation for MLM head
		if (this.layernorm) {
			x = this.layernorm.forward(x) as Tensor;
		}
		return this.decoder.forward(x);
	}
}

/**
 * Pooler for classification tasks
 */
export class Pooler extends nn.Module {
	private readonly dense: nn.Linear;

	/**
	 * @param hiddenSize - Hidden dimension
	 */
	constructor(hiddenSize: number) {
		super();
		this.dense = new nn.Linear(hiddenSize, hiddenSize);
	}

	/**
	 * @param hiddenStates - Hidden states from encoder [B, T, H]
	 * @returns Pooled representation [B, H]
	 */
	forward(hiddenStates: Tensor): Tensor {
		// Take the first token (CLS token) representation
		let pooled = hiddenStates
			.slice([
				[0, hiddenStates.size(0)],
				[0, 1],
				[0, hiddenStates.size(2)]
			])
			.squeeze({ dim: 1 });

		pooled = this.dense.forward(pooled);
		return pooled.tanh();
	}
}
