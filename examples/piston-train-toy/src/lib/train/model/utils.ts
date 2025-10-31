import type { LayerNormalizationConfig, PositionEncodingConfig } from '$lib/workspace/config';

import {
	arange,
	CrossEntropyLoss,
	Device,
	gpu,
	int32,
	type Module as ModuleType,
	nn,
	type Tensor
} from '@piston-ml/piston-web';

import { createNorm } from './modules/norm';
import { SinusoidalEncoding } from './modules/positional';

/**
 * Create a causal (lower triangular) mask.
 * @param queryLen - Length of the current query.
 * @param keyLen - Length of the key (which may include cached tokens).
 * @returns Causal mask tensor of shape [1, numHeads, queryLen, keyLen].
 */
export function createCausalMask(queryLen: number, keyLen: number): Tensor {
	// General causal mask supporting past KV cache where keyLen may exceed queryLen.
	// We want to mask future positions: for each query i, keys j > pastLen + i are masked.
	// pastLen is inferred as keyLen - queryLen when using KV cache (else 0).
	const pastLen = Math.max(0, keyLen - queryLen);
	const i = arange({ end: queryLen, device: gpu, dtype: int32 })
		.unsqueeze(1)
		.broadcastTo([queryLen, keyLen]);
	const j = arange({ end: keyLen, device: gpu, dtype: int32 })
		.unsqueeze(0)
		.broadcastTo([queryLen, keyLen]);
	// Mask is true where positions are allowed: j <= pastLen + i
	return j.le(i.add(pastLen));
}

/**
 * Create position IDs tensor [0, 1, 2, ..., seqLen-1] and broadcast to batch size
 * @param seqLen - Sequence length
 * @param batchSize - Batch size
 * @param device - Device to place tensor on
 * @returns Position IDs tensor
 */
function createPositionIds(
	seqLen: number,
	batchSize: number,
	device: Device,
	offset: number = 0
): Tensor {
	// Create position IDs tensor [offset, offset+1, ..., offset+seqLen-1] and broadcast to batch
	const positionIds = arange({ end: seqLen, device, dtype: int32 }).add(offset).cast(int32);
	// Reshape to [1, seqLen] and broadcast to [batchSize, seqLen]
	return positionIds.unsqueeze(0).broadcastTo([batchSize, seqLen]);
}

/**
 * Apply mask to attention scores
 * @param onFalse - Attention scores
 * @param mask - Mask tensor
 * @param onTrueValue - Value to fill masked positions with
 * @returns Masked scores
 */
export function maskedFill(onTrue: Tensor, mask: Tensor, onFalseValue: number): Tensor {
	return onTrue.where(mask, onFalseValue);
}

export function addPositionalEncodingToEmbeddings(
	embeddings: Tensor,
	positionalEncodingConfig: PositionEncodingConfig,
	sinusoidalEncoding?: ModuleType<[Tensor], Tensor>,
	positionEmbeddings?: ModuleType<[Tensor], Tensor>,
	dropout?: ModuleType<[Tensor], Tensor>,
	additionalPositionOffset: number = 0
): Tensor {
	const [batchSize, seqLen] = embeddings.size();
	if (positionalEncodingConfig.present) {
		if (positionalEncodingConfig.type === 'learned') {
			// Add positional embeddings
			const positions = createPositionIds(
				seqLen,
				batchSize,
				embeddings.device,
				additionalPositionOffset
			);
			const positionEmbeddingsOutput = positionEmbeddings!.forward(positions);
			embeddings = embeddings.add(positionEmbeddingsOutput);
			// Apply embedding dropout if configured
			if (dropout) {
				embeddings = dropout.forward(embeddings);
			}
		} else if (positionalEncodingConfig.type === 'sinusoidal') {
			embeddings = (sinusoidalEncoding as SinusoidalEncoding).forward(
				embeddings,
				additionalPositionOffset
			);
		} else if (
			positionalEncodingConfig.type === 'rope' ||
			positionalEncodingConfig.type === 'alibi'
		) {
			// These are applied in the attention mechanism, so we only apply embedding dropout here, if
			// configured
			if (dropout) {
				embeddings = dropout.forward(embeddings);
			}
		}
	}
	return embeddings;
}

export function createCrossEntropyCriterion() {
	return new CrossEntropyLoss({
		ignoreIndex: -100
	});
}

/**
 * Optionally create a learned positional embedding layer (nn.Embedding variant)
 */
export function maybeCreateLearnedPositionEmbedding(
	positionalEncoding: PositionEncodingConfig,
	blockSize: number,
	embeddingSize: number
): nn.Embedding | undefined {
	if (positionalEncoding.present && positionalEncoding.type === 'learned') {
		return new nn.Embedding(blockSize, embeddingSize);
	}
	return undefined;
}

/**
 * Optionally create sinusoidal encoding module based on positional encoding config
 */
export function maybeCreateSinusoidalEncoding(
	positionalEncoding: PositionEncodingConfig,
	embeddingSize: number,
	dropout: number
): SinusoidalEncoding | undefined {
	if (positionalEncoding.present && positionalEncoding.type === 'sinusoidal') {
		return new SinusoidalEncoding(embeddingSize, { dropout });
	}
	return undefined;
}

/**
 * Optionally create final layer norm if transformer normalization position is 'pre'
 */
export function maybeCreateFinalLayerNorm(
	embeddingSize: number,
	layerNormalization: LayerNormalizationConfig
): ModuleType<[Tensor], Tensor> | undefined {
	if (layerNormalization.transformer.present && layerNormalization.transformer.position === 'pre') {
		return createNorm(embeddingSize, layerNormalization) as unknown as ModuleType<[Tensor], Tensor>;
	}
	return undefined;
}

/**
 * Build a language modeling head using standard nn.Linear with optional weight tying.
 */
export function buildLmHead(embeddingSize: number, vocabSize: number): nn.Linear {
	return new nn.Linear(embeddingSize, vocabSize, false);
}

/**
 * Compute cross-entropy loss if both logits and targets are provided, flattening as needed
 */
export function computeAutoregressiveCrossEntropyLoss(
	logits: Tensor | null,
	targets: Tensor | null,
	criterion: CrossEntropyLoss
): Tensor | null {
	if (!logits || !targets) {
		return null;
	}
	return criterion.forward(logits.view([-1, logits.size(-1)]), targets.view(-1));
}
