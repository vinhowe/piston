import { arange, Device, gpu, int32, type Tensor } from '@piston-ml/piston-web';

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
export function createPositionIds(
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
