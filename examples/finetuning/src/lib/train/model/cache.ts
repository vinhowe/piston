import type { Tensor } from '@piston-ml/piston-web';

export type SelfAttentionCache = {
	k: Tensor;
	v: Tensor;
	length: number;
};

export type DecoderLayerCache = {
	self?: SelfAttentionCache;
	cross?: SelfAttentionCache;
};

export type DecoderKVCache = {
	layers: DecoderLayerCache[];
};

export function createEmptyDecoderKVCache(nLayers: number): DecoderKVCache {
	return { layers: Array.from({ length: nLayers }, () => ({})) };
}
