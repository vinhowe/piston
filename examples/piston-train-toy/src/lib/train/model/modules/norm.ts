import type { LayerNormalizationConfig } from '$lib/workspace/config';

import { nn } from '@piston-ml/piston-web';

export type NormModule = nn.LayerNorm | nn.RMSNorm;

export function createNorm(normalizedShape: number, config: LayerNormalizationConfig): NormModule {
	if (config.type === 'layernorm') {
		return new nn.LayerNorm(normalizedShape, { eps: config.eps });
	} else if (config.type === 'rmsnorm') {
		return new nn.RMSNorm(normalizedShape, { eps: config.eps });
	} else {
		throw new Error(`Unknown norm type: ${config.type}`);
	}
}
