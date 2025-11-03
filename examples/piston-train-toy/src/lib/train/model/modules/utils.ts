import type { Tensor } from '@piston-ml/piston-web';

export function applySoftcap(logits: Tensor, softcap: number): Tensor {
	return logits.div(softcap).tanh().mul(softcap);
}
