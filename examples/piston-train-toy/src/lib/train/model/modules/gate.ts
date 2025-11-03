import type { Activation } from '$lib/workspace/config';

import { nn, Tensor } from '@piston-ml/piston-web';

export class SimpleHadamardGate extends nn.Module {
	private readonly tau: nn.Linear;
	private readonly activation: (x: Tensor) => Tensor;

	constructor(controlDim: number, targetDim: number, activationName: Activation) {
		super();
		this.tau = new nn.Linear(controlDim, targetDim);
		this.activation = (x: Tensor): Tensor => x[activationName]();
	}

	forward(gatedTensor: Tensor, gateInput: Tensor): Tensor {
		const g = this.activation(this.tau.forward(gateInput));
		return gatedTensor.mul(g.size().length === gatedTensor.size().length ? g : g.unsqueeze(1));
	}
}
