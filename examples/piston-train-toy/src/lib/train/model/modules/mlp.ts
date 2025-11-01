import type { MLPConfig } from '$lib/workspace/config';

import { nn, Tensor } from '@piston-ml/piston-web';

export class MLP extends nn.Module {
	private readonly gateProj: nn.Linear | undefined;
	private readonly upProj: nn.Linear;
	private readonly downProj: nn.Linear;
	private readonly activation: (x: Tensor) => Tensor;
	private readonly config: MLPConfig;

	/**
	 * @param config - Model configuration
	 */
	constructor(nEmbed: number, mlpConfig: MLPConfig) {
		super();

		this.config = mlpConfig;

		const intermediateSize = this.config.hiddenExpansionFactor * nEmbed;

		if (this.config.variant === 'gated') {
			this.gateProj = new nn.Linear(nEmbed, intermediateSize);
		}

		this.upProj = new nn.Linear(nEmbed, intermediateSize);
		this.downProj = new nn.Linear(intermediateSize, nEmbed);

		this.activation = (x: Tensor): Tensor => x[mlpConfig.activation]();
	}

	/**
	 * Forward pass through the MLP
	 * @param input - Input tensor
	 * @returns Output tensor
	 */
	forward(input: Tensor): Tensor {
		let h = this.upProj.forward(input);
		h = this.activation(h);
		if (this.gateProj) {
			const gate = this.gateProj.forward(input);
			h = h.mul(gate);
		}
		return this.downProj.forward(h);
	}
}
