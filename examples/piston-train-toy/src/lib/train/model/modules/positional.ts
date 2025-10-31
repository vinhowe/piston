import { arange, cat, float32, gpu, nn, type Tensor } from '@piston-ml/piston-web';

export interface SinusoidalPositionalEncodingConfig {
	dropout?: number;
	maxLen?: number;
}

/**
 * Implements sinusoidal positional encodings as described in "Attention Is All You Need".
 */
export class SinusoidalEncoding extends nn.Module<[Tensor], Tensor> {
	private dropout: nn.Dropout;
	private pe: nn.Buffer;

	constructor(dModel: number, config: SinusoidalPositionalEncodingConfig = {}) {
		super();
		const { dropout = 0.1, maxLen = 500 } = config;
		this.dropout = new nn.Dropout(dropout);

		// Create positional encoding matrix
		const position = arange({ end: maxLen, dtype: float32, device: gpu }).unsqueeze(1);
		const divTerm = arange({ end: dModel, dtype: float32, device: gpu })
			.mul(-Math.log(10000.0) / dModel)
			.exp();
		const angles = position.mul(divTerm.unsqueeze(0));

		// Build [sin, cos] pairs and flatten -> (max_len, d_model)
		const pe = cat([angles.sin(), angles.cos()], { dim: -1 }).flatten({ startDim: 1 }).unsqueeze(0);

		this.pe = new nn.Buffer(pe);
	}

	forward(x: Tensor, offset: number = 0): Tensor {
		// x shape: (batch, seqLen, dModel)
		const seqLen = x.size(1);
		const start = offset;
		const end = offset + seqLen;
		x = x.add(
			this.pe.slice([
				[0, 1],
				[start, end],
				[0, x.size(2)]
			])
		);
		return this.dropout.forward(x);
	}
}
