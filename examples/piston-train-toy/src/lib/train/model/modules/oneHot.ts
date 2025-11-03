import { float32, nn, oneHot, type Tensor } from '@piston-ml/piston-web';

export class OneHotEmbedding extends nn.Module {
	private readonly vocabSize: number;

	constructor(vocabSize: number) {
		super();
		this.vocabSize = vocabSize;
	}

	forward(inputIds: Tensor): Tensor {
		return oneHot(inputIds, this.vocabSize).cast(float32);
	}
}
