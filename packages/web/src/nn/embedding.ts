import { gpu, randn } from "@/core";
import { Module } from "@/nn/module";
import { Parameter } from "@/nn/parameter";
import { Tensor } from "@/tensor";

export class Embedding extends Module<[Tensor], Tensor> {
  weight: Parameter;
  hiddenSize: number;
  /**
   * Create a new embedding module
   * @param {number} numEmbeddings - Number of embeddings
   * @param {number} embeddingDim - Embedding dimension
   */
  constructor(numEmbeddings: number, embeddingDim: number) {
    super();
    this.weight = randn([numEmbeddings, embeddingDim], {
      device: gpu,
      requiresGrad: true,
    });
    this.hiddenSize = embeddingDim;
  }

  forward(input: Tensor) {
    const finalDims = input.size();
    finalDims.push(this.hiddenSize);
    const indexes = input.flatten();
    const values = this.weight.indexSelect(indexes, 0);
    return values.view(finalDims);
  }
}
