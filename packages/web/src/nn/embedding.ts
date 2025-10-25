import { gpu, initNormal_, zeros } from "@/core";
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
    this.weight = zeros([numEmbeddings, embeddingDim], {
      device: gpu,
      requiresGrad: true,
    });
    this.hiddenSize = embeddingDim;

    this.resetParameters();
  }

  resetParameters() {
    initNormal_(this.weight);
  }

  forward(input: Tensor) {
    const finalDims = input.size();
    finalDims.push(this.hiddenSize);
    const indexes = input.flatten();
    const values = this.weight.indexSelect(indexes, 0);
    // Basic guard against accidental zero-sized dims which would break view
    const inputSize = input.size();
    for (let i = 0; i < inputSize.length; i++) {
      if (inputSize[i] === 0) {
        throw new Error("Input tensor has a dimension with size 0, which is not allowed");
      }
    }
    return values.view(finalDims);
  }
}
