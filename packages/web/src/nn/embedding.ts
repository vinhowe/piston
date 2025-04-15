import { randn } from "@/core";
import { Module } from "@/nn/module";
import { Parameter } from "@/parameter";
import { Tensor } from "@/tensor";

export class Embedding extends Module {
  weight: Parameter;
  hiddenSize: number;
  /**
   * Create a new embedding module
   * @param {number} numEmbeddings - Number of embeddings
   * @param {number} embeddingDim - Embedding dimension
   */
  constructor(numEmbeddings: number, embeddingDim: number) {
    super();
    this.weight = new Parameter(randn([numEmbeddings, embeddingDim]));
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
