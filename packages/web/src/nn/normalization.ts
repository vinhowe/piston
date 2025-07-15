import { ones, zeros } from "@/globals";
import { Module } from "@/nn/module";
import { Parameter } from "@/parameter";
import { Tensor } from "@/tensor";

export class LayerNorm extends Module {
  weight: Parameter;
  bias: Parameter;

  constructor(
    normalizedShape: number,
    private eps = 1e-5,
  ) {
    super();
    this.weight = new Parameter(ones([normalizedShape]));
    this.bias = new Parameter(zeros([normalizedShape]));
  }

  forward(input: Tensor) {
    return input.layerNorm(this.weight, this.bias, this.eps);
  }
}
