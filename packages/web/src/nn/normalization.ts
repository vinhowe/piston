import { gpu, ones, zeros } from "@/globals";
import { Module } from "@/nn/module";
import { Parameter } from "@/nn/parameter";
import { Tensor } from "@/tensor";

export class LayerNorm extends Module {
  weight: Parameter;
  bias: Parameter;

  constructor(
    normalizedShape: number,
    private eps = 1e-5,
  ) {
    super();
    this.weight = ones([normalizedShape], { device: gpu, requiresGrad: true });
    this.bias = zeros([normalizedShape], { device: gpu, requiresGrad: true });
  }

  forward(input: Tensor) {
    return input.layerNorm(this.weight, this.bias, this.eps);
  }
}

export class RMSNorm extends Module {
  weight: Parameter;

  constructor(
    normalizedShape: number,
    private eps = 1e-5,
  ) {
    super();
    this.weight = ones([normalizedShape], { device: gpu, requiresGrad: true });
  }

  forward(input: Tensor) {
    return input.rmsNorm(this.weight, this.eps);
  }
}
