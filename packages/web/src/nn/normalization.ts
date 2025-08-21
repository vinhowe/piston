import { gpu, ones, zeros } from "@/globals";
import { Module } from "@/nn/module";
import { Parameter } from "@/nn/parameter";
import { Tensor } from "@/tensor";

export class LayerNorm extends Module<[Tensor], Tensor> {
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
    return input.layerNorm({ weight: this.weight, bias: this.bias, eps: this.eps });
  }
}

export class RMSNorm extends Module<[Tensor], Tensor> {
  weight: Parameter;

  constructor(
    normalizedShape: number,
    private eps = 1e-5,
  ) {
    super();
    this.weight = ones([normalizedShape], { device: gpu, requiresGrad: true });
  }

  forward(input: Tensor) {
    return input.rmsNorm({ weight: this.weight, eps: this.eps });
  }
}
