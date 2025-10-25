import { gpu, ones, zeros } from "@/globals";
import { Module } from "@/nn/module";
import { Parameter } from "@/nn/parameter";
import { Tensor } from "@/tensor";

export interface LayerNormConfig {
  eps?: number;
  bias?: boolean;
}

export class LayerNorm extends Module<[Tensor], Tensor> {
  weight: Parameter;
  bias?: Parameter;
  eps: number;

  constructor(normalizedShape: number, config: LayerNormConfig = {}) {
    super();
    this.eps = config.eps ?? 1e-5;
    this.weight = ones([normalizedShape], { device: gpu, requiresGrad: true });
    this.bias =
      (config.bias ?? true)
        ? zeros([normalizedShape], { device: gpu, requiresGrad: true })
        : undefined;
  }

  forward(input: Tensor) {
    return input.layerNorm({ weight: this.weight, bias: this.bias, eps: this.eps });
  }
}

export interface RMSNormConfig {
  eps?: number;
}

export class RMSNorm extends Module<[Tensor], Tensor> {
  weight: Parameter;
  eps: number;

  constructor(normalizedShape: number, config: RMSNormConfig = {}) {
    super();
    this.eps = config.eps ?? 1e-5;
    this.weight = ones([normalizedShape], { device: gpu, requiresGrad: true });
  }

  forward(input: Tensor) {
    return input.rmsNorm({ weight: this.weight, eps: this.eps });
  }
}
