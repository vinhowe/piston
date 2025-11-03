import { gpu, initOnes_, initZeros_, ones, zeros } from "@/globals";
import { Module } from "@/nn/module";
import { Parameter } from "@/nn/parameter";
import { Tensor } from "@/tensor";

export interface LayerNormConfig {
  eps?: number;
  elementwiseAffine?: boolean;
  bias?: boolean;
}

export class LayerNorm extends Module<[Tensor], Tensor> {
  weight: Parameter;
  bias?: Parameter;
  eps: number;
  elementwiseAffine: boolean;

  constructor(normalizedShape: number, config: LayerNormConfig = {}) {
    super();
    this.eps = config.eps ?? 1e-5;
    this.elementwiseAffine = config.elementwiseAffine ?? true;
    this.weight = ones([normalizedShape], { device: gpu, requiresGrad: true });
    this.bias =
      (config.bias ?? true)
        ? zeros([normalizedShape], { device: gpu, requiresGrad: true })
        : undefined;

    this.resetParameters();
  }

  resetParameters() {
    if (this.elementwiseAffine) {
      initOnes_(this.weight);
      if (this.bias) {
        initZeros_(this.bias);
      }
    }
  }

  forward(input: Tensor) {
    return input.layerNorm({ weight: this.weight, bias: this.bias, eps: this.eps });
  }
}

export interface RMSNormConfig {
  eps?: number;
  elementwiseAffine?: boolean;
}

export class RMSNorm extends Module<[Tensor], Tensor> {
  weight: Parameter;
  eps: number;
  elementwiseAffine: boolean;

  constructor(normalizedShape: number, config: RMSNormConfig = {}) {
    super();
    this.eps = config.eps ?? 1e-5;
    this.elementwiseAffine = config.elementwiseAffine ?? true;
    this.weight = ones([normalizedShape], { device: gpu, requiresGrad: true });

    this.resetParameters();
  }

  resetParameters() {
    if (this.elementwiseAffine) {
      initOnes_(this.weight);
    }
  }

  forward(input: Tensor) {
    return input.rmsNorm({ weight: this.weight, eps: this.eps });
  }
}
