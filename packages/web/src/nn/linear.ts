import { gpu, initKaimingUniform_, initUniform_, zeros } from "@/globals";
import { Module } from "@/nn/module";
import { Parameter } from "@/nn/parameter";
import { Tensor } from "@/tensor";

import { calculateFanInAndFanOut } from "./utils";

export class Linear extends Module<[Tensor], Tensor> {
  weight: Parameter;
  bias?: Parameter;
  /**
   * Create a new linear module
   * @param {number} inFeatures - Input features
   * @param {number} outFeatures - Output features
   * @param {boolean} bias - Whether to use bias
   */
  constructor(inFeatures: number, outFeatures: number, bias = true) {
    super();
    this.weight = zeros([outFeatures, inFeatures], { device: gpu, requiresGrad: true });

    if (bias) {
      this.bias = zeros([outFeatures], { device: gpu, requiresGrad: true });
    }

    this.resetParameters();
  }

  resetParameters() {
    initKaimingUniform_(this.weight, { a: Math.sqrt(5) });
    if (this.bias) {
      const [fanIn] = calculateFanInAndFanOut(this.weight);
      const bound = fanIn > 0 ? Math.sqrt(1 / fanIn) : 0;
      initUniform_(this.bias, { low: -bound, high: bound });
    }
  }

  forward(input: Tensor) {
    let w;
    if (input.size().length === 4) {
      w = this.weight.broadcastLeft([input.size(0), input.size(1)]);
    } else if (input.size().length === 3) {
      w = this.weight.broadcastLeft([input.size(0)]);
    } else {
      w = this.weight;
    }

    const x = input.matmul(w, { transRhs: true });

    if (this.bias) {
      return x.add(this.bias.cast(x.dtype));
    }

    return x;
  }
}
