import { randn, zeros } from "@/globals";
import { Module } from "@/nn/module";
import { Parameter } from "@/parameter";
import { Tensor } from "@/tensor";

export class Linear extends Module {
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
    this.weight = new Parameter(randn([outFeatures, inFeatures]));

    if (bias) {
      this.bias = new Parameter(zeros([outFeatures]));
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

    const x = input.matmul(w, false, true);

    if (this.bias) {
      return x.add(this.bias.cast(x.dtype));
    }

    return x;
  }
}
