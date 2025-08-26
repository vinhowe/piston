import { bernoulli, full } from "@/globals";
import { Module } from "@/nn/module";
import { Tensor } from "@/tensor";

export class Dropout extends Module<[Tensor], Tensor> {
  constructor(private p = 0.5) {
    super();
  }

  forward(input: Tensor) {
    if (!this.training || this.p === 0) {
      return input;
    }

    const probs = full(input.size(), this.p, { device: input.device });
    const mask = bernoulli(probs);
    return input.mul(mask).div(1 - this.p);
  }
}
