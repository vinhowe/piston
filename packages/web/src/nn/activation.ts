import { Module } from "@/nn/module";
import { Tensor } from "@/tensor";

export class LogSoftmax extends Module {
  constructor(private dim: number = 1) {
    super();
  }

  forward(input: Tensor): Tensor {
    // TODO: Make this a custom operation (shouldn't be hard; just need to add
    // log as an option to the softmax operation, including the max subtract)
    //
    // For numerical stability, subtract max value before exponential
    const max = input.max({ dim: this.dim, keepdim: true });
    const diff = input.sub(max);
    const sumExp = diff.exp().sum({ dim: this.dim, keepdim: true });
    const logSm = diff.sub(sumExp.log());
    return logSm;
  }
}
