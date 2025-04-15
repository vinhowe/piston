import { Module } from "@/nn/module";
import { Tensor } from "@/tensor";

export function debugBufferHook<Input>(
  module: Module<Input, Tensor>,
): Promise<Tensor> {
  const promise = new Promise<Tensor>((resolve) => {
    const hook = module.registerForwardHook(
      (_module, _input: Input, output: Tensor) => {
        resolve(output.debugTensor());
        hook.remove();
        return output;
      },
    );
  });
  return promise;
}
