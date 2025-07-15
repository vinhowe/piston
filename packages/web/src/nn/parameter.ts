import { Tensor } from "@/tensor";

export class Parameter extends Tensor {
  constructor(tensor: Tensor) {
    super(Tensor._unwrap(tensor).requiresGrad_(true));
  }
}

export class Buffer extends Tensor {
  constructor(
    tensor: Tensor,
    readonly persistent: boolean = true,
  ) {
    super(Tensor._unwrap(tensor).requiresGrad_(false));
  }
}
