import { Tensor } from "@/tensor";

export class Parameter extends Tensor {
  constructor(tensor: Tensor) {
    super(Tensor._unwrap(tensor).requiresGrad_(true));
  }

  __wbg_piston_tensor() {}
}

export class Buffer extends Tensor {
  constructor(
    tensor: Tensor,
    readonly persistent: boolean = true,
  ) {
    super(Tensor._unwrap(tensor).requiresGrad_(false));
  }

  __wbg_piston_tensor() {}
}
