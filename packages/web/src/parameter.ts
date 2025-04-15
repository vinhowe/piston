import { Tensor } from "@/tensor";

export class Parameter extends Tensor {
  constructor(tensor: Tensor) {
    super(Tensor._unwrap(tensor).requiresGrad_(true));
  }
}
