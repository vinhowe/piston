import { stack, zeros } from "@/globals";
import { Tensor } from "@/tensor";

export interface TotalNormOptions {
  normType: "fro" | "inf" | "-inf" | "0" | "1" | "-1" | "2" | number | null | undefined;
  errorIfNonfinite?: boolean;
}

export function getTotalNorm(tensors: Tensor | Tensor[], options?: TotalNormOptions): Tensor {
  const { normType, errorIfNonfinite } = options ?? {};

  if (errorIfNonfinite) {
    throw new Error("errorIfNonfinite is not yet implemented (no isnan/isinf support yet)");
  }

  if (!Array.isArray(tensors)) {
    tensors = [tensors];
  }

  if (tensors.length === 0) {
    return zeros([1], { device: tensors[0].device });
  }

  // TODO: We assume the tensors are all on the same device (single GPU) for now
  const norms = tensors.map((tensor) => tensor.norm({ ord: normType }));
  const totalNorm = stack(norms).norm({ ord: normType });

  // TODO: This is where we'd implement nonfinite handling, but lazy evaluation makes it unlikely
  // that we'd know this here. A drawback of lazy evaluation, but we can check this manually later
  // when we pull to the CPU later.

  return totalNorm;
}

export function getTotalGradNorm(parameters: Tensor[], options?: TotalNormOptions): Tensor {
  const grads = parameters.map((parameter) => parameter.grad).filter((grad) => grad !== undefined);
  return getTotalNorm(grads, options);
}

export function clipGradsWithNorm_(parameters: Tensor[], maxNorm: number, totalNorm: Tensor) {
  if (!Array.isArray(parameters)) {
    parameters = [parameters];
  }

  const grads = parameters.map((parameter) => parameter.grad).filter((grad) => grad !== undefined);

  if (grads.length === 0) {
    return;
  }

  const clipCoef = totalNorm.add(1e-6).recip().mul(maxNorm);
  const clipCoefClamped = clipCoef.clamp({ max: 1 });
  grads.forEach((grad) => grad.mul_(clipCoefClamped));
}

export function clipGradNorm_(
  parameters: Tensor | Tensor[],
  maxNorm: number,
  options?: TotalNormOptions,
): Tensor | undefined {
  if (!Array.isArray(parameters)) {
    parameters = [parameters];
  }

  if (parameters.length === 0) {
    return;
  }

  const { normType, errorIfNonfinite } = options ?? {};

  const totalNorm = getTotalGradNorm(parameters, { normType, errorIfNonfinite });
  clipGradsWithNorm_(parameters, maxNorm, totalNorm);
  return totalNorm;
}

export function calculateFanInAndFanOut(tensor: Tensor) {
  const dims = tensor.size();
  const receptive_field_size = dims.slice(2).reduce((acc, dim) => acc * dim, 1);
  const fan_in = dims.length < 2 ? 1 : dims[1] * receptive_field_size;
  const fan_out = dims.length === 0 ? 1 : dims[0] * receptive_field_size;
  return [fan_in, fan_out];
}
