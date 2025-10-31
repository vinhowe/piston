import { Module } from "@/nn/module";

import type { ParsedSlice } from "./types";

import { Tensor } from "../tensor";
import { DiagnosticError } from "./error";

export function isTensorLike(obj: unknown): obj is Tensor {
  return obj instanceof Tensor;
}

export function applySlicing(tensors: Tensor[], slice: ParsedSlice | null): Tensor[] {
  if (!slice) {
    return tensors;
  }

  const resultTensors: Tensor[] = [];

  for (const tensor of tensors) {
    // Get tensor rank
    const dim = tensor.dim();

    // Handle ellipsis and rank adjustment for sliceArgs
    const finalSliceArgs = [...slice.items];
    const ellipsisIndex = finalSliceArgs.findIndex((s) => s === "ellipsis");

    if (ellipsisIndex !== -1) {
      const numNonEllipsisSlices = finalSliceArgs.length - 1;
      const numEllipsisFill = dim - numNonEllipsisSlices;

      if (numEllipsisFill < 0) {
        throw new DiagnosticError(
          `Slice expression has too many dimensions for tensor of rank ${dim}.`,
          {
            from: slice.from,
            to: slice.to,
            source: slice.source,
          },
        );
      }

      const ellipsisFill = Array(numEllipsisFill).fill({
        start: null,
        stop: null,
        step: 1,
      });
      finalSliceArgs.splice(ellipsisIndex, 1, ...ellipsisFill);
    }

    // Pad with full slices if fewer slices than rank
    if (finalSliceArgs.length < dim) {
      const diff = dim - finalSliceArgs.length;
      finalSliceArgs.push(...Array(diff).fill({ start: null, stop: null, step: 1 }));
    } else if (finalSliceArgs.length > dim) {
      throw new DiagnosticError(
        `Slice expression has too many dimensions for tensor of rank ${dim}.`,
        {
          from: slice.from,
          to: slice.to,
          source: slice.source,
        },
      );
    }

    // Build the ranges for slicing
    const ranges: [number, number][] = finalSliceArgs.map((arg, dim) => {
      // For single index, create a simple range
      if (typeof arg === "number") {
        return [arg, 1]; // [start, size, step]
      }

      if (arg === "ellipsis") {
        throw new DiagnosticError("Multiple ellipsis are not allowed in slice expression.", {
          from: slice.from,
          to: slice.to,
          source: slice.source,
        });
      }

      // Handle generic slice
      const dimSize = tensor.size(dim);
      const start = arg.start !== null ? arg.start : 0;
      const stop = arg.stop !== null ? arg.stop : dimSize;

      // TODO: support step; we leave this undocumented, in a little bit of a broken state, because
      // I don't want to think about how to slice in ways that are not contiguous right now
      // const step = arg.step !== null ? arg.step : 1;
      // return [start, stop - start, step];

      return [start, stop - start];
    });

    // Apply the slice
    try {
      const slicedTensor = tensor.slice(ranges);
      resultTensors.push(slicedTensor);
    } catch (error) {
      new Error();
      throw new DiagnosticError("Error slicing tensor", {
        from: slice.from,
        to: slice.to,
        source: slice.source,
        cause: error instanceof Error ? error : new Error(String(error)),
      });
    }
  }

  return resultTensors;
}

/**
 * Applies transformations to a list of tensors.
 * This combines slicing and JavaScript expression execution.
 */
export function applyTransformations(
  module: Module | undefined,
  tensors: unknown,
  slice: ParsedSlice | null,
  transformation: ((input: unknown) => Tensor | Tensor[]) | null,
): Tensor | Tensor[] {
  let transformedTensors: Tensor | Tensor[] | unknown = tensors;
  // First apply JavaScript expression, if provided
  if (transformation) {
    transformedTensors = transformation.call(module, transformedTensors);
  }
  // Then apply slicing
  const slicedTensors = Array.isArray(transformedTensors as Tensor | Tensor[])
    ? applySlicing(transformedTensors as Tensor[], slice)
    : applySlicing([transformedTensors as Tensor], slice)[0];
  return slicedTensors;
}
