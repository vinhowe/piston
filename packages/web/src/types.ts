import { Parameter } from "@/nn/parameter";
import { Tensor } from "@/tensor";
import { Device, DType } from "@/wasm";

export type ShapeType = number | number[] | Uint32Array;
export type ShapeWithHolesType = number | number[] | Int32Array;
export type DimsType = number[] | Int32Array | number;
export type NestedNumberList = number | NestedNumberList[];
export type TensorOrScalar = Tensor | number;

export type DTypeInitConfig = {
  dtype?: DType;
};
export type DeviceInitConfig = {
  device?: Device;
};
export type VarConfig = {
  requiresGrad?: boolean;
};
export type OptionalShapeConfig = {
  shape?: ShapeType;
};
export type TensorOptions = {
  dtype?: DType;
  device?: Device;
  requiresGrad?: boolean;
};

export type TensorCreationFunction<Args extends unknown[]> = {
  (...args: [...Args, TensorOptions?]): Parameter;
  (...args: [...Args, TensorOptions?]): Tensor;
};

export * from "@/core";
