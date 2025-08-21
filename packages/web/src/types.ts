import { Parameter } from "@/nn/parameter";
import { Tensor } from "@/tensor";
import { Device, DType, TensorOptions } from "@/wasm";

export type ShapeType = number | number[] | Uint32Array;
export type ShapeWithHolesType = number | number[] | Int32Array;
export type DimsType = number[] | Int32Array | number;
export type NestedNumberList = number | NestedNumberList[];
export type DeviceType = Device | "cpu" | "gpu" | "webgpu";
export type TensorOrScalar = Tensor | number;

export type DTypeInitConfig = {
  dtype?: DType;
};
export type DeviceInitConfig = {
  device?: DeviceType;
};
export type VarConfig = {
  requiresGrad?: boolean;
};
export type OptionalShapeConfig = {
  shape?: ShapeType;
};
export type FullInitConfig = DTypeInitConfig & DeviceInitConfig & VarConfig;

export type RequiresGradConfig = VarConfig & { requiresGrad: true };
export type TensorCreationFunction<Args extends unknown[]> = {
  (...args: [...Args, TensorOptions?]): Parameter;
  (...args: [...Args, TensorOptions?]): Tensor;
};

export * from "@/core";
