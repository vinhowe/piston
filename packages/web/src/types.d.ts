export type ShapeType = number[] | Uint32Array;
export type ShapeWithHolesType = number[] | Int32Array;
export type DimsType = number[] | Int32Array | number;
export type NestedNumberList = number | NestedNumberList[];
export type DeviceType = Device | "cpu" | "gpu" | "webgpu";
export type BinaryOpInput = Tensor | number;

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
  (...args: [...Args, RequiresGradConfig]): Parameter;
  (...args: [...Args, FullInitConfig?]): Tensor;
};

export * from "@/core";
