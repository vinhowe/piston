import { Parameter } from "@/nn/parameter";
import { Tensor } from "@/tensor";
import {
  FullInitConfig,
  NestedNumberList,
  OptionalShapeConfig,
  RequiresGradConfig,
  ShapeType,
  TensorCreationFunction,
  VarConfig,
} from "@/types";
import {
  cpu_wasm,
  Device,
  DType,
  float16_wasm,
  float32_wasm,
  gpu_wasm,
  int32_wasm,
  seed_wasm,
  Tensor_wasm,
  uint32_wasm,
} from "@/wasm";

// dtypes
export let float32: DType;
export let float16: DType;
export let int32: DType;
export let uint32: DType;

// devices
export let cpu: Device;
export let gpu: Device;

// general utils
export let seed: (seed: number) => void;

// tensor creation functions
export let full: TensorCreationFunction<[shape: ShapeType, value: number]>;
export let cat: (tensors: Tensor[], dim: number) => Tensor;
export let stack: (tensors: Tensor[], dim: number) => Tensor;
export let arange: TensorCreationFunction<[start: number, end?: number, step?: number]>;
export let randint: TensorCreationFunction<[low: number, high: number, shape: ShapeType]>;
export let randn: TensorCreationFunction<[shape: ShapeType]>;
export let randnMeanStd: TensorCreationFunction<[mean: number, std: number, shape: ShapeType]>;
export let rand: TensorCreationFunction<[lo: number, up: number, shape: ShapeType]>;
// This just calls .bernoulli() on the tensor
export let bernoulli: (t: Tensor) => Tensor;
export let zeros: TensorCreationFunction<[shape: ShapeType]>;
export let zerosLike: TensorCreationFunction<[t: Tensor]>;
export let ones: TensorCreationFunction<[shape: ShapeType]>;
export let onesLike: TensorCreationFunction<[t: Tensor]>;

export let tensor: {
  (data: CreateTensorData, config: RequiresGradConfig & OptionalShapeConfig): Parameter;
  (data: CreateTensorData, config?: FullInitConfig & OptionalShapeConfig): Tensor;
};

type CreateTensorData =
  | number[]
  | NestedNumberList[]
  | Uint8Array
  | Float32Array
  | Float64Array
  | Int32Array
  | Uint32Array;

function parseDevice(device?: Device | string | undefined): Device {
  if (typeof device === "string") {
    if (device === "cpu") {
      return cpu._clone();
    } else if (device === "webgpu" || device === "gpu") {
      return gpu._clone();
    } else {
      throw new Error(`Unknown device: ${device}`);
    }
  }
  return (device ?? gpu)._clone();
}

function parseDType(dtype?: DType | string | undefined): DType {
  if (typeof dtype === "string") {
    if (dtype === "float32") {
      return float32._clone();
    } else if (dtype === "float16") {
      return float16._clone();
    } else if (dtype === "int32") {
      return int32._clone();
    } else if (dtype === "uint32") {
      return uint32._clone();
    }
    throw new Error(`Unknown dtype: ${dtype}`);
  }

  return (dtype ?? float32)._clone();
}

function tensorCreationArgs(config?: FullInitConfig): [DType, Device, boolean] {
  return [parseDType(config?.dtype), parseDevice(config?.device), config?.requiresGrad ?? false];
}

function unwrapFullConfigArgs<Args extends unknown[]>(
  fn: (...args: [...Args, DType, Device, boolean]) => Tensor_wasm,
): (...args: [...Args, FullInitConfig?]) => Tensor_wasm {
  return (...args: [...Args, FullInitConfig?]) => {
    let configArg = args[args.length - 1] as FullInitConfig | undefined;
    let restArgs;
    if (
      (configArg &&
        Object.prototype.hasOwnProperty.call(configArg as Record<string, unknown>, "device")) ||
      Object.prototype.hasOwnProperty.call(configArg as Record<string, unknown>, "dtype") ||
      Object.prototype.hasOwnProperty.call(configArg as Record<string, unknown>, "requiresGrad")
    ) {
      // We have a config argument, so we need to remove it from the rest of the
      // arguments
      restArgs = args.slice(0, args.length - 1) as Args;
    } else {
      restArgs = args as unknown as Args;
      configArg = undefined;
    }

    const tensor = fn(
      ...restArgs,
      configArg?.dtype?._clone() ?? float32._clone(),
      parseDevice(configArg?.device),
      configArg?.requiresGrad ?? false,
    );
    return tensor;
  };
}

function wrapWithLibTensor<Args extends unknown[]>(
  fn: (...args: Args) => Tensor_wasm,
): (...args: Args) => Tensor {
  return (...args: Args) => {
    return Tensor._wrap(fn(...args));
  };
}

function wrapWithParam<Args extends unknown[]>(
  fn: (...args: [...Args, VarConfig?]) => Tensor,
): TensorCreationFunction<Args> {
  return (...args: [...Args, VarConfig?]): Parameter => {
    const configArg = args[args.length - 1] as VarConfig | undefined;
    const restArgs = args.slice(0, args.length - 1) as Args;

    const tensor = fn(...restArgs, configArg);
    if (configArg?.requiresGrad) {
      // Create a Parameter from the tensor and wrap it with proxy
      return new Parameter(tensor);
    }
    return tensor;
  };
}

export function inferShapeFromNestedArray(arr: NestedNumberList): number[] {
  if (!Array.isArray(arr)) {
    // Base case: number
    return [];
  }

  if (!Array.isArray(arr[0])) {
    // Base case: array of numbers
    return [arr.length];
  }

  if (arr.length === 0) return [0];

  // Get shape of first element
  const restShape = inferShapeFromNestedArray(arr[0]);

  // Verify all elements have the same shape
  for (let i = 1; i < arr.length; i++) {
    const shape = inferShapeFromNestedArray(arr[i]);
    if (shape.length !== restShape.length) {
      throw new Error("Inconsistent dimensions in nested array");
    }
    for (let j = 0; j < shape.length; j++) {
      if (shape[j] !== restShape[j]) {
        throw new Error("Inconsistent dimensions in nested array");
      }
    }
  }

  // Return the full shape: [this level's length, ...shape of elements]
  return [arr.length, ...restShape];
}

export async function initGlobals() {
  float32 = float32_wasm();
  float16 = float16_wasm();
  int32 = int32_wasm();
  uint32 = uint32_wasm();

  cpu = await cpu_wasm();
  gpu = await gpu_wasm();
  seed = (seed: number | bigint) => {
    seed_wasm(typeof seed === "bigint" ? seed : BigInt(seed));
  };

  full = wrapWithParam(wrapWithLibTensor(unwrapFullConfigArgs(Tensor_wasm.full)));
  cat = wrapWithLibTensor((tensors: Tensor[], dim: number) => {
    return Tensor_wasm.cat(
      tensors.map((t) => Tensor._unwrap(t)),
      dim,
    );
  });
  stack = wrapWithLibTensor((tensors: Tensor[], dim: number) => {
    return Tensor_wasm.stack(
      tensors.map((t) => Tensor._unwrap(t)),
      dim,
    );
  });
  arange = wrapWithParam(
    wrapWithLibTensor((start: number, end?: number, step?: number, config?: FullInitConfig) => {
      if (end === undefined) {
        end = start;
        start = 0;
      }
      if (step === undefined) {
        step = 1;
      }
      return Tensor_wasm.arangeStep(start, end, step, ...tensorCreationArgs(config));
    }),
  );
  randint = wrapWithParam(wrapWithLibTensor(unwrapFullConfigArgs(Tensor_wasm.randint)));
  // This puts us in line with the PyTorch API
  randn = wrapWithParam(
    wrapWithLibTensor((shape?: ShapeType, config?: FullInitConfig) => {
      return Tensor_wasm.randn(0, 1, shape as Uint32Array, ...tensorCreationArgs(config));
    }),
  );
  randnMeanStd = wrapWithParam(wrapWithLibTensor(unwrapFullConfigArgs(Tensor_wasm.randn)));
  rand = wrapWithParam(wrapWithLibTensor(unwrapFullConfigArgs(Tensor_wasm.rand)));
  bernoulli = wrapWithLibTensor((t: Tensor) => Tensor._unwrap(t).bernoulli());
  zeros = wrapWithParam(wrapWithLibTensor(unwrapFullConfigArgs(Tensor_wasm.zeros)));
  zerosLike = wrapWithParam(
    wrapWithLibTensor((t: Tensor, config?: FullInitConfig) => {
      return Tensor._unwrap(t).zerosLike(...tensorCreationArgs(config));
    }),
  );
  ones = wrapWithParam(wrapWithLibTensor(unwrapFullConfigArgs(Tensor_wasm.ones)));
  onesLike = wrapWithParam(
    wrapWithLibTensor((t: Tensor, config?: FullInitConfig) => {
      return Tensor._unwrap(t).onesLike(...tensorCreationArgs(config));
    }),
  );
  tensor = wrapWithParam(
    wrapWithLibTensor(
      (
        data:
          | NestedNumberList[]
          | Uint8Array
          | Float32Array
          | Float64Array
          | Int32Array
          | Uint32Array,
        config?: FullInitConfig & OptionalShapeConfig,
      ) => {
        // Infer shape here, if we're looking at a nested number list
        let shape: ShapeType | undefined = config?.shape;

        let dataShape;
        if (Array.isArray(data)) {
          // For nested arrays, recursively determine shape
          dataShape = inferShapeFromNestedArray(data);
        } else {
          // For TypedArrays, use length as a 1D shape
          dataShape = [data.length];
        }

        const dataNumel = dataShape.reduce((a, b) => a * b, 1);

        if (!shape) {
          shape = dataShape;
        } else {
          // Verify that data shape and shape are correct
          const providedShapeArray = Array.isArray(shape) ? shape : Array.from(shape);
          const shapeNumel = providedShapeArray.reduce((a, b) => a * b, 1);
          if (shapeNumel !== dataNumel) {
            throw Error(
              `Data shape [${dataShape}] can't be reshaped to provided shape [${providedShapeArray}]`,
            );
          }
        }

        let castData;
        if (Array.isArray(data)) {
          data = data.flat();
        }
        if (config?.dtype) {
          if (config.dtype === float32) {
            castData = new Float32Array(data as number[]);
          } else if (config.dtype === float16) {
            castData = new Float16Array(data as number[]);
          } else if (config.dtype === int32) {
            castData = new Int32Array(data as number[]);
          } else if (config.dtype === uint32) {
            castData = new Uint32Array(data as number[]);
          }
        } else {
          castData = Array.isArray(data) ? new Float32Array(data as number[]) : data;
        }

        return Tensor_wasm.fromData(
          castData,
          shape,
          parseDevice(config?.device),
          config?.requiresGrad ?? false,
        );
      },
    ),
  );
}
