import { Parameter } from "@/nn/parameter";
import { Tensor } from "@/tensor";
import {
  NestedNumberList,
  OptionalShapeConfig,
  ShapeType,
  TensorOptions,
} from "@/types";
import {
  abs as abs_wasm,
  add as add_wasm,
  addcdiv as addcdiv_wasm,
  addcmul as addcmul_wasm,
  arange as arange_wasm,
  bernoulli as bernoulli_wasm,
  cat as cat_wasm,
  ceil as ceil_wasm,
  cos as cos_wasm,
  cpu_wasm,
  Device,
  DType,
  eq as eq_wasm,
  exp as exp_wasm,
  eye as eye_wasm,
  float16_wasm,
  float32_wasm,
  floor as floor_wasm,
  fromData,
  full as full_wasm,
  ge as ge_wasm,
  gelu as gelu_wasm,
  gpu_wasm,
  gt as gt_wasm,
  int32_wasm,
  isinf as isinf_wasm,
  isnan as isnan_wasm,
  le as le_wasm,
  log as log_wasm,
  logicalAnd as logicalAnd_wasm,
  logicalNot as logicalNot_wasm,
  logicalOr as logicalOr_wasm,
  logicalXor as logicalXor_wasm,
  lt as lt_wasm,
  ne as ne_wasm,
  neg as neg_wasm,
  ones as ones_wasm,
  onesLike as onesLike_wasm,
  rand as rand_wasm,
  randint as randint_wasm,
  randn as randn_wasm,
  recip as recip_wasm,
  relu2 as relu2_wasm,
  relu as relu_wasm,
  seed_wasm,
  sigmoid as sigmoid_wasm,
  silu as silu_wasm,
  sin as sin_wasm,
  sqrt as sqrt_wasm,
  square as square_wasm,
  stack as stack_wasm,
  swiglu as swiglu_wasm,
  tanh as tanh_wasm,
  Tensor_wasm,
  uint32_wasm,
  zeros as zeros_wasm,
  zerosLike as zerosLike_wasm,
} from "@/wasm";

export { __pistonActiveTensors } from "@/wasm";
export * as wasm from "@/wasm";

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
export let full: typeof full_wasm;
export let cat: typeof cat_wasm;
export let stack: typeof stack_wasm;
export let arange: typeof arange_wasm;
export let randint: typeof randint_wasm;
export let randn: typeof randn_wasm;
export let rand: typeof rand_wasm;
export let bernoulli: typeof bernoulli_wasm;
export let zeros: typeof zeros_wasm;
export let zerosLike: typeof zerosLike_wasm;
export let ones: typeof ones_wasm;
export let onesLike: typeof onesLike_wasm;
export let add: typeof add_wasm;
export let addcdiv: typeof addcdiv_wasm;
export let addcmul: typeof addcmul_wasm;
export let eq: typeof eq_wasm;
export let ne: typeof ne_wasm;
export let gt: typeof gt_wasm;
export let ge: typeof ge_wasm;
export let lt: typeof lt_wasm;
export let le: typeof le_wasm;
export let logicalAnd: typeof logicalAnd_wasm;
export let logicalNot: typeof logicalNot_wasm;
export let logicalOr: typeof logicalOr_wasm;
export let logicalXor: typeof logicalXor_wasm;
export let gelu: typeof gelu_wasm;
export let tanh: typeof tanh_wasm;
export let exp: typeof exp_wasm;
export let log: typeof log_wasm;
export let sin: typeof sin_wasm;
export let cos: typeof cos_wasm;
export let abs: typeof abs_wasm;
export let sqrt: typeof sqrt_wasm;
export let relu: typeof relu_wasm;
export let relu2: typeof relu2_wasm;
export let floor: typeof floor_wasm;
export let ceil: typeof ceil_wasm;
export let neg: typeof neg_wasm;
export let sigmoid: typeof sigmoid_wasm;
export let swiglu: typeof swiglu_wasm;
export let silu: typeof silu_wasm;
export let square: typeof square_wasm;
export let recip: typeof recip_wasm;
export let eye: typeof eye_wasm;
export let isnan: typeof isnan_wasm;
export let isinf: typeof isinf_wasm;

export let tensor: {
  (
    data: CreateTensorData,
    config: TensorOptions & OptionalShapeConfig,
  ): Parameter;
  (data: CreateTensorData, config?: TensorOptions & OptionalShapeConfig): Tensor;
};

type CreateTensorData =
  | number
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

function wrapWithLibTensor<Args extends unknown[]>(
  fn: (...args: Args) => Tensor_wasm,
): (...args: Args) => Tensor {
  return (...args: Args) => {
    return Tensor._wrap(fn(...args));
  };
}

function wrapWithParam<Args extends unknown[], O extends TensorOptions>(
  fn: (...args: [...Args, O?]) => Tensor,
): {
  (...args: [...Args, O & TensorOptions]): Parameter;
  (...args: [...Args, O?]): Tensor;
} {
  function wrapped(...args: [...Args, O & TensorOptions]): Parameter;
  function wrapped(...args: [...Args, O?]): Tensor;
  function wrapped(...args: [...Args, O?]): Tensor | Parameter {
    const maybeCfg = args[args.length - 1] as O | undefined;
    const hasCfg = typeof maybeCfg === "object" && maybeCfg !== null;

    const restArgs = (hasCfg ? args.slice(0, -1) : args) as Args;
    const configArg = (hasCfg ? (maybeCfg as O) : undefined) as O | undefined;

    const tensor = fn(...restArgs, configArg);
    return (configArg as TensorOptions | undefined)?.requiresGrad ? new Parameter(tensor) : tensor;
  }

  return wrapped;
}

function inferDTypeFromTypedArray(
  data: Float32Array | Float64Array | Int32Array | Uint32Array | Uint8Array,
): DType {
  if (data instanceof Float32Array) return float32._clone();
  if (data instanceof Float64Array) return float32._clone();
  if (data instanceof Int32Array) return int32._clone();
  if (data instanceof Uint32Array) return uint32._clone();
  if (data instanceof Uint8Array) return uint32._clone();
  return float32._clone();
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

  full = wrapWithParam(wrapWithLibTensor(full_wasm));
  cat = wrapWithLibTensor(cat_wasm);
  stack = wrapWithLibTensor(stack_wasm);
  arange = wrapWithParam(wrapWithLibTensor(arange_wasm));
  randint = wrapWithParam(wrapWithLibTensor(randint_wasm));
  randn = wrapWithParam(wrapWithLibTensor(randn_wasm));
  rand = wrapWithParam(wrapWithLibTensor(rand_wasm));
  eye = wrapWithLibTensor(eye_wasm);
  add = wrapWithLibTensor(add_wasm);
  addcdiv = wrapWithLibTensor(addcdiv_wasm);
  addcmul = wrapWithLibTensor(addcmul_wasm);
  eq = wrapWithLibTensor(eq_wasm);
  ne = wrapWithLibTensor(ne_wasm);
  gt = wrapWithLibTensor(gt_wasm);
  ge = wrapWithLibTensor(ge_wasm);
  lt = wrapWithLibTensor(lt_wasm);
  le = wrapWithLibTensor(le_wasm);
  gelu = wrapWithLibTensor(gelu_wasm);
  tanh = wrapWithLibTensor(tanh_wasm);
  exp = wrapWithLibTensor(exp_wasm);
  log = wrapWithLibTensor(log_wasm);
  sin = wrapWithLibTensor(sin_wasm);
  cos = wrapWithLibTensor(cos_wasm);
  abs = wrapWithLibTensor(abs_wasm);
  sqrt = wrapWithLibTensor(sqrt_wasm);
  relu = wrapWithLibTensor(relu_wasm);
  relu2 = wrapWithLibTensor(relu2_wasm);
  floor = wrapWithLibTensor(floor_wasm);
  ceil = wrapWithLibTensor(ceil_wasm);
  neg = wrapWithLibTensor(neg_wasm);
  sigmoid = wrapWithLibTensor(sigmoid_wasm);
  swiglu = wrapWithLibTensor(swiglu_wasm);
  silu = wrapWithLibTensor(silu_wasm);
  square = wrapWithLibTensor(square_wasm);
  recip = wrapWithLibTensor(recip_wasm);
  isnan = wrapWithLibTensor(isnan_wasm);
  isinf = wrapWithLibTensor(isinf_wasm);
  logicalAnd = wrapWithLibTensor(logicalAnd_wasm);
  logicalNot = wrapWithLibTensor(logicalNot_wasm);
  logicalOr = wrapWithLibTensor(logicalOr_wasm);
  logicalXor = wrapWithLibTensor(logicalXor_wasm);
  bernoulli = wrapWithLibTensor(bernoulli_wasm);
  zeros = wrapWithParam(wrapWithLibTensor(zeros_wasm));
  zerosLike = wrapWithParam(wrapWithLibTensor(zerosLike_wasm));
  ones = wrapWithParam(wrapWithLibTensor(ones_wasm));
  onesLike = wrapWithParam(wrapWithLibTensor(onesLike_wasm));
  tensor = wrapWithParam(
    wrapWithLibTensor(
      (
        data:
          | number
          | NestedNumberList[]
          | Uint8Array
          | Float32Array
          | Float64Array
          | Int32Array
          | Uint32Array,
        config?: TensorOptions & OptionalShapeConfig,
      ) => {
        // Infer shape here, if we're looking at a nested number list
        let shape: ShapeType | undefined = config?.shape;

        let dataShape: number[];
        if (Array.isArray(data) || typeof data === "number") {
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
          const providedShapeArray = Array.isArray(shape)
            ? shape
            : typeof shape === "number"
              ? [shape]
              : Array.from(shape);
          const shapeNumel = providedShapeArray.reduce((a, b) => a * b, 1);
          if (shapeNumel !== dataNumel) {
            throw Error(
              `Data shape [${dataShape}] can't be reshaped to provided shape [${providedShapeArray}]`,
            );
          }
        }

        // Determine effective dtype: explicit config wins; otherwise infer from input
        const isTyped =
          data instanceof Float32Array ||
          data instanceof Float64Array ||
          data instanceof Int32Array ||
          data instanceof Uint32Array ||
          data instanceof Uint8Array;

        const targetDType = config?.dtype
          ? parseDType(config.dtype)
          : isTyped
            ? inferDTypeFromTypedArray(
                data as Float32Array | Float64Array | Int32Array | Uint32Array | Uint8Array,
              )
            : float32._clone();

        // Prepare data with minimal conversions
        let castData:
          | Float32Array
          | Float64Array
          | Int32Array
          | Uint32Array
          | Uint8Array
          | number[];
        if (Array.isArray(data)) {
          const flattened = (data as unknown[]).flat(Infinity) as number[];
          // If user specified dtype, prefer constructing the corresponding typed array
          if (config?.dtype) {
            if (targetDType.isFloatingPoint) {
              // Use Float32Array for both F32 and F16 inputs
              castData = new Float32Array(flattened);
            } else if (targetDType.isSigned) {
              castData = new Int32Array(flattened);
            } else {
              castData = new Uint32Array(flattened);
            }
          } else {
            // Default to F32
            castData = flattened; // pass number[] to avoid extra JS copy
          }
        } else if (typeof data === "number") {
          if (config?.dtype) {
            if (targetDType.isFloatingPoint) {
              castData = new Float32Array([data]);
            } else if (targetDType.isSigned) {
              castData = new Int32Array([data]);
            } else {
              castData = new Uint32Array([data]);
            }
          } else {
            castData = [data];
          }
        } else if (data instanceof Uint8Array) {
          // Uint8Array is not directly handled; promote once using fast typed-array copy
          if (targetDType.isFloatingPoint) {
            const out = new Float32Array(data.length);
            out.set(data);
            castData = out;
          } else if (targetDType.isSigned) {
            const out = new Int32Array(data.length);
            out.set(data);
            castData = out;
          } else {
            const out = new Uint32Array(data.length);
            out.set(data);
            castData = out;
          }
        } else {
          // Already one of the supported typed arrays; avoid JS-side casts
          castData = data as Float32Array | Float64Array | Int32Array | Uint32Array | Uint8Array;
        }

        if (!castData) {
          throw new Error("No data to create tensor from");
        }

        return fromData(castData, shape, {
          dtype: targetDType,
          device: parseDevice(config?.device),
          requiresGrad: config?.requiresGrad ?? false,
        });
      },
    ),
  );
}
