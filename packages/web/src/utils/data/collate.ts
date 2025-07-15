import { stack, tensor } from "@/globals";
import { Tensor } from "@/tensor";

const defaultCollateErrMsgFormat =
  "defaultCollate: batch must contain tensors, numbers, dicts or lists; found {}";

export type CollateOptions = {
  collateFnMap?: CollateFnMap;
};

export type CollateFunction = (batch: unknown[], options?: CollateOptions) => unknown;

export type CollateFnMap = Map<unknown, CollateFunction>;

/**
 * General collate function that handles collection type of element within each batch.
 *
 * The function also opens function registry to deal with specific element types.
 * `defaultCollateFnMap` provides default collate functions for tensors, TypedArrays, numbers,
 * strings, and other JS types.
 *
 * @param batch - a single batch to be collated
 * @param options - Optional configuration object
 * @param options.collateFnMap - Optional map from element type to the corresponding collate
 *                               function. If the element type isn't present in this map, this
 *                               function will go through each key of the map in the insertion order
 *                               to invoke the corresponding collate function if the element is an
 *                               instance of the key.
 *
 * @example
 * ```typescript
 * function collateTensorFn(batch: unknown[], options?: CollateOptions) {
 *   // Extend this function to handle batch of tensors
 *   return stack(batch as Tensor[], 0);
 * }
 * function customCollate(batch: unknown[]) {
 *   const collateMap = new Map([[Tensor, collateTensorFn]]);
 *   return collate(batch, { collateFnMap: collateMap });
 * }
 * // Extend `defaultCollate` by modifying `defaultCollateFnMap`
 * defaultCollateFnMap.set(Tensor, collateTensorFn);
 * ```
 *
 * Note:
 * Each collate function requires a positional argument for batch and an optional options object
 * with the map of collate functions as `collateFnMap`.
 */
export function collate(batch: unknown[], options?: CollateOptions): unknown {
  const elem = batch[0];
  const elemType = elem?.constructor;
  const { collateFnMap } = options || {};

  if (collateFnMap) {
    // Check for string-based type matches for primitives first (highest priority)
    const elemTypeString = typeof elem;
    if (collateFnMap.has(elemTypeString)) {
      return collateFnMap.get(elemTypeString)!(batch, options);
    }

    // Check for exact constructor type match
    if (collateFnMap.has(elemType)) {
      return collateFnMap.get(elemType)!(batch, options);
    }

    // Check for instanceof matches (only for constructor functions, lowest priority)
    for (const [collateType, collateFn] of collateFnMap) {
      if (typeof collateType === "function" && elem instanceof collateType) {
        return collateFn(batch, options);
      }
    }
  }

  // Handle plain objects
  if (isPlainObject(elem)) {
    const result: Record<string, unknown> = {};
    const keys = Object.keys(elem);

    for (const key of keys) {
      result[key] = collate(
        batch.map((d) => (d as Record<string, unknown>)[key]),
        options,
      );
    }
    return result;
  }

  // Handle arrays
  if (Array.isArray(elem)) {
    // Check that all elements in batch have consistent size
    const elemSize = elem.length;
    if (!batch.every((e) => Array.isArray(e) && e.length === elemSize)) {
      throw new Error("each element in list of batch should be of equal size");
    }

    // Transpose the batch
    const transposed: unknown[][] = [];
    for (let i = 0; i < elemSize; i++) {
      transposed.push(batch.map((e) => (e as unknown[])[i]));
    }

    return transposed.map((samples) => collate(samples, options));
  }

  throw new TypeError(
    defaultCollateErrMsgFormat.replace("{}", String(elemType?.name || typeof elem)),
  );
}

function collateTensorFn(batch: unknown[], _options?: CollateOptions): Tensor {
  return stack(batch as Tensor[], 0);
}

function collateFloat32ArrayFn(batch: unknown[], _options?: CollateOptions): Tensor {
  const arrays = batch as Float32Array[];
  // Convert each Float32Array to a tensor and stack them
  const tensors = arrays.map((arr) => tensor(arr));
  return stack(tensors, 0);
}

function collateInt32ArrayFn(batch: unknown[], _options?: CollateOptions): Tensor {
  const arrays = batch as Int32Array[];
  // Convert each Int32Array to a tensor and stack them
  const tensors = arrays.map((arr) => tensor(arr, { dtype: "int32" }));
  return stack(tensors, 0);
}

function collateUint32ArrayFn(batch: unknown[], _options?: CollateOptions): Tensor {
  const arrays = batch as Uint32Array[];
  // Convert each Uint32Array to a tensor and stack them
  const tensors = arrays.map((arr) => tensor(arr, { dtype: "uint32" }));
  return stack(tensors, 0);
}

function collateUint8ArrayFn(batch: unknown[], _options?: CollateOptions): Tensor {
  const arrays = batch as Uint8Array[];
  // Convert Uint8Array to Uint32Array (natural integer upcast)
  const tensors = arrays.map((arr) => tensor(new Uint32Array(arr), { dtype: "uint32" }));
  return stack(tensors, 0);
}

function collateFloat64ArrayFn(batch: unknown[], _options?: CollateOptions): Tensor {
  const arrays = batch as Float64Array[];
  // Convert Float64Array to Float32Array (downcast)
  const tensors = arrays.map((arr) => tensor(new Float32Array(arr)));
  return stack(tensors, 0);
}

function collateNumberFn(batch: unknown[], _options?: CollateOptions): Tensor {
  return tensor(batch as number[]);
}

function collateBigIntFn(batch: unknown[], _options?: CollateOptions): Tensor {
  // Convert bigints to numbers (with potential precision loss warning in console)
  const numbers = (batch as bigint[]).map((bi) => {
    const num = Number(bi);
    if (bi > Number.MAX_SAFE_INTEGER || bi < Number.MIN_SAFE_INTEGER) {
      console.warn(`BigInt ${bi} exceeds safe integer range, precision may be lost`);
    }
    return num;
  });
  return tensor(numbers);
}

function collateStringFn(batch: unknown[], _options?: CollateOptions): string[] {
  return batch as string[];
}

function collateBooleanFn(batch: unknown[], _options?: CollateOptions): Tensor {
  // Convert booleans to numbers (0/1) and create tensor
  const numbers = (batch as boolean[]).map((b) => (b ? 1 : 0));
  return tensor(numbers, { dtype: "int32" });
}

// Helper function to check if an object is a plain object
function isPlainObject(obj: unknown): obj is Record<string, unknown> {
  return (
    typeof obj === "object" &&
    obj !== null &&
    obj.constructor === Object &&
    Object.prototype.toString.call(obj) === "[object Object]"
  );
}

export const defaultCollateFnMap: CollateFnMap = new Map();

// Initialize the map to avoid type issues
defaultCollateFnMap.set(Tensor, collateTensorFn);
defaultCollateFnMap.set(Float32Array, collateFloat32ArrayFn);
defaultCollateFnMap.set(Int32Array, collateInt32ArrayFn);
defaultCollateFnMap.set(Uint32Array, collateUint32ArrayFn);
defaultCollateFnMap.set(Uint8Array, collateUint8ArrayFn);
defaultCollateFnMap.set(Float64Array, collateFloat64ArrayFn);
// Use string keys for primitive types since they don't have constructors we can reference
defaultCollateFnMap.set("number", collateNumberFn);
defaultCollateFnMap.set("bigint", collateBigIntFn);
defaultCollateFnMap.set("string", collateStringFn);
defaultCollateFnMap.set("boolean", collateBooleanFn);

/**
 * Take in a batch of data and put the elements within the batch into a tensor with an additional
 * outer dimension - batch size.
 *
 * The exact output type can be a Tensor, an Array of Tensors, a Collection of Tensors, or left
 * unchanged, depending on the input type.
 * This is used as the default function for collation when `batchSize` is defined in a DataLoader.
 *
 * Here is the general input type (based on the type of the element within the batch) to output type
 * mapping:
 *
 * * `Tensor` -> `Tensor` (with an added outer dimension batch size)
 * * `TypedArray` -> `Tensor`
 * * `number` -> `Tensor`
 * * `bigint` -> `Tensor` (with potential precision loss warning)
 * * `boolean` -> `Tensor`
 * * `string` -> `string[]` (unchanged)
 * * `Uint8Array` -> `Tensor` (equivalent to bytes)
 * * `Record<K, V_i>` -> `Record<K, defaultCollate([V_1, V_2, ...])>`
 * * `Array<V1_i, V2_i, ...>` -> `Array<defaultCollate([V1_1, V1_2, ...]), defaultCollate([V2_1, V2_2, ...]), ...>`
 *
 * @param batch - a single batch to be collated
 *
 * @example
 * ```typescript
 * // Example with a batch of numbers:
 * defaultCollate([0, 1, 2, 3])
 * // tensor([0, 1, 2, 3])
 *
 * // Example with a batch of strings:
 * defaultCollate(['a', 'b', 'c'])
 * // ['a', 'b', 'c']
 *
 * // Example with objects inside the batch:
 * defaultCollate([{A: 0, B: 1}, {A: 100, B: 100}])
 * // {A: tensor([0, 100]), B: tensor([1, 100])}
 *
 * // Example with arrays inside the batch:
 * defaultCollate([[0, 1], [2, 3]])
 * // [tensor([0, 2]), tensor([1, 3])]
 * ```
 */
export function defaultCollate(batch: unknown[]): unknown {
  return collate(batch, { collateFnMap: defaultCollateFnMap });
}
