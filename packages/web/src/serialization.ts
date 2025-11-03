import { Buffer, Parameter } from "@/nn/parameter";
import { Tensor } from "@/tensor";
import { Device, load_wasm, save_wasm } from "@/wasm";

export function save(stateDict: Record<string, Parameter | Buffer>, extra?: unknown) {
  return save_wasm(
    Object.fromEntries(
      Object.entries(stateDict).map(([name, paramOrBuffer]) => {
        if (paramOrBuffer instanceof Buffer || paramOrBuffer instanceof Parameter) {
          return [name, Tensor._unwrap(paramOrBuffer)];
        }
        throw new TypeError(
          `Cannot save '${
            (paramOrBuffer as unknown)?.constructor?.name || typeof paramOrBuffer
          }' as parameter '${name}'`,
        );
      }),
    ),
    extra ? JSON.stringify(extra) : undefined,
  );
}

export function load(
  bytes: Uint8Array,
  mapDevice?: Device,
): { state: Record<string, Tensor>; extra?: unknown } {
  const result = load_wasm(bytes, mapDevice?._clone());
  return {
    state: Object.fromEntries(
      Object.entries(result.state).map(([key, value]) => [key, Tensor._wrap(value)]),
    ),
    extra: result.extra,
  };
}
