import { Buffer, Parameter } from "@/parameter";
import { Tensor } from "@/tensor";
import { save_wasm } from "@/wasm";

export function save(stateDict: Record<string, Parameter | Buffer>) {
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
  );
}
