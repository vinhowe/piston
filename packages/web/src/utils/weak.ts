import { FunctionModeGuard, PistonFunctionMode } from "@/function";
import { Tensor } from "@/tensor";
import { __pistonActiveTensors, Tensor_wasm } from "@/wasm";

import { forEachTensorDeep } from ".";

export interface WeakTensorFunctionModeOptions {
  label?: string;
  trackLeaks?: boolean;
}

export class WeakTensorFunctionMode extends PistonFunctionMode {
  private weakTensors: Map<number, Tensor_wasm>;
  public readonly label?: string;
  private readonly trackLeaks: boolean;
  private allocatedPtrsBefore: Set<number>;
  private allocatedIdsBefore: Set<number>;
  private pinnedPtrs: Set<number>;

  constructor({ label, trackLeaks = false }: WeakTensorFunctionModeOptions = {}) {
    super();
    this.weakTensors = new Map<number, Tensor_wasm>();
    this.label = label;
    this.trackLeaks = trackLeaks;
    if (this.trackLeaks) {
      const beforeMap = __pistonActiveTensors();
      const beforeList = Array.from(beforeMap.values()).flat();
      this.allocatedPtrsBefore = new Set<number>(beforeList.map((t) => t.__pistonStrongId));
      this.allocatedIdsBefore = new Set<number>(beforeList.map((t) => t.id));
    } else {
      this.allocatedPtrsBefore = new Set<number>();
      this.allocatedIdsBefore = new Set<number>();
    }
    this.pinnedPtrs = new Set<number>();
  }

  _pistonFunction<T>(
    func: <FT>(...args: unknown[]) => FT | Promise<FT>,
    _types: unknown[],
    args: unknown[],
    kwargs: Record<string, unknown>,
  ): T | Promise<T> {
    const after = (result: T) => {
      if (result instanceof Tensor_wasm) {
        this.weakTensors.set((result as unknown as { __wbg_ptr: number }).__wbg_ptr, result);
      }
      if (this.trackLeaks) this.logLeaks(false);
      return result;
    };

    const result = func(...args, kwargs);

    if (result instanceof Promise) {
      return result.then(after);
    }

    return after(result as T);
  }

  private logLeaks(committed: boolean = false) {
    if (!this.trackLeaks) return;
    const allocatedAfterMap = __pistonActiveTensors();
    const allocatedAfterList = Array.from(allocatedAfterMap.values()).flat();
    const leakedPtrs = new Set<number>();
    const newLeakedIds = new Set<number>();
    const leakedIds = new Set<number>();
    const pendingWeakStrongIds = committed
      ? undefined
      : new Set<number>(Array.from(this.weakTensors.values()).map((t) => t.__pistonStrongId));
    for (const tensor of allocatedAfterList) {
      const strongId = tensor.__pistonStrongId;
      const isPendingWeak = !committed && pendingWeakStrongIds?.has(strongId);
      if (
        !this.allocatedPtrsBefore.has(strongId) &&
        !this.pinnedPtrs.has(strongId) &&
        !isPendingWeak
      ) {
        leakedPtrs.add(strongId);
        const id = tensor.id;
        leakedIds.add(id);
        if (!this.allocatedIdsBefore.has(id)) {
          newLeakedIds.add(id);
        }
      }
    }

    const leakedCount = leakedPtrs.size;
    if (leakedCount > 0) {
      console.warn(
        `Weak tensor mode leaked tensors [total=${leakedCount}] [new=${newLeakedIds.size}]` +
          ` [label=${JSON.stringify(this.label)}]` +
          ` [leakedIds=${JSON.stringify(Array.from(leakedIds).toSorted())}]` +
          ` [newLeakedIds=${JSON.stringify(Array.from(newLeakedIds).toSorted())}]`,
      );
    }
  }

  pin<T>(input: T): T {
    forEachTensorDeep(input, (tensor) => {
      this.weakTensors.delete((tensor as unknown as { __wbg_ptr: number }).__wbg_ptr);
      this.pinnedPtrs.add(tensor.__pistonStrongId);
    });
    return input;
  }

  markWeak<T>(input: T) {
    forEachTensorDeep(input, (tensor) => {
      this.weakTensors.set((tensor as unknown as { __wbg_ptr: number }).__wbg_ptr, tensor);
    });
    return input;
  }

  [Symbol.dispose]() {
    super[Symbol.dispose]();
    this.weakTensors.forEach((tensor) => tensor.__pistonDrop());
    if (this.trackLeaks) this.logLeaks(true);
  }
}

export function pin<T>(input: T): T {
  using guard = new FunctionModeGuard();
  const mode = guard.mode;
  if (mode instanceof WeakTensorFunctionMode) {
    return mode.pin(input);
  }
  // Fine to silently ignore; default "strong" mode will pin anyway
  return input;
}

export function markWeak<T>(input: T) {
  using guard = new FunctionModeGuard();
  const mode = guard.mode;
  if (mode instanceof WeakTensorFunctionMode) {
    mode.markWeak(input);
  }
}

export async function weak<T>(
  input: (mode: WeakTensorFunctionMode) => Promise<T>,
  options?: WeakTensorFunctionModeOptions,
): Promise<T>;
export function weak<T>(
  input: (mode: WeakTensorFunctionMode) => T,
  options?: WeakTensorFunctionModeOptions,
): T;
export function weak<T>(input: T, options?: WeakTensorFunctionModeOptions): T;
export function weak<T>(
  input: ((mode: WeakTensorFunctionMode) => T | Promise<T>) | T,
  options?: WeakTensorFunctionModeOptions,
): T | Promise<T> {
  const weakMode = new WeakTensorFunctionMode(options);

  const after = (result: T) => {
    try {
      if (result instanceof Tensor) {
        weakMode.pin(result);
        return result;
      }

      forEachTensorDeep(result, (tensor) => weakMode.pin(tensor));

      return result;
    } finally {
      weakMode[Symbol.dispose]();
    }
  };

  if (typeof input === "function") {
    const fn = input as (mode: WeakTensorFunctionMode) => T | Promise<T>;
    try {
      const fnOutput = fn(weakMode);
      if (fnOutput instanceof Promise) {
        return fnOutput.then(after);
      }

      return after(fnOutput);
    } catch (e) {
      weakMode[Symbol.dispose]();
      throw e;
    }
  }

  return after(input as T);
}
