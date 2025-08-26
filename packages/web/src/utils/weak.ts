import { FunctionModeGuard, PistonFunctionMode } from "@/function";
import { Tensor } from "@/tensor";
import { __pistonActiveTensors, Tensor_wasm } from "@/wasm";

function forEachTensorDeep(
  value: unknown,
  visit: (t: Tensor) => void,
  seen: WeakSet<object> = new WeakSet(),
): void {
  if (value instanceof Tensor) {
    visit(value);
    return;
  }
  if (value === null) return;

  const t = typeof value;
  if (t !== "object" && t !== "function") return;

  const obj = value as object;
  if (seen.has(obj)) return;
  seen.add(obj);

  if (Array.isArray(obj)) {
    for (const v of obj) forEachTensorDeep(v, visit, seen);
    return;
  }
  if (obj instanceof Map) {
    for (const [k, v] of obj) {
      forEachTensorDeep(k, visit, seen);
      forEachTensorDeep(v, visit, seen);
    }
    return;
  }
  if (obj instanceof Set) {
    for (const v of obj) forEachTensorDeep(v, visit, seen);
    return;
  }
  if (
    ArrayBuffer.isView(obj) ||
    obj instanceof ArrayBuffer ||
    obj instanceof Date ||
    obj instanceof RegExp
  ) {
    return;
  }

  // Own props only; avoid calling getters
  for (const key of Reflect.ownKeys(obj)) {
    const desc = Object.getOwnPropertyDescriptor(obj, key);
    if (!desc || !("value" in desc)) continue;
    forEachTensorDeep(obj[key as keyof typeof obj], visit, seen);
  }
}

interface WeakTensorFunctionModeOptions {
  label?: string;
  trackLeaks?: boolean;
}

class WeakTensorFunctionMode extends PistonFunctionMode {
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

  pinTensor(tensor: Tensor) {
    this.weakTensors.delete((tensor as unknown as { __wbg_ptr: number }).__wbg_ptr);
    this.pinnedPtrs.add(tensor.__pistonStrongId);
  }

  [Symbol.dispose]() {
    super[Symbol.dispose]();
    this.weakTensors.forEach((tensor) => tensor.__pistonDrop());
    if (this.trackLeaks) this.logLeaks(true);
  }
}

export function pin(tensor: Tensor) {
  using guard = new FunctionModeGuard();
  const mode = guard.mode;
  if (mode instanceof WeakTensorFunctionMode) {
    mode.pinTensor(tensor);
  }
  // Fine to silently ignore; default "strong" mode will pin anyway
  return tensor;
}

export async function weak<T>(
  input: () => Promise<T>,
  options?: WeakTensorFunctionModeOptions,
): Promise<T>;
export function weak<T>(input: () => T, options?: WeakTensorFunctionModeOptions): T;
export function weak<T>(input: T, options?: WeakTensorFunctionModeOptions): T;
export function weak<T>(
  input: (() => T | Promise<T>) | T,
  options?: WeakTensorFunctionModeOptions,
): T | Promise<T> {
  const weakMode = new WeakTensorFunctionMode(options);

  const after = (result: T) => {
    try {
      if (result instanceof Tensor) {
        weakMode.pinTensor(result);
        return result;
      }

      forEachTensorDeep(result, (tensor) => weakMode.pinTensor(tensor));

      return result;
    } finally {
      weakMode[Symbol.dispose]();
    }
  };

  if (typeof input === "function") {
    const fn = input as () => T | Promise<T>;
    try {
      const fnOutput = fn();
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
