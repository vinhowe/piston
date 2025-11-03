import { Tensor } from "@/tensor";

/**
 * A handle which provides the capability to remove a hook.
 *
 * This class generates unique IDs for hooks and provides methods to cleanly
 * remove hooks from their associated collections.
 */
export class RemovableHandle {
  /** The unique ID of this handle */
  id: number;

  private hooksMap: WeakRef<Map<number, unknown>>;
  private static nextId: number = 0;

  /**
   * Creates a new removable handle
   *
   * @param hooksMap - The primary map of hooks indexed by hook ID
   */
  constructor(hooksMap: Map<number, unknown>) {
    // Assign a unique ID from the static counter
    this.id = RemovableHandle.nextId++;

    // Store weak references to avoid potential memory leaks
    this.hooksMap = new WeakRef(hooksMap);

    // Set this ID in the hooks map
    hooksMap.set(this.id, null);
  }

  /**
   * Removes the hook from all associated collections
   */
  remove(): void {
    // Get the actual map from the weak reference
    const hooksMap = this.hooksMap.deref();
    if (hooksMap && hooksMap.has(this.id)) {
      hooksMap.delete(this.id);
    }
  }

  [Symbol.dispose]() {
    this.remove();
  }
}

export function forEachTensorDeep(
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

export function cloneReplaceTensorsDeep(
  value: unknown,
  replacer: (t: Tensor) => unknown,
  seen: WeakMap<object, unknown> = new WeakMap(),
): unknown {
  if (value instanceof Tensor) {
    return replacer(value);
  }
  if (value === null) return value;

  const t = typeof value;
  if (t !== "object" && t !== "function") return value;

  const obj = value as object;
  const cached = seen.get(obj);
  if (cached) return cached;

  if (Array.isArray(obj)) {
    const out: unknown[] = new Array(obj.length);
    seen.set(obj, out);
    for (let i = 0; i < obj.length; i++) {
      out[i] = cloneReplaceTensorsDeep(obj[i], replacer, seen);
    }
    return out;
  }
  if (obj instanceof Map) {
    const out = new Map<unknown, unknown>();
    seen.set(obj, out);
    for (const [k, v] of obj) {
      const nk = cloneReplaceTensorsDeep(k, replacer, seen);
      const nv = cloneReplaceTensorsDeep(v, replacer, seen);
      out.set(nk, nv);
    }
    return out;
  }
  if (obj instanceof Set) {
    const out = new Set<unknown>();
    seen.set(obj, out);
    for (const v of obj) {
      out.add(cloneReplaceTensorsDeep(v, replacer, seen));
    }
    return out;
  }
  if (
    ArrayBuffer.isView(obj) ||
    obj instanceof ArrayBuffer ||
    obj instanceof Date ||
    obj instanceof RegExp
  ) {
    return obj;
  }

  const out = Object.create(Object.getPrototypeOf(obj));
  seen.set(obj, out);
  for (const key of Reflect.ownKeys(obj)) {
    const desc = Object.getOwnPropertyDescriptor(obj, key);
    if (!desc) continue;
    if ("value" in desc) {
      const newValue = cloneReplaceTensorsDeep(
        (obj as Record<PropertyKey, unknown>)[key as keyof typeof obj],
        replacer,
        seen,
      );
      Object.defineProperty(out, key, { ...desc, value: newValue });
    } else {
      Object.defineProperty(out, key, desc);
    }
  }
  return out;
}

export * from "./data";
export * from "./weak";
