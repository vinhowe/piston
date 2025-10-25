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

export * from "./data";
export * from "./weak";
