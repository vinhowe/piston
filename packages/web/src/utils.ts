import { NestedNumberList } from "@/types";

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
}
