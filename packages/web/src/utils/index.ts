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

export * from "./data";
