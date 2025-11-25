import {
  enterAutocast as enterAutocast_wasm,
  exitAutocast as exitAutocast_wasm,
  isAutocastEnabled as isAutocastEnabled_wasm,
} from "@/wasm";

/**
 * Autocast scope for automatic mixed precision.
 *
 * Uses explicit resource management (Symbol.dispose) to ensure autocast
 * is properly disabled when the scope exits.
 *
 * @example
 * ```typescript
 * {
 *   using _ = autocast();
 *   // Operations here will automatically cast FP32 inputs to FP16
 *   const output = model.forward(input);
 * }
 * // Autocast is disabled after the block
 * ```
 */
export class AutocastScope implements Disposable {
  private _wasEnabled: boolean;

  constructor() {
    // Store previous state to support nesting
    this._wasEnabled = isAutocastEnabled_wasm();
    enterAutocast_wasm();
  }

  [Symbol.dispose](): void {
    if (!this._wasEnabled) {
      exitAutocast_wasm();
    }
  }
}

/**
 * Creates an autocast scope for automatic mixed precision.
 *
 * Within the scope, participating tensor operations (matmul, gemm, softmax,
 * layer_norm, group_norm, rms_norm, conv1d) will automatically cast FP32
 * inputs to FP16 for faster computation.
 *
 * @returns An AutocastScope that should be used with the `using` syntax
 *
 * @example
 * ```typescript
 * {
 *   using _ = autocast();
 *   const output = model.forward(input);  // matmul etc. auto-cast to FP16
 * }
 * ```
 */
export function autocast(): AutocastScope {
  return new AutocastScope();
}

/**
 * Returns whether autocast mode is currently enabled.
 */
export function isAutocastEnabled(): boolean {
  return isAutocastEnabled_wasm();
}

