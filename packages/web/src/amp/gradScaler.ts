import { isinf, isnan, stack } from "@/globals";
import { Optimizer } from "@/optim/optimizer";
import { Tensor } from "@/tensor";

/**
 * Configuration for GradScaler
 * Matches PyTorch's GradScaler parameters in camelCase
 */
export interface GradScalerConfig {
  /** Initial scale factor (default: 65536.0) */
  initScale?: number;
  /** Factor by which the scale is multiplied during growth (default: 2.0) */
  growthFactor?: number;
  /** Factor by which the scale is multiplied during backoff (default: 0.5) */
  backoffFactor?: number;
  /** Number of consecutive non-inf/nan iterations before growing scale (default: 2000) */
  growthInterval?: number;
  /** Whether grad scaling is enabled (default: true) */
  enabled?: boolean;
}

/**
 * State dictionary for GradScaler serialization
 */
export interface GradScalerStateDict {
  scale: number;
  growthTracker: number;
}

/**
 * GradScaler for automatic mixed precision training.
 *
 * Gradient scaling helps prevent underflow when training with mixed precision (FP16).
 * It multiplies the loss by a scale factor before backward pass, then unscales
 * gradients before the optimizer step.
 *
 * @example
 * ```typescript
 * const scaler = new GradScaler();
 *
 * // Training loop
 * {
 *   using _ = autocast();
 *   const output = model.forward(input);
 *   const loss = criterion(output, target);
 * }
 *
 * // Scale loss and backward
 * const scaledLoss = scaler.scale(loss);
 * await scaledLoss.backward();
 *
 * // Unscale, step, and update
 * await scaler.step(optimizer);
 * scaler.update();
 *
 * optimizer.zeroGrad();
 * ```
 */
export class GradScaler {
  private _scale: number;
  private _growthFactor: number;
  private _backoffFactor: number;
  private _growthInterval: number;
  private _enabled: boolean;
  private _growthTracker: number;
  private _foundInfOrNan: boolean;
  private _lastNonfiniteCount: number;

  /**
   * Creates a new GradScaler
   *
   * @param config - Configuration options
   */
  constructor(config: GradScalerConfig = {}) {
    const {
      initScale = 65536.0,
      growthFactor = 2.0,
      backoffFactor = 0.5,
      growthInterval = 2000,
      enabled = true,
    } = config;

    if (growthFactor < 1.0) {
      throw new Error("growthFactor must be >= 1.0");
    }
    if (backoffFactor > 1.0) {
      throw new Error("backoffFactor must be <= 1.0");
    }
    if (growthInterval < 1) {
      throw new Error("growthInterval must be >= 1");
    }

    this._scale = initScale;
    this._growthFactor = growthFactor;
    this._backoffFactor = backoffFactor;
    this._growthInterval = growthInterval;
    this._enabled = enabled;
    this._growthTracker = 0;
    this._foundInfOrNan = false;
    this._lastNonfiniteCount = 0;
  }

  /**
   * Scales the loss tensor by the current scale factor.
   *
   * @param loss - The loss tensor to scale
   * @returns Scaled loss tensor
   */
  scale(loss: Tensor): Tensor {
    if (!this._enabled) {
      return loss;
    }
    return loss.mul(this._scale);
  }

  /**
   * Unscales gradients in place by dividing by the scale factor.
   * Also checks for inf/nan values in the gradients.
   *
   * @param optimizer - The optimizer whose parameter gradients to unscale
   * @returns Promise that resolves to true if inf/nan was found
   */
  async unscale(optimizer: Optimizer): Promise<boolean> {
    if (!this._enabled) {
      return false;
    }

    const invScale = 1.0 / this._scale;

    this._lastNonfiniteCount = 0;

    // Collect all found_inf tensors to sum at the end
    const foundInfTensors: Tensor[] = [];

    // Process all parameter gradients
    for (const group of optimizer.paramGroups) {
      for (const param of group.params) {
        const grad = param.grad;
        if (grad) {
          // Check for inf/nan before unscaling
          const hasNan = isnan(grad).sum();
          const hasInf = isinf(grad).sum();
          foundInfTensors.push(hasNan, hasInf);

          // Unscale the gradient
          param.grad = grad.mul(invScale);
        }
      }
    }

    const totalInfNan = Math.round(await (await stack(foundInfTensors).sum().to("cpu")).item());

    this._lastNonfiniteCount = Math.round(totalInfNan);

    const foundInfOrNan = totalInfNan > 0;

    if (foundInfOrNan) {
      this._foundInfOrNan = true;
    }

    return foundInfOrNan;
  }

  /**
   * Performs the optimizer step if gradients are valid (no inf/nan).
   *
   * This method:
   * 1. Unscales gradients (if not already done)
   * 2. Checks for inf/nan
   * 3. Only calls optimizer.step() if gradients are valid
   *
   * @param optimizer - The optimizer to step
   * @returns Promise that resolves to the loss value if step was taken, undefined otherwise
   */
  async step(optimizer: Optimizer): Promise<number | undefined> {
    if (!this._enabled) {
      return optimizer.step();
    }

    // Unscale gradients and check for inf/nan
    await this.unscale(optimizer);

    if (this._foundInfOrNan) {
      // Skip step if inf/nan was found
      return undefined;
    }

    return optimizer.step();
  }

  /**
   * Updates the scale factor based on whether inf/nan was encountered.
   *
   * Should be called after each training iteration. If inf/nan was found,
   * the scale is reduced. Otherwise, after growthInterval clean iterations,
   * the scale is increased.
   */
  update(): void {
    if (!this._enabled) {
      return;
    }

    if (this._foundInfOrNan) {
      // Backoff: reduce scale
      this._scale *= this._backoffFactor;
      this._growthTracker = 0;
    } else {
      // No inf/nan: increment tracker
      this._growthTracker += 1;

      if (this._growthTracker >= this._growthInterval) {
        // Grow scale
        this._scale *= this._growthFactor;
        this._growthTracker = 0;
      }
    }

    // Reset found flag for next iteration
    this._foundInfOrNan = false;
  }

  /**
   * Returns the current scale factor
   */
  getScale(): number {
    return this._scale;
  }

  /**
   * Sets the scale factor
   *
   * @param scale - The new scale factor
   */
  setScale(scale: number): void {
    this._scale = scale;
  }

  /**
   * Returns whether the scaler is enabled
   */
  isEnabled(): boolean {
    return this._enabled;
  }

  /**
   * Returns the NaN count from the last unscale operation
   */
  getLastNonfiniteCount(): number {
    return this._lastNonfiniteCount;
  }

  /**
   * Returns the state of the scaler as a dictionary for serialization
   */
  stateDict(): GradScalerStateDict {
    return {
      scale: this._scale,
      growthTracker: this._growthTracker,
    };
  }

  /**
   * Loads scaler state from a dictionary
   *
   * @param stateDict - The state dictionary to load
   */
  loadStateDict(stateDict: GradScalerStateDict): void {
    this._scale = stateDict.scale;
    this._growthTracker = stateDict.growthTracker;
  }
}
