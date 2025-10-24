import { Optimizer } from "./optimizer";

/**
 * State dictionary for learning rate scheduler serialization
 */
export type SchedulerStateDict<T> = {
  lastEpoch: number;
  lastLr?: number[];
} & T;

/**
 * Configuration for schedulers that use step-based decay
 */
export interface StepConfig {
  stepSize: number;
  gamma: number;
}

/**
 * Configuration for schedulers that use multiple milestones
 */
export interface MultiStepConfig {
  milestones: number[];
  gamma: number;
}

/**
 * Configuration for cosine annealing scheduler
 */
export interface CosineAnnealingConfig {
  tMax: number;
  etaMin?: number;
}

/**
 * Configuration for exponential decay scheduler
 */
export interface ExponentialConfig {
  gamma: number;
}

/**
 * Configuration for linear scheduler
 */
export interface LinearConfig {
  startFactor: number;
  endFactor: number;
  totalIters: number;
}

/**
 * Configuration for constant scheduler
 */
export interface ConstantConfig {
  factor: number;
  totalIters: number;
}

/**
 * Base class for all learning rate schedulers
 *
 * This class provides the interface and common functionality for all learning rate schedulers.
 * It follows PyTorch's LRScheduler design but adapted for TypeScript.
 */
export abstract class LRScheduler<T> {
  protected optimizer: Optimizer;
  protected lastEpoch: number;
  protected baseLrs: number[];
  protected lastLr: number[];
  protected stepCount: number = 0;
  private _initialized: boolean = false;

  /**
   * Creates a new learning rate scheduler
   *
   * @param optimizer - The optimizer to schedule
   * @param lastEpoch - The index of last epoch, used for resuming training
   */
  constructor(optimizer: Optimizer, lastEpoch: number = -1) {
    this.optimizer = optimizer;
    this.lastEpoch = lastEpoch;

    // Initialize base learning rates
    if (lastEpoch === -1) {
      this.baseLrs = this.optimizer.paramGroups.map((group) => {
        const initial = group.lr!;
        // Persist initialLr on the param group for resume consistency
        group.initialLr = initial;
        return initial;
      });
    } else {
      this.baseLrs = this.optimizer.paramGroups.map((group) => {
        if (!group.initialLr) {
          throw new Error(
            "param 'initialLr' is not specified in param groups when resuming an optimizer",
          );
        }
        return group.initialLr as number;
      });
    }

    this.lastLr = this.baseLrs.slice();
  }

  /**
   * Ensure the scheduler is initialized before first use
   */
  private ensureInitialized(): void {
    if (this._initialized) return;

    this.stepCount = 0;
    // Doing this here prevents infinite recursion
    this._initialized = true;
    this.step();
  }

  /**
   * Return the state of the scheduler as a dictionary
   */
  stateDict(): SchedulerStateDict<T> {
    const state: Record<string, unknown> = {
      lastEpoch: this.lastEpoch,
      lastLr: this.lastLr.slice(),
    };

    // Add any additional state from subclasses
    for (const [key, value] of Object.entries(this)) {
      if (
        key !== "optimizer" &&
        key !== "baseLrs" &&
        key !== "lastLr" &&
        key !== "_initialized" &&
        key !== "stepCount"
      ) {
        state[key] = value;
      }
    }

    return state as SchedulerStateDict<T>;
  }

  /**
   * Load the scheduler's state
   *
   * @param stateDict - Scheduler state dictionary
   */
  loadStateDict(stateDict: SchedulerStateDict<T>): void {
    const { lastLr, ...rest } = stateDict;
    Object.assign(this, rest);

    // Restore lastLr for correct chainable updates; if missing, sync from optimizer groups
    if (Array.isArray(lastLr)) {
      this.lastLr = lastLr.slice();
    } else {
      this.lastLr = this.optimizer.paramGroups.map((g) => g.lr!);
    }

    // Mark initialized to avoid performing an extra initial step after loading state
    this._initialized = true;
  }

  /**
   * Return last computed learning rate by current scheduler
   */
  getLastLr(): number[] {
    this.ensureInitialized();
    return this.lastLr.slice();
  }

  /**
   * Compute learning rate using chainable form of the scheduler
   */
  abstract getLr(): number[];

  /**
   * Perform a scheduler step
   */
  step(): void {
    this.ensureInitialized();
    this.stepCount += 1;
    this.lastEpoch += 1;
    const values = this.getLr();

    // Update learning rates in optimizer
    for (let i = 0; i < this.optimizer.paramGroups.length; i++) {
      this.optimizer.paramGroups[i].lr = values[i];
    }

    this.lastLr = values.slice();
  }

  /**
   * Get current learning rate (alias for getLr for compatibility)
   */
  getCurrentLr(): number[] {
    this.ensureInitialized();
    return this.getLr();
  }
}

/**
 * Keeps learning rate constant for a number of epochs
 *
 * Multiplies the learning rate of each parameter group by a given factor
 * for a specified number of epochs.
 */
export class ConstantLR extends LRScheduler<ConstantConfig> {
  private factor: number;
  private totalIters: number;

  /**
   * @param optimizer - Wrapped optimizer
   * @param factor - Factor to multiply learning rate by (default: 1/3)
   * @param totalIters - Number of epochs to keep the factor applied (default: 5)
   * @param lastEpoch - The index of last epoch (default: -1)
   */
  constructor(
    optimizer: Optimizer,
    factor: number = 1 / 3,
    totalIters: number = 5,
    lastEpoch: number = -1,
  ) {
    super(optimizer, lastEpoch);
    this.factor = factor;
    this.totalIters = totalIters;
  }

  getLr(): number[] {
    if (this.lastEpoch === 0) {
      return this.getLastLr().map((lr) => lr * this.factor);
    }

    if (this.lastEpoch !== this.totalIters) {
      return this.getLastLr().slice();
    }

    return this.getLastLr().map((lr) => lr * (1 / this.factor));
  }
}

/**
 * Linearly adjusts learning rate between two boundaries over a number of epochs
 *
 * The learning rate is multiplied by a factor that linearly changes from
 * startFactor to endFactor over totalIters epochs.
 */
export class LinearLR extends LRScheduler<LinearConfig> {
  private startFactor: number;
  private endFactor: number;
  private totalIters: number;

  /**
   * @param optimizer - Wrapped optimizer
   * @param startFactor - Starting factor for learning rate (default: 1/3)
   * @param endFactor - Ending factor for learning rate (default: 1.0)
   * @param totalIters - Number of epochs over which to linearly change (default: 5)
   * @param lastEpoch - The index of last epoch (default: -1)
   */
  constructor(
    optimizer: Optimizer,
    startFactor: number = 1 / 3,
    endFactor: number = 1.0,
    totalIters: number = 5,
    lastEpoch: number = -1,
  ) {
    if (startFactor > 1.0 || startFactor <= 0) {
      throw new Error(
        "Starting multiplicative factor expected to be greater than 0 and less or equal to 1.",
      );
    }

    if (endFactor > 1.0 || endFactor < 0) {
      throw new Error("Ending multiplicative factor expected to be between 0 and 1.");
    }

    super(optimizer, lastEpoch);
    this.startFactor = startFactor;
    this.endFactor = endFactor;
    this.totalIters = totalIters;
  }

  getLr(): number[] {
    if (this.lastEpoch === 0) {
      console.log(
        "startFactor",
        this.startFactor,
        this.getLastLr(),
        this.getLastLr().map((lr) => lr * this.startFactor),
      );
      return this.getLastLr().map((lr) => lr * this.startFactor);
    }

    if (this.lastEpoch > this.totalIters) {
      return this.getLastLr().map((lr) => lr);
    }

    const factor =
      1 +
      (this.endFactor - this.startFactor) /
        (this.totalIters * this.startFactor +
          (this.lastEpoch - 1) * (this.endFactor - this.startFactor));
    return this.getLastLr().map((lr) => lr * factor);
  }
}

/**
 * Decays the learning rate by gamma every stepSize epochs
 *
 * When last_epoch=-1, sets initial lr as lr.
 */
export class StepLR extends LRScheduler<StepConfig> {
  private stepSize: number;
  private gamma: number;

  /**
   * @param optimizer - Wrapped optimizer
   * @param stepSize - Period of learning rate decay
   * @param gamma - Multiplicative factor of learning rate decay (default: 0.1)
   * @param lastEpoch - The index of last epoch (default: -1)
   */
  constructor(optimizer: Optimizer, stepSize: number, gamma: number = 0.1, lastEpoch: number = -1) {
    super(optimizer, lastEpoch);
    this.stepSize = stepSize;
    this.gamma = gamma;
  }

  getLr(): number[] {
    if (this.lastEpoch === 0 || this.lastEpoch % this.stepSize !== 0) {
      return this.getLastLr();
    }
    return this.getLastLr().map((lr) => lr * this.gamma);
  }
}

/**
 * Decays the learning rate by gamma once the number of epochs reaches one of the milestones
 *
 * When last_epoch=-1, sets initial lr as lr.
 */
export class MultiStepLR extends LRScheduler<MultiStepConfig> {
  private milestones: Set<number>;
  private gamma: number;

  /**
   * @param optimizer - Wrapped optimizer
   * @param milestones - List of epoch indices at which to decay learning rate
   * @param gamma - Multiplicative factor of learning rate decay (default: 0.1)
   * @param lastEpoch - The index of last epoch (default: -1)
   */
  constructor(
    optimizer: Optimizer,
    milestones: number[],
    gamma: number = 0.1,
    lastEpoch: number = -1,
  ) {
    super(optimizer, lastEpoch);
    this.milestones = new Set(milestones);
    this.gamma = gamma;
  }

  getLr(): number[] {
    if (this.lastEpoch === 0 || !this.milestones.has(this.lastEpoch)) {
      return this.getLastLr();
    }
    return this.getLastLr().map((lr) => lr * this.gamma);
  }
}

/**
 * Decays the learning rate by gamma every epoch
 */
export class ExponentialLR extends LRScheduler<ExponentialConfig> {
  private gamma: number;

  /**
   * @param optimizer - Wrapped optimizer
   * @param gamma - Multiplicative factor of learning rate decay
   * @param lastEpoch - The index of last epoch (default: -1)
   */
  constructor(optimizer: Optimizer, gamma: number, lastEpoch: number = -1) {
    super(optimizer, lastEpoch);
    this.gamma = gamma;
  }

  getLr(): number[] {
    if (this.lastEpoch === 0) {
      return this.getLastLr().slice();
    }
    return this.getLastLr().map((lr) => lr * this.gamma);
  }
}

/**
 * Set the learning rate using a cosine annealing schedule
 *
 * The learning rate is annealed from the initial lr to etaMin using a cosine function.
 */
export class CosineAnnealingLR extends LRScheduler<CosineAnnealingConfig> {
  private tMax: number;
  private etaMin: number;

  /**
   * @param optimizer - Wrapped optimizer
   * @param tMax - Maximum number of iterations
   * @param etaMin - Minimum learning rate (default: 0)
   * @param lastEpoch - The index of last epoch (default: -1)
   */
  constructor(optimizer: Optimizer, tMax: number, etaMin: number = 0, lastEpoch: number = -1) {
    super(optimizer, lastEpoch);
    this.tMax = tMax;
    this.etaMin = etaMin;
  }

  getLr(): number[] {
    if (this.lastEpoch === 0) {
      return this.getLastLr().slice();
    } else if (this.lastEpoch === this.tMax) {
      return this.getLastLr().map(() => this.etaMin);
    } else if ((this.lastEpoch - 1 - this.tMax) % (2 * this.tMax) === 0) {
      const lastLr = this.getLastLr();
      return lastLr.map(
        (lr, i) => lr + ((lastLr[i] - this.etaMin) * (1 - Math.cos(Math.PI / this.tMax))) / 2,
      );
    }

    return this.getLastLr().map(
      (lr) =>
        ((1 + Math.cos((Math.PI * this.lastEpoch) / this.tMax)) /
          (1 + Math.cos((Math.PI * (this.lastEpoch - 1)) / this.tMax))) *
          (lr - this.etaMin) +
        this.etaMin,
    );
  }
}
