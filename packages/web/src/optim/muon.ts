import { float32, zerosLike } from "@/globals";
import { Parameter } from "@/nn/parameter";
import { AdamW } from "@/optim/adamw";
import { Optimizer } from "@/optim/optimizer";
import { OptimizerParamState, ParamGroup } from "@/optim/optimizer";
import { Tensor } from "@/tensor";
import { AdamWConfig, AdamWParamGroup } from "@/types";
import { pin } from "@/utils/weak";
import { Device } from "@/wasm";

interface MuonParamState extends OptimizerParamState {
  momentumBuffer: Tensor | null;
}

export interface MuonParamGroup extends ParamGroup {
  lr: number;
  weightDecay: number;
  momentum: number;
  nsSteps: number;
  nesterov: boolean;
}

export interface MuonConfig {
  lr?: number;
  weightDecay?: number;
  momentum?: number;
  nsSteps?: number;
  nesterov?: boolean;
}

export type MuonWithAdamWParamGroup =
  | ({ optimizer: "muon" } & MuonParamGroup)
  | ({ optimizer: "adamw" } & AdamWParamGroup);

export type MuonWithAdamWconfig = {
  muon: MuonConfig;
  adamw: AdamWConfig;
};

/**
 * Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
 * We use a quintic iteration whose coefficients are selected to maximize the slope at zero.
 *
 * @param G - Input tensor to orthogonalize
 * @param steps - Number of Newton-Schulz iterations to perform
 * @returns Orthogonalized tensor
 */
function zeropowerViaNewtonSchulz5(G: Tensor, steps: number): Tensor {
  if (G.ndim < 2) {
    throw new Error("G must have at least 2 dimensions");
  }
  // Coefficients for quintic Newton-Schulz iteration
  const [a, b, c] = [3.4445, -4.775, 2.0315];

  let X = G.cast(float32);

  if (G.size(-2) > G.size(-1)) {
    X = X.mT;
  }

  // Ensure spectral norm is at most 1
  X = X.div(X.norm({ ord: "fro", dim: [-2, -1], keepdim: true }).add(1e-7));

  // Perform the Newton-Schulz iterations
  for (let i = 0; i < steps; i++) {
    const A = X.matmul(X.mT);
    // TODO: A.add(0) is a hack because the backend doesn't like it when we try to do A.matmul(A)
    // (it locks up)
    const Asquared = A.matmul(A.add(0));
    const B = A.mul(b).add(Asquared.mul(c));
    X = X.mul(a).add(B.matmul(X));
  }

  // Transpose back if needed
  if (G.size(-2) > G.size(-1)) {
    X = X.mT;
  }
  return X;
}

/**
 * Perform a single Muon update step
 *
 * @param grad - Gradient tensor
 * @param momentum - Momentum buffer tensor
 * @param beta - Momentum coefficient
 * @param nsSteps - Number of Newton-Schulz steps
 * @param nesterov - Whether to use Nesterov momentum
 * @returns Update tensor
 */
function muonUpdate(
  grad: Tensor,
  momentum: Tensor,
  beta: number = 0.95,
  nsSteps: number = 5,
  nesterov: boolean = true,
): Tensor {
  // Update momentum: momentum = momentum * beta + grad * (1 - beta)
  momentum.lerp_(grad, 1 - beta);

  // Compute update using Nesterov momentum if enabled
  let update = nesterov ? grad.lerp_(momentum, beta) : momentum;

  // For the case of conv filters
  if (update.ndim === 4) {
    update = update.view([update.size(0), -1]);
  }

  // Apply Newton-Schulz orthogonalization
  update = zeropowerViaNewtonSchulz5(update, nsSteps);

  // Scale by sqrt(max(1, rows/cols))
  update = update.mul(Math.sqrt(Math.max(1, grad.size(-2) / grad.size(-1))));

  return update;
}

/**
 * Implementation of the Muon optimizer
 *
 * Muon is a memory-efficient optimizer that uses Newton-Schulz orthogonalization
 * for preconditioning gradients. It's particularly effective for transformer models.
 */
export class Muon extends Optimizer<MuonParamState> {
  /**
   * Creates a new Muon optimizer
   *
   * @param params - Parameters or parameter groups to optimize
   * @param device - The device to perform computations on
   * @param config - Configuration options for the optimizer:
   *  - `lr` (number): Learning rate (default: 0.02)
   *  - `weightDecay` (number): Weight decay coefficient (default: 0)
   *  - `momentum` (number): Momentum factor (default: 0.95)
   */
  constructor(params: Parameter[] | ParamGroup[], device: Device, config?: Partial<MuonConfig>) {
    const {
      lr = 0.02,
      weightDecay = 0,
      momentum = 0.95,
      nsSteps = 5,
      nesterov = true,
    } = config ?? {};

    // Validate parameters
    if (!(lr >= 0)) {
      throw new Error(`Invalid learning rate: ${lr}`);
    }
    if (!(weightDecay >= 0)) {
      throw new Error(`Invalid weight_decay value: ${weightDecay}`);
    }
    if (!(momentum >= 0 && momentum < 1)) {
      throw new Error(`Invalid momentum value: ${momentum}`);
    }

    // Initialize with default options
    const defaults = {
      lr,
      weightDecay,
      momentum,
      nsSteps,
      nesterov,
    };

    super(params, device, defaults);
  }

  /**
   * Performs a single optimization step
   *
   * @param closure - Closure that reevaluates the model and returns the loss
   * @returns The loss value if closure is provided
   */
  async step(closure?: () => number): Promise<number | undefined> {
    let loss: number | undefined;

    // Call closure if provided
    if (closure) {
      loss = closure();
    }

    // Perform optimization for each parameter group
    for (const group of this.paramGroups) {
      const {
        lr = this.defaults.lr as number,
        weightDecay = this.defaults.weightDecay as number,
        momentum = this.defaults.momentum as number,
        nsSteps = this.defaults.nsSteps as number,
        nesterov = this.defaults.nesterov as boolean,
      } = group as MuonParamGroup;

      for (const param of group.params) {
        if (!param.grad) {
          // Force synchronization for parameters without gradients
          param.grad = pin(zerosLike(param));
        }

        const grad = param.grad as Tensor;

        // Get parameter state
        let state = this.state.get(param);
        if (!state) {
          state = {
            momentumBuffer: pin(zerosLike(param)),
          };
          this.state.set(param, state);
        }

        const update = muonUpdate(grad, state.momentumBuffer!, momentum, nsSteps, nesterov);

        // Apply weight decay
        if (weightDecay !== 0) {
          param.mul_(1 - lr * weightDecay);
        }

        param.add_(update.view(param.shape).mul(-lr));
      }
    }

    await this.device.markStep();

    return loss;
  }
}

type MuonWithAdamWParamState = OptimizerParamState;

export class MuonWithAdamW extends Optimizer<MuonWithAdamWParamState> {
  muon: Muon;
  adamw: AdamW;

  constructor(paramGroups: ParamGroup[], device: Device, config?: Partial<MuonWithAdamWconfig>) {
    const { muon: muonConfig, adamw: adamwConfig } = config ?? {};

    super(paramGroups, device, config ?? {});

    const muonParamGroups = paramGroups.filter(
      (p) => (p as MuonWithAdamWParamGroup).optimizer === "muon",
    );
    const adamwParamGroups = paramGroups.filter(
      (p) => (p as MuonWithAdamWParamGroup).optimizer === "adamw",
    );

    this.muon = new Muon(muonParamGroups, device, muonConfig);
    this.adamw = new AdamW(adamwParamGroups, device, adamwConfig);

    for (const paramGroup of muonParamGroups) {
      this.paramGroups.push(paramGroup);
    }
    for (const paramGroup of adamwParamGroups) {
      this.paramGroups.push(paramGroup);
    }
  }

  override addParamGroup(paramGroup: ParamGroup): void {
    if (!this.muon || !this.adamw) {
      // super constructor:
      // 1. Will attept to do addParamGroup for each param group in paramGroups
      // 2. Will throw an error on an empty list
      // So if the optimizers haven't been created yet, we assume we are in that state, and simply
      // fail silently.
      return;
    }
    const typedGroup = paramGroup as MuonWithAdamWParamGroup;
    if (typedGroup.optimizer === "muon") {
      this.muon.addParamGroup(paramGroup);
    } else if (typedGroup.optimizer === "adamw") {
      this.adamw.addParamGroup(paramGroup);
    } else {
      throw new Error(
        `Unknown optimizer: ${(typedGroup as unknown as { optimizer: string })?.optimizer}`,
      );
    }
    this.paramGroups.push(paramGroup);
  }

  /**
   * Performs a single optimization step, forwarding to the appropriate optimizer
   * based on each parameter group's optimizer configuration
   *
   * @param closure - Closure that reevaluates the model and returns the loss
   * @returns The loss value if closure is provided
   */
  async step(closure?: () => number): Promise<number | undefined> {
    let loss: number | undefined;

    // Call closure if provided
    if (closure) {
      loss = closure();
    }

    await this.muon.step(closure);
    await this.adamw.step(closure);

    return loss;
  }
}
