import { zerosLike } from "@/globals";
import { Parameter } from "@/nn/parameter";
import { Optimizer } from "@/optim/optimizer";
import { OptimizerParamState, ParamGroup } from "@/optim/optimizer";
import { Tensor } from "@/tensor";
import { pin } from "@/utils/weak";
import { Device } from "@/wasm";

interface AdamParamState extends OptimizerParamState {
  step: number;
  expAvg: Tensor | null;
  expAvgSq: Tensor | null;
  maxExpAvgSq?: Tensor | null;
}

export interface AdamParamGroup extends ParamGroup {
  lr?: number;
  betas?: [number, number];
  eps?: number;
  weightDecay?: number;
  amsgrad?: boolean;
}

export interface AdamConfig {
  lr: number;
  betas: [number, number];
  eps: number;
  weightDecay: number;
  amsgrad: boolean;
}

/**
 * Implementation of the Adam optimizer (based on original PyTorch implementation)
 *
 * The original Adam algorithm was proposed in "Adam: A Method for Stochastic Optimization".
 */
export class Adam extends Optimizer<AdamParamState, AdamConfig, AdamParamGroup> {
  /**
   * Creates a new Adam optimizer
   *
   * @param params - Parameters or parameter groups to optimize
   * @param device - The device to perform computations on
   * @param config - Configuration options for the optimizer:
   *  - `lr` (number): Learning rate (default: 1e-3)
   *  - `betas` ([number, number]): Coefficients for computing running averages of gradient and its
   *     square (default: [0.9, 0.999])
   *  - `eps` (number): Term added to denominator for numerical stability (default: 1e-8)
   *  - `weightDecay` (number): Weight decay (L2 penalty) (default: 0)
   *  - `amsgrad` (boolean): Whether to use the AMSGrad variant (default: false)
   */
  constructor(params: Parameter[] | ParamGroup[], device: Device, config?: Partial<AdamConfig>) {
    const {
      lr = 1e-3,
      betas = [0.9, 0.999],
      eps = 1e-8,
      weightDecay = 0,
      amsgrad = false,
    } = config ?? {};

    // Validate parameters
    if (!(lr >= 0)) {
      throw new Error(`Invalid learning rate: ${lr}`);
    }
    if (!(eps >= 0)) {
      throw new Error(`Invalid epsilon value: ${eps}`);
    }
    if (!(betas[0] >= 0 && betas[0] < 1)) {
      throw new Error(`Invalid beta parameter at index 0: ${betas[0]}`);
    }
    if (!(betas[1] >= 0 && betas[1] < 1)) {
      throw new Error(`Invalid beta parameter at index 1: ${betas[1]}`);
    }
    if (!(weightDecay >= 0)) {
      throw new Error(`Invalid weight_decay value: ${weightDecay}`);
    }

    // Initialize with default options
    const defaults = {
      lr,
      betas,
      eps,
      weightDecay,
      amsgrad,
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
        lr = this.defaults.lr,
        betas = this.defaults.betas,
        eps = this.defaults.eps,
        weightDecay = this.defaults.weightDecay,
        amsgrad = this.defaults.amsgrad,
      } = group;

      const [beta1, beta2] = betas;

      for (const param of group.params) {
        if (!param.grad) {
          continue;
        }

        const grad = param.grad;

        // Get parameter state
        let state = this.state.get(param);
        if (!state) {
          state = {
            step: 0,
            expAvg: pin(zerosLike(param)),
            expAvgSq: pin(zerosLike(param)),
          };

          if (amsgrad) {
            state.maxExpAvgSq = pin(zerosLike(param));
          }

          this.state.set(param, state);
        }

        const expAvg = state.expAvg as Tensor;
        const expAvgSq = state.expAvgSq as Tensor;

        // Increment step counter
        state.step++;

        // Compute bias correction terms
        const biasCorrection1 = 1 - Math.pow(beta1, state.step);
        const biasCorrection2 = 1 - Math.pow(beta2, state.step);

        // Apply weight decay
        if (weightDecay !== 0) {
          grad.add_(param.mul(weightDecay));
        }

        // Decay the first and second moment running average coefficient
        expAvg.mul_(beta1).add_(grad.mul(1 - beta1));
        expAvgSq.mul_(beta2).addcmul_(grad, grad, 1 - beta2);

        // Use the max avg squared if using amsgrad
        let denom;
        if (amsgrad) {
          const maxExpAvgSq = state.maxExpAvgSq as Tensor;
          // Update max exponential moving average of squared gradient
          maxExpAvgSq.maximum_(expAvgSq);
          // Use the max for normalization
          denom = maxExpAvgSq.sqrt().div(Math.sqrt(biasCorrection2)).add(eps);
        } else {
          denom = expAvgSq.sqrt().div(Math.sqrt(biasCorrection2)).add(eps);
        }

        // Compute step size
        const stepSize = lr / biasCorrection1;

        // Update parameters
        param.addcdiv_(expAvg, denom, -stepSize);
      }
    }

    await this.device.markStep();

    return loss;
  }
}
