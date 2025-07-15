import { zerosLike } from "@/globals";
import { tensorName, withScope } from "@/nn/tracking";
import { Parameter } from "@/nn/parameter";
import { Optimizer } from "@/optim/optimizer";
import { OptimizerParamState, ParamGroup } from "@/optim/optimizer";
import { Tensor } from "@/tensor";
import { Device } from "@/wasm";

export interface AdamWParamState extends OptimizerParamState {
  step: number;
  expAvg: Tensor | null;
  expAvgSq: Tensor | null;
  maxExpAvgSq?: Tensor | null;
}

export interface AdamWParamGroup extends ParamGroup {
  lr: number;
  betas: [number, number];
  eps: number;
  weightDecay: number;
  amsgrad: boolean;
}

export interface AdamWConfig {
  lr: number;
  betas: [number, number];
  eps: number;
  weightDecay: number;
  amsgrad: boolean;
}

/**
 * Implementation of the AdamW optimizer (based on original PyTorch implementation)
 *
 * The original Adam algorithm was proposed in "Adam: A Method for Stochastic Optimization".
 * The AdamW variant was proposed in "Decoupled Weight Decay Regularization".
 */
export class AdamW extends Optimizer<AdamWParamState> {
  /**
   * Creates a new AdamW optimizer
   *
   * @param params - Parameters or parameter groups to optimize
   * @param device - The device to perform computations on
   * @param config - Configuration options for the optimizer:
   *  - `lr` (number): Learning rate (default: 1e-3)
   *  - `betas` ([number, number]): Coefficients for computing running averages of gradient and its
   *     square (default: [0.9, 0.999])
   *  - `eps` (number): Term added to denominator for numerical stability (default: 1e-8)
   *  - `weightDecay` (number): Weight decay coefficient (default: 1e-2)
   *  - `amsgrad` (boolean): Whether to use the AMSGrad variant (default: false)
   */
  constructor(params: Parameter[] | ParamGroup[], device: Device, config?: Partial<AdamWConfig>) {
    const {
      lr = 1e-3,
      betas = [0.9, 0.999],
      eps = 1e-8,
      weightDecay = 1e-2,
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
    let paramGroupIndex = 0;
    for (const group of this.paramGroups) {
      withScope(
        [
          {
            type: "optimizer-param-group",
            optimizer: this,
            paramGroup: group,
            name: paramGroupIndex.toString(),
          },
        ],
        () => {
          paramGroupIndex++;

          const {
            lr = this.defaults.lr as number,
            betas = this.defaults.betas as [number, number],
            eps = this.defaults.eps as number,
            weightDecay = this.defaults.weightDecay as number,
            amsgrad = this.defaults.amsgrad as number,
          } = group as AdamWParamGroup;

          const [beta1, beta2] = betas as [number, number];

          for (const param of group.params) {
            if (!param.grad()) {
              continue;
            }

            withScope(
              [
                {
                  type: "optimizer-param-update",
                  optimizer: this,
                  parameter: param,
                  name: tensorName(param),
                },
              ],
              () => {
                const grad = param.grad() as Tensor;

                // Apply weight decay
                if (weightDecay !== 0) {
                  param.mul_(1 - lr * weightDecay);
                }

                // Get parameter state
                let state = this.state.get(param);
                if (!state) {
                  state = {
                    step: 0,
                    expAvg: zerosLike(param),
                    expAvgSq: zerosLike(param),
                  };

                  if (amsgrad) {
                    state.maxExpAvgSq = zerosLike(param);
                  }

                  this.state.set(param, state);
                }

                const expAvg = state.expAvg as Tensor;
                const expAvgSq = state.expAvgSq as Tensor;
                const maxExpAvgSq = amsgrad ? (state.maxExpAvgSq as Tensor) : null;

                // Increment step counter
                state.step++;

                // Decay the first and second moment running average coefficient
                expAvg.mul_(beta1).add_(grad.mul(1 - beta1));
                expAvgSq.mul_(beta2).addcmul_(grad, grad, 1 - beta2);

                // Compute bias correction terms
                const biasCorrection1 = 1 - Math.pow(beta1, state.step);
                const biasCorrection2 = 1 - Math.pow(beta2, state.step);

                // Use the max avg squared if using amsgrad
                let denom;
                if (amsgrad) {
                  // Update max exponential moving average of squared gradient
                  maxExpAvgSq!.maximum_(expAvgSq);
                  // Use the max for normalization
                  denom = maxExpAvgSq!.sqrt().div(Math.sqrt(biasCorrection2)).add(eps);
                } else {
                  denom = expAvgSq.sqrt().div(Math.sqrt(biasCorrection2)).add(eps);
                }

                // Compute step size
                const stepSize = lr / biasCorrection1;

                // Update parameters
                param.addcdiv_(expAvg, denom, -stepSize);
              },
            );
          }
        },
      );
    }

    await this.device.markStep();

    return loss;
  }
}
