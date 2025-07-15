import { zerosLike } from "@/globals";
import { tensorName, withScope } from "@/nn/tracking";
import { Optimizer } from "@/optim/optimizer";
import { OptimizerParamState, ParamGroup } from "@/optim/optimizer";
import { Parameter } from "@/parameter";
import { Tensor } from "@/tensor";
import { Device } from "@/wasm";

export interface SGDParamState extends OptimizerParamState {
  momentumBuffer?: Tensor | null;
}

export interface SGDParamGroup extends ParamGroup {
  lr: number;
  momentum: number;
  dampening: number;
  weightDecay: number;
  nesterov: boolean;
}

export interface SGDConfig {
  lr: number;
  momentum: number;
  dampening: number;
  weightDecay: number;
  nesterov: boolean;
}

/**
 * Implementation of the SGD optimizer (based on original PyTorch implementation)
 *
 * Nesterov momentum is based on the paper "On the importance of initialization and momentum in deep
 * learning".
 */
export class SGD extends Optimizer<SGDParamState> {
  /**
   * Creates a new SGD optimizer
   *
   * @param params - Parameters or parameter groups to optimize
   * @param device - The device to perform computations on
   * @param config - Configuration options for the optimizer:
   *  - `lr` (number): Learning rate (REQUIRED)
   *  - `momentum` (number): Momentum factor (default: 0)
   *  - `dampening` (number): Dampening for momentum (default: 0)
   *  - `weightDecay` (number): Weight decay (L2 penalty) (default: 0)
   *  - `nesterov` (boolean): Enables Nesterov momentum (default: false)
   */
  constructor(
    params: Parameter[] | ParamGroup[],
    device: Device,
    config: Partial<SGDConfig> & Pick<SGDConfig, "lr">,
  ) {
    const {
      lr,
      momentum = 0,
      dampening = 0,
      weightDecay = 0,
      nesterov = false,
    } = config;

    // Validate parameters
    if (!(lr >= 0)) {
      throw new Error(`Invalid learning rate: ${lr}`);
    }
    if (!(momentum >= 0)) {
      throw new Error(`Invalid momentum value: ${momentum}`);
    }
    if (!(weightDecay >= 0)) {
      throw new Error(`Invalid weight_decay value: ${weightDecay}`);
    }
    if (nesterov && (momentum <= 0 || dampening !== 0)) {
      throw new Error(
        "Nesterov momentum requires a momentum and zero dampening",
      );
    }

    // Initialize with default options
    const defaults = {
      lr,
      momentum,
      dampening,
      weightDecay,
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
    let paramGroupIndex = 0;
    for (const group of this.paramGroups) {
      const {
        lr = this.defaults.lr as number,
        momentum = this.defaults.momentum as number,
        dampening = this.defaults.dampening as number,
        weightDecay = this.defaults.weightDecay as number,
        nesterov = this.defaults.nesterov as boolean,
      } = group as SGDParamGroup;

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
                let grad = param.grad() as Tensor;

                // Apply weight decay
                if (weightDecay !== 0) {
                  grad = grad.add(param.mul(weightDecay));
                }

                // Apply momentum
                if (momentum !== 0) {
                  let state = this.state.get(param);
                  if (!state) {
                    state = {};
                    this.state.set(param, state);
                  }

                  let buf = state.momentumBuffer;
                  if (!buf) {
                    buf = zerosLike(grad);
                    state.momentumBuffer = buf;
                  } else {
                    buf.mul_(momentum).add_(grad.mul(1 - dampening));
                  }

                  if (nesterov) {
                    grad = grad.add(buf!.mul(momentum));
                  } else {
                    grad = buf!;
                  }
                }

                // Update parameters
                param.add_(grad.mul(-lr));
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
