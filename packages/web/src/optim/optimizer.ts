import { ScopeItem, withScope } from "@/nn/tracking";
import { Parameter } from "@/parameter";
import { Tensor, TensorHook } from "@/tensor";
import { RemovableHandle } from "@/utils";
import { Device } from "@/wasm";

/**
 * Base configuration for parameter groups
 */
export interface ParamGroupConfig {
  params: number[];
  [key: string]: unknown;
}

/**
 * Runtime parameter group with actual Parameter objects
 */
export interface ParamGroup {
  params: Parameter[];
  lr?: number;
  weightDecay?: number;
  [key: string]: unknown;
}

/**
 * Type alias for optimizer pre-hook function
 * Returns modified args and kwargs if needed, or undefined if no changes
 */
export type OptimizerPreHook = (
  optimizer: Optimizer,
  args: unknown[],
  kwargs: Record<string, unknown>,
) => [unknown[], Record<string, unknown>] | undefined;

/**
 * Type alias for optimizer post-hook function
 */
export type OptimizerPostHook = (
  optimizer: Optimizer,
  args: unknown[],
  kwargs: Record<string, unknown>,
) => void;

/**
 * Base type for optimizer parameter state
 */
export interface OptimizerParamState {
  step?: number;
}

/**
 * Type for optimizer state dictionary serialization
 */
export interface StateDict {
  state: Record<number, OptimizerParamState>;
  paramGroups: ParamGroupConfig[];
}

/**
 * Base class for all optimizers
 *
 * This class defines the interface and common functionality for all optimizers.
 * Subclasses should implement the `step` method which performs the actual
 * parameter updates.
 */
export class Optimizer<TState extends OptimizerParamState = OptimizerParamState> {
  defaults: Record<string, unknown>;
  device: Device;
  state: Map<Parameter, TState>;
  paramGroups: ParamGroup[];
  scope: ScopeItem[] | undefined;
  private _optimizerStepPreHooks: Map<number, OptimizerPreHook>;
  private _optimizerStepPostHooks: Map<number, OptimizerPostHook>;
  /** @internal Tensor class can access the map on its parent module to execute hooks */
  _tensorHooks: Map<number, TensorHook>;

  /**
   * Creates a new optimizer
   *
   * @param params - Iterable of parameters or parameter groups
   * @param defaults - Default optimization options
   */
  constructor(
    params: Parameter[] | ParamGroup[],
    device: Device,
    defaults: Record<string, unknown>,
  ) {
    this.defaults = defaults;
    this.device = device;
    this.state = new Map();
    this.paramGroups = [];
    this._optimizerStepPreHooks = new Map();
    this._optimizerStepPostHooks = new Map();
    this._tensorHooks = new Map();

    if (params instanceof Parameter) {
      throw new TypeError(
        "params argument given to the optimizer should be an iterable of " +
          "Parameters or parameter groups",
      );
    }

    const paramGroups = Array.isArray(params) ? [...params] : params;
    if (paramGroups.length === 0) {
      throw new Error("Optimizer got an empty parameter list");
    }

    if (!("params" in paramGroups[0])) {
      // If the first element doesn't have a params field, assume all elements are Parameters
      // and convert to a single parameter group
      this.addParamGroup({ params: paramGroups as Parameter[] });
    } else {
      // Otherwise, we have parameter groups
      for (const group of paramGroups as ParamGroup[]) {
        this.addParamGroup(group);
      }
    }

    // Return a proxy to intercept method calls like 'step'
    return new Proxy(this, {
      get: (
        target: Optimizer<TState>,
        prop: string | symbol,
        receiver: unknown,
      ): unknown => {
        // Intercept the 'step' method
        if (prop === "step" && typeof target.step === "function") {
          const originalStep = Reflect.get(target, prop, receiver) as (
            ...args: unknown[]
          ) => Promise<number | undefined>;

          // Return a wrapped async function
          return async function (
            ...args: unknown[]
          ): Promise<number | undefined> {
            return withScope(
              [
                {
                  type: "optimizer",
                  optimizer: target,
                  name: undefined,
                },
                ...(target.scope ?? []),
              ],
              () => {
                return Reflect.apply(originalStep, receiver, args);
              },
              { replace: true },
            );
          };
        }

        // Default behavior for other properties and methods
        return Reflect.get(target, prop, receiver);
      },
    }) as Optimizer<TState>;
  }

  /**
   * Register an optimizer step pre-hook
   * The hook will be called before each optimizer step
   *
   * @param hook - Function to call before step
   * @returns A handle that can be used to remove the hook
   */
  registerStepPreHook(hook: OptimizerPreHook): RemovableHandle {
    const handle = new RemovableHandle(this._optimizerStepPreHooks);
    this._optimizerStepPreHooks.set(handle.id, hook);
    return handle;
  }

  /**
   * Register an optimizer step post-hook
   * The hook will be called after each optimizer step
   *
   * @param hook - Function to call after step
   * @returns A handle that can be used to remove the hook
   */
  registerStepPostHook(hook: OptimizerPostHook): RemovableHandle {
    const handle = new RemovableHandle(this._optimizerStepPostHooks);
    this._optimizerStepPostHooks.set(handle.id, hook);
    return handle;
  }

  /**
   * Register a tensor hook
   * The hook will be called when a tensor is created
   *
   * @param hook - Function to call when a tensor is created
   * @returns A handle that can be used to remove the hook
   */
  registerTensorHook(hook: TensorHook): RemovableHandle {
    const handle = new RemovableHandle(this._tensorHooks);
    this._tensorHooks.set(handle.id, hook);
    return handle;
  }

  /**
   * Adds a parameter group to the optimizer
   *
   * @param paramGroup - Parameter group to add
   */
  addParamGroup(paramGroup: ParamGroup): void {
    if (typeof paramGroup !== "object") {
      throw new TypeError("Parameter group must be an object");
    }

    let params = paramGroup.params;
    if (!Array.isArray(params)) {
      params = [params as Parameter];
    }

    // Create a new parameter group with defaults
    const group: ParamGroup = { ...this.defaults, ...paramGroup, params: [] };

    // Check parameters
    for (const param of params) {
      if (!(param instanceof Parameter)) {
        throw new TypeError(
          "Optimizer can only optimize Parameters, " +
            // @ts-expect-error - we're checking that param is a Parameter
            `but one of the params is ${param?.constructor?.name}`,
        );
      }

      // Check if parameter is already in another group
      for (const existingGroup of this.paramGroups) {
        if (existingGroup.params.includes(param)) {
          throw new Error(
            "Some parameters appear in more than one parameter group",
          );
        }
      }

      (group.params as Parameter[]).push(param);
    }

    this.paramGroups.push(group);
  }

  /**
   * Returns the state of the optimizer as a dict
   *
   * @returns State dictionary
   */
  stateDict(): StateDict {
    // Map parameters to indices
    const paramMappings = new Map<Parameter, number>();
    let startIndex = 0;

    // Pack groups with parameter indices
    const packedGroups = this.paramGroups.map((group) => {
      const packed: Record<string, unknown> = { ...group };
      delete packed.params;

      // Map parameters to indices
      for (const param of group.params) {
        if (!paramMappings.has(param)) {
          paramMappings.set(param, startIndex++);
        }
      }

      // Store param indices
      packed.params = group.params.map((p) => paramMappings.get(p));
      return packed;
    });

    // Remap state to use indices as keys
    const packedState: Record<number, unknown> = {};
    this.state.forEach((value, key) => {
      const paramId = paramMappings.get(key) as number;
      packedState[paramId] = { ...value };
    });

    return {
      state: packedState as Record<number, TState>,
      paramGroups: packedGroups as ParamGroupConfig[],
    };
  }

  /**
   * Loads optimizer state
   *
   * @param stateDict - State dictionary to load
   */
  loadStateDict(stateDict: StateDict): void {
    // Validate the state dict
    const groups = this.paramGroups;
    const savedGroups = stateDict.paramGroups;

    if (groups.length !== savedGroups.length) {
      throw new Error(
        "Loaded state dict has a different number of parameter groups",
      );
    }

    const paramLens = groups.map((g) => g.params.length);
    const savedLens = savedGroups.map((g) => g.params.length);

    if (paramLens.some((len, i) => len !== savedLens[i])) {
      throw new Error(
        "Loaded state dict contains a parameter group " +
          "that doesn't match the size of optimizer's group",
      );
    }

    // Create a map from saved parameter indices to current parameters
    const idMap = new Map<number, Parameter>();
    savedGroups.forEach((savedGroup, i) => {
      const group = groups[i];
      (savedGroup.params as number[]).forEach((paramId, j) => {
        idMap.set(paramId, group.params[j]);
      });
    });

    // Update the state
    const newState = new Map<Parameter, TState>();
    for (const [paramId, savedState] of Object.entries(stateDict.state)) {
      const param = idMap.get(Number(paramId));
      if (param) {
        // Copy state to parameter
        newState.set(param, this._processValue(param, savedState) as TState);
      }
    }
    this.state = newState;

    // Update parameter groups
    groups.forEach((group, i) => {
      const savedGroup = savedGroups[i];
      for (const key of Object.keys(savedGroup)) {
        if (key !== "params") {
          group[key] = savedGroup[key];
        }
      }
    });
  }

  /**
   * Helper function to process values from state dict
   */
  protected _processValue(param: Parameter, value: unknown): unknown {
    if (value instanceof Tensor) {
      // Move tensor to same device and cast to param dtype if floating point
      if (param.dtype().isFloatingPoint()) {
        return value.cast(param.dtype());
      }
      return value;
    } else if (Array.isArray(value)) {
      return value.map((v) => this._processValue(param, v));
    } else if (typeof value === "object" && value !== null) {
      const result: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(value)) {
        result[k] = this._processValue(param, v);
      }
      return result;
    }
    return value;
  }

  /**
   * Zero out the gradients of all parameters
   *
   * @param setToNone - If true, set gradients to null instead of zero
   */
  zeroGrad(setToNone: boolean = true): void {
    for (const group of this.paramGroups) {
      for (const param of group.params) {
        const grad = param.grad();
        if (grad) {
          if (setToNone) {
            // TODO: Need to implement proper "set grad to null" logic
            grad.mul(0); // For now, just zero it out
          } else {
            grad.mul(0);
          }
        }
      }
    }
  }

  /**
   * Performs a single optimization step
   *
   * @param closure - Closure that reevaluates the model and returns the loss
   * @returns The loss value
   */
  async step(_closure?: () => number): Promise<number | undefined> {
    throw new Error("Subclasses must implement the step method");
  }

  /**
   * Performs a single optimization step with hook handling
   *
   * @param originalStep - The actual step implementation
   * @param args - Arguments to pass to step
   * @param kwargs - Keyword arguments to pass to step
   * @returns The result of the step function
   */
  protected _stateWithHooks(
    originalStep: (...args: unknown[]) => unknown,
    args: unknown[] = [],
    kwargs: Record<string, unknown> = {},
  ): unknown {
    // Call pre-hooks
    for (const preHook of this._optimizerStepPreHooks.values()) {
      const result = preHook(this, args, kwargs);
      if (result) {
        [args, kwargs] = result;
      }
    }

    // Call the actual step function
    const result = originalStep.apply(this, args);

    // Call post-hooks
    for (const postHook of this._optimizerStepPostHooks.values()) {
      postHook(this, args, kwargs);
    }

    return result;
  }
}
