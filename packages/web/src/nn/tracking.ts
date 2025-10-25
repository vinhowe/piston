import { PistonFunctionMode } from "@/function";
import {
  Module,
  registerModuleBufferRegistrationHook,
  registerModuleModuleRegistrationHook,
  registerModuleParameterRegistrationHook,
} from "@/nn/module";
import { Buffer, Parameter } from "@/nn/parameter";
import { Optimizer } from "@/optim";
import { OpDescription, Tensor } from "@/tensor";
import { RemovableHandle } from "@/utils";
import { Tensor_wasm } from "@/wasm";

export interface BaseScopeItem {
  type: "module" | "optimizer";
  name: string | undefined;
}

export interface ModuleScopeItem extends BaseScopeItem {
  type: "module";
  module: Module<unknown, unknown>;
}

export interface OptimizerScopeItem extends BaseScopeItem {
  type: "optimizer";
  optimizer: Optimizer;
}

export type ScopeItem = ModuleScopeItem | OptimizerScopeItem;

export interface TrackedTensor {
  id: number;
  op: OpDescription;
  shape: number[];
  srcIds: number[];
  nameOnParent: string | undefined;
  scope: ScopeItem[] | undefined;
  tensor: Tensor | undefined;
  debugTensor: Tensor | undefined;
  dependency: boolean;
  index: number;
}

export interface TensorTrackOptions {
  dependency?: boolean;
}

/**
 * Tracking dispatch mode that intercepts tensor operations and builds comprehensive tracking data.
 * This replaces the old tensor hook system with a dispatch-based approach.
 */
export class TrackingFunctionMode extends PistonFunctionMode {
  private trackStack: TrackedTensor[] = [];
  private alreadyTracked = new Set<number>();
  private index = 0;

  // Current scope tracking
  private currentScope: ScopeItem[] = [];
  private currentTensorName: string | undefined;

  // Module path tracking for scope building
  private modulePathMap = new WeakMap<Module, string>();
  private parameterNameMap = new WeakMap<Parameter, string>();
  private bufferNameMap = new WeakMap<Buffer, string>();

  // Module scope stack for proper nesting
  private moduleStack: Module<unknown, unknown>[] = [];

  // Hook handles for cleanup
  private hookHandles: RemovableHandle[] = [];

  // Track modules that already have forward hooks registered
  private hookedModules: WeakSet<Module<unknown, unknown>> = new WeakSet();

  constructor() {
    super();
    this.setupHooks();
  }

  private setupHooks(): void {
    // Register global hooks to track module structure
    this.hookHandles.push(
      registerModuleModuleRegistrationHook((parentModule, name, submodule) => {
        if (submodule) {
          const parentPath = this.modulePathMap.get(parentModule) || "";
          const fullPath = parentPath ? `${parentPath}.${name}` : name;
          this.modulePathMap.set(submodule, fullPath);

          // Register forward hooks on the submodule for scope tracking
          this.setupModuleForwardHooks(submodule, fullPath, true);
        }
      }),
    );

    this.hookHandles.push(
      registerModuleParameterRegistrationHook((_module, name, param) => {
        if (param) {
          this.parameterNameMap.set(param, name);
        }
      }),
    );

    this.hookHandles.push(
      registerModuleBufferRegistrationHook((_module, name, buffer) => {
        if (buffer) {
          this.bufferNameMap.set(buffer, name);
        }
      }),
    );
  }

  setupModuleForwardHooks(
    module: Module<unknown, unknown>,
    path?: string,
    recurse: boolean = false,
  ): void {
    if (path) {
      this.modulePathMap.set(module, path);
    }

    if (this.hookedModules.has(module)) {
      return;
    }

    // Pre-hook: push module to scope before forward
    this.hookHandles.push(
      module.registerForwardPreHook((mod, _inputs) => {
        this.moduleStack.push(mod);
        this.updateCurrentScope();
        return undefined; // Don't modify inputs
      }),
    );

    // Post-hook: pop module from scope after forward
    this.hookHandles.push(
      module.registerForwardHook((_mod, _inputs, _output) => {
        this.moduleStack.pop();
        this.updateCurrentScope();
        return undefined; // Don't modify output
      }),
    );

    this.hookedModules.add(module);

    if (recurse) {
      const basePath = path || this.modulePathMap.get(module) || module.constructor.name || "";
      for (const [name, child] of module.namedChildren()) {
        if (child) {
          const childPath = basePath ? `${basePath}.${name}` : name;
          this.setupModuleForwardHooks(child as Module<unknown, unknown>, childPath, true);
        }
      }
    }
  }

  private updateCurrentScope(): void {
    this.currentScope = this.moduleStack.map((module) => ({
      type: "module" as const,
      name: this.modulePathMap.get(module) || module.constructor.name,
      module,
    }));
  }

  initializeRootModule(module: Module<unknown, unknown>): void {
    // Set the root module path
    this.modulePathMap.set(module, module.constructor.name);
  }

  _pistonFunction<T>(
    func: <FT>(...args: unknown[]) => FT | Promise<FT>,
    _types: unknown[],
    args: unknown[],
    kwargs: Record<string, unknown>,
  ): Promise<T> | T {
    const after = (result: T) => {
      // Track the resulting tensor
      if (result instanceof Tensor_wasm) {
        this.trackTensor(result, { dependency: false });
      }
      return result;
    };

    // Call the next dispatch mode or default implementation
    const result = func(...args, kwargs);

    if (result instanceof Promise) {
      return result.then(after);
    }

    return after(result as T);
  }

  private trackTensor(tensor: Tensor, options: TensorTrackOptions): Tensor {
    const { dependency = false } = options;
    const id = tensor.id;

    if (this.alreadyTracked.has(id)) {
      return tensor;
    }

    // Try to determine tensor name from parameter/buffer maps
    let tensorName = this.currentTensorName;
    if (!tensorName) {
      tensorName = this.findTensorName(tensor);
    }

    const trackedTensor: TrackedTensor = {
      id,
      op: tensor.op(),
      shape: tensor.shape,
      srcIds: tensor.srcIds(),
      nameOnParent: tensorName,
      scope: [...this.currentScope], // Copy current scope
      tensor: tensor,
      debugTensor: tensor.debugTensor,
      dependency,
      index: this.index++,
    };

    this.trackStack.push(trackedTensor);
    this.alreadyTracked.add(id);

    // Reset tensor name after tracking
    this.currentTensorName = undefined;

    return tensor;
  }

  private findTensorName(tensor: Tensor): string | undefined {
    // Check if this tensor is a parameter or buffer in the current module
    if (this.moduleStack.length > 0) {
      const currentModule = this.moduleStack[this.moduleStack.length - 1];

      // Check parameters using the public API
      for (const [name, param] of currentModule.namedParametersIter("", false)) {
        if (param && param === tensor) {
          return name;
        }
      }

      // Check buffers using the public API
      for (const [name, buffer] of currentModule.namedBuffersIter("", false)) {
        if (buffer && buffer === tensor) {
          return name;
        }
      }
    }

    return undefined;
  }

  /**
   * Set the current scope for subsequent tensor operations
   */
  withScope<T>(scope: ScopeItem[], f: () => T): T {
    const originalScope = [...this.currentScope];
    this.currentScope = scope;
    try {
      return f();
    } finally {
      this.currentScope = originalScope;
    }
  }

  /**
   * Set the current tensor name for the next tensor operation
   */
  withTensorName<T>(name: string, f: () => T): T {
    const originalName = this.currentTensorName;
    this.currentTensorName = name;
    try {
      return f();
    } finally {
      this.currentTensorName = originalName;
    }
  }

  /**
   * Get all tracked tensors
   */
  getTrackedTensors(): TrackedTensor[] {
    return [...this.trackStack];
  }

  /**
   * Clear tracking state
   */
  clear(): void {
    this.trackStack = [];
    this.alreadyTracked.clear();
    this.index = 0;
    this.currentScope = [];
    this.currentTensorName = undefined;
    this.moduleStack = [];
  }

  /**
   * Clean up hooks when disposing
   */
  [Symbol.dispose](): void {
    super[Symbol.dispose]();
    this.hookHandles.forEach((handle) => handle.remove());
    this.hookHandles = [];
  }
}

export function nameFromScope(scope: ScopeItem[]): string {
  return (
    scope.map((item) => (item.name?.includes(".") ? `[${item.name}]` : item.name)).join(".") || "."
  );
}

export function tensorName(tensor: TrackedTensor): string {
  return nameFromScope(tensor.scope ?? []) + (tensor.nameOnParent ? `.${tensor.nameOnParent}` : "");
}

/**
 * Track tensor operations during module forward pass
 */
export function track<Output>(
  module: Module<unknown, Output>,
  ...args: Parameters<typeof module.forward>
): [Output, TrackedTensor[]] {
  using trackingMode = new TrackingFunctionMode();

  // Initialize the root module in the path map
  const rootModule = module as Module<unknown, unknown>;
  trackingMode.initializeRootModule(rootModule);

  // Set up forward hooks for the root module and recurse into children
  trackingMode.setupModuleForwardHooks(rootModule, rootModule.constructor.name, true);

  const result = module.forward(...args);

  return [result, trackingMode.getTrackedTensors()];
}
