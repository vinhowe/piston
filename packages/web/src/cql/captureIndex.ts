import { PistonFunctionMode } from "@/function";
import { wasm } from "@/globals";
import { Module } from "@/nn/module";
import { RemovableHandle } from "@/utils";

import { ModelAdapter, ModelNode } from "./types";

export interface IndexParameter {
  shape: number[];
}

export interface IndexOperation {
  name: string;
  shape: number[];
  executionIndex: number;
}

export interface IndexModule {
  name: string | number;
  typeName: string;
  executionIndex: number;
  children: Map<string | number, IndexModule>;
  descendants: Map<string | number, IndexModule>;
  parameters: Map<string, IndexParameter>;
  buffers: Map<string, IndexParameter>;
  operations: IndexOperation[];
}

export interface IndexState {
  root: IndexModule;
  moduleExecutionOrder: IndexModule[];
  opExecutionOrder: IndexOperation[];
}

function processModuleName(name: string): string | number {
  const lastPart = name.split(".").pop()!;
  if (lastPart.match(/^\d+$/)) {
    return parseInt(lastPart);
  }
  return lastPart;
}

export class CaptureIndexMode extends PistonFunctionMode {
  private moduleStack: Module[] = [];
  private indexModuleByModule: Map<Module, IndexModule> = new Map();
  private stateStack: IndexModule[] = [];
  private forwardHookHandles: RemovableHandle[] = [];
  private hookedModules: WeakSet<Module> = new WeakSet();
  private moduleExecutionIndex: number = 0;
  private opExecutionIndex: number = 0;
  readonly index: IndexState;

  constructor(private readonly root: Module) {
    super();

    const registerModule = (module: Module) => {
      if (this.hookedModules.has(module)) return;

      const indexModule = this.indexModuleByModule.get(module);

      if (!indexModule) {
        throw new Error("Module not found in index");
      }

      // Load up parameters
      for (const [name, param] of module.namedParameters("", false)) {
        indexModule.parameters.set(name.split(".").pop()!, { shape: param.shape });
      }

      // Load up buffers
      for (const [name, buffer] of module.namedBuffers("", false)) {
        indexModule.buffers.set(name.split(".").pop()!, { shape: buffer.shape });
      }

      // Load up descendants
      for (const [name, descendant] of module.namedModules()) {
        if (descendant === module) continue;
        const indexDescendant = this.indexModuleByModule.get(descendant);
        if (!indexDescendant) {
          throw new Error("Descendant module not found in index");
        }
        indexModule.descendants.set(processModuleName(name), indexDescendant);
      }

      // Load up children
      for (const [name, child] of module.namedChildren()) {
        if (child === module) continue;
        const indexChild = this.indexModuleByModule.get(child);
        if (!indexChild) {
          throw new Error("Child module not found in index");
        }
        indexModule.children.set(processModuleName(name), indexChild);
      }

      const pre = module.registerForwardPreHook((m) => {
        this.onPreForward(m);
      });

      const post = module.registerForwardHook(() => {
        this.onPostForward();
      });

      this.forwardHookHandles.push(pre, post);
      this.hookedModules.add(module);
      // this._modulePathMap.set(mod, name);
    };

    for (const [name, module] of this.root.namedModules()) {
      this.indexModuleByModule.set(module, {
        name: processModuleName(name),
        typeName: module.constructor.name,
        executionIndex: this.moduleExecutionIndex,
        children: new Map(),
        descendants: new Map(),
        parameters: new Map(),
        buffers: new Map(),
        operations: [],
      });
    }

    for (const module of this.root.modules()) {
      registerModule(module);
    }

    registerModule(this.root);

    this.stateStack.push(this.indexModuleByModule.get(this.root)!);

    this.index = {
      root: this.indexModuleByModule.get(this.root)!,
      moduleExecutionOrder: [],
      opExecutionOrder: [],
    };
  }

  // Note here that we could simply iterate over descendants statically instead of during the
  // forward pass, but that will register as false positives any modules that aren't actually used.
  // Additionally, we need execution order data.
  private onPreForward(module: Module): void {
    // TODO: Determine whether the module stack is necessary
    this.moduleStack.push(module);

    const indexModule = this.indexModuleByModule.get(module);

    if (!indexModule) {
      throw new Error("Iterated module not found in index");
    }

    this.stateStack.push(indexModule);

    this.moduleExecutionIndex++;

    this.index.moduleExecutionOrder.push(indexModule);
  }

  private onPostForward(): void {
    this.stateStack.pop();
    this.moduleStack.pop();
  }

  _pistonFunction<T>(
    func: <FT>(...args: unknown[]) => FT | Promise<FT>,
    _types: unknown[],
    args: unknown[],
    kwargs: Record<string, unknown>,
  ): Promise<T> | T {
    const after = (result: T) => {
      if (result instanceof wasm.Tensor_wasm) {
        const currentModule = this.moduleStack[this.moduleStack.length - 1];
        const activeState = this.stateStack[this.stateStack.length - 1];
        const operation = {
          name: result.op().name,
          shape: result.shape,
          executionIndex: this.opExecutionIndex,
        };
        if (currentModule && activeState) {
          activeState.operations.push(operation);
        }
        this.index.opExecutionOrder.push(operation);
        this.opExecutionIndex++;
      }
      return result;
    };

    const ret = func(...args, kwargs);
    if (ret instanceof Promise) {
      return ret.then(after);
    }
    return after(ret as T);
  }
}

export class CaptureIndexModelNode implements ModelNode<IndexParameter> {
  constructor(
    private readonly node: IndexModule,
    public readonly parent: CaptureIndexModelNode | null = null,
  ) {}

  get typeName(): string {
    return this.node.typeName;
  }

  get name(): string {
    return this.node.name.toString();
  }

  getChildren(): CaptureIndexModelNode[] {
    const children: CaptureIndexModelNode[] = [];
    for (const childModule of this.node.children.values()) {
      if (childModule) {
        children.push(new CaptureIndexModelNode(childModule, this));
      }
    }
    return children;
  }

  getChild(nameOrIndex: string | number): CaptureIndexModelNode | undefined {
    let childModule: IndexModule | undefined;

    if (typeof nameOrIndex === "string") {
      childModule = this.node.children.get(nameOrIndex);
    } else if (typeof nameOrIndex === "number") {
      childModule = this.node.children.get(nameOrIndex);
    }

    return childModule ? new CaptureIndexModelNode(childModule, this) : undefined;
  }

  getParameters(): Record<string, IndexParameter> {
    return Object.fromEntries(this.node.parameters.entries());
  }

  getOperations(): IndexOperation[] {
    return this.node.operations;
  }

  matchesRegex(regex: RegExp): boolean {
    return regex.test(this.name);
  }

  path(): string[] {
    const parentPath = this.parent ? this.parent.path() : [];
    return [...parentPath, this.name];
  }
}

export class CaptureIndexModelAdapter implements ModelAdapter<IndexParameter> {
  constructor(private readonly state: IndexState) {}

  /**
   * Get the root modules.
   */
  getRootModules(): CaptureIndexModelNode[] {
    return [new CaptureIndexModelNode(this.state.root)];
  }

  /**
   * Get a parameter by name from a node.
   */
  getParameter(node: CaptureIndexModelNode, name: string): IndexParameter | undefined {
    const params = node.getParameters();
    return params[name];
  }

  /**
   * Get a child node by name.
   */
  getChild(node: CaptureIndexModelNode, name: string): CaptureIndexModelNode | undefined {
    return node.getChild(name);
  }

  /**
   * Find descendants of a node that match a predicate.
   */
  findDescendants(
    node: CaptureIndexModelNode,
    predicate: (node: CaptureIndexModelNode) => boolean,
    results: CaptureIndexModelNode[],
    maxDepth?: number,
  ): void {
    if (maxDepth !== undefined && maxDepth <= 0) {
      return;
    }

    // Check direct children
    const children = node.getChildren();
    for (const child of children) {
      if (predicate(child)) {
        results.push(child);
      }
      // Recursively check descendants
      this.findDescendants(
        child,
        predicate,
        results,
        maxDepth !== undefined ? maxDepth - 1 : undefined,
      );
    }
  }
}
