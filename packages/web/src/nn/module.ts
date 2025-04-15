import { Parameter } from "@/parameter";
import { Tensor } from "@/tensor";
import { RemovableHandle } from "@/utils";
import { Tensor_wasm } from "@/wasm";

import { ModuleScopeItem, ScopeItem, withScope } from "./tracking";

// TODO(vinhowe): These implementations are not very good right now

/**
 * Buffer class to represent non-trainable tensors
 * Buffers are tensors that do not require gradients for optimization during training.
 * This is similar to torch.Tensor in PyTorch.
 */
export class Buffer {
  data: Tensor;
  persistent: boolean;

  constructor(data: Tensor, persistent: boolean = true) {
    this.data = data;
    this.persistent = persistent;
  }
}

/**
 * Module class to represent neural network modules
 * Modules are the building blocks of neural networks.
 * They can contain parameters, buffers, and other modules.
 * This is similar to torch.nn.Module in PyTorch.
 */
export class Module<Input = unknown, Output = unknown> {
  training: boolean;
  protected _parameters: Record<string, Parameter>;
  protected _buffers: Record<string, Buffer>;
  protected _modules: Record<string, Module<unknown, unknown>>;
  protected _nonPersistentBuffersSet: Set<string>;
  protected _forwardPreHooks: Map<
    number,
    (module: Module<Input, Output>, inputs: Input) => Input | undefined
  >;
  protected _forwardHooks: Map<
    number,
    (
      module: Module<Input, Output>,
      inputs: Input,
      output: Output,
    ) => Output | undefined
  >;
  protected _nextHookId: number;
  // Name of the property on the parent module that this module is assigned to
  // Helps us keep track of module scopes
  nameOnParent: string | undefined;
  scope: ScopeItem[] | undefined;

  constructor() {
    this.training = true;
    this._parameters = {};
    this._buffers = {};
    this._modules = {};
    this._nonPersistentBuffersSet = new Set<string>();
    this._nextHookId = 0;
    this._forwardPreHooks = new Map();
    this._forwardHooks = new Map();
    this.scope = [
      {
        type: "module",
        module: this as Module<unknown, unknown>,
        name: undefined,
      },
    ];

    // Use Proxy to intercept property assignments and access
    return new Proxy(this, {
      set: (
        target: Module<Input, Output>,
        prop: string | symbol,
        value: unknown,
      ): boolean => {
        // Allow direct setting of internal properties
        if (typeof prop === "symbol" || prop.toString().startsWith("_")) {
          (target as unknown as Record<string, unknown>)[prop as string] =
            value;
          return true;
        }

        // Handle attribute setting through _setAttr
        return target._setAttr(prop.toString(), value);
      },

      get: (
        target: Module<Input, Output>,
        prop: string | symbol,
        receiver: unknown,
      ): unknown => {
        // Handle hooks
        if (prop === "forward" && typeof target.forward === "function") {
          return function (...args: unknown[]) {
            // Apply pre-hooks
            let hookInput = args;
            for (const hook of target._forwardPreHooks.values()) {
              const result = hook(target, args as Input);
              if (result !== undefined) {
                hookInput = Array.isArray(result) ? result : [result];
              }
            }

            const output = withScope(
              target.scope ?? [],
              () => Reflect.apply(target.forward, receiver, hookInput),
              { replace: true },
            );

            // Apply post-hooks
            let hookOutput = output;
            for (const hook of target._forwardHooks.values()) {
              const result = hook(target, hookInput as Input, hookOutput);
              if (result !== undefined) {
                hookOutput = result;
              }
            }

            return hookOutput;
          };
        }

        // Return methods and internal properties directly
        if (
          prop in target ||
          typeof prop === "symbol" ||
          prop.toString().startsWith("_")
        ) {
          return (target as unknown as Record<string, unknown>)[prop as string];
        }

        // Return registered parameters, buffers, or modules
        const propStr = prop.toString();
        if (propStr in target._parameters) return target._parameters[propStr];
        if (propStr in target._buffers) return target._buffers[propStr];
        if (propStr in target._modules) return target._modules[propStr];

        return undefined;
      },
    }) as Module<Input, Output>;
  }

  _setAttr(name: string, value: unknown): boolean {
    // Helper function to remove attribute from various collections
    const removeFrom = (
      ...containers: Array<Record<string, unknown> | Set<string> | object>
    ): void => {
      for (const container of containers) {
        if (container instanceof Set) {
          container.delete(name);
        } else if (container instanceof Object && name in container) {
          delete (container as Record<string, unknown>)[name];
        }
      }
    };

    // Handle Parameter assignment
    if (Parameter._unwrap(value as Parameter) instanceof Tensor_wasm) {
      if (!this._parameters) {
        throw new Error(
          "Cannot assign parameters before Module constructor call",
        );
      }
      removeFrom(
        this,
        this._buffers,
        this._modules,
        this._nonPersistentBuffersSet,
      );
      this.registerParameter(name, value as Parameter);
    }
    // Handle existing Parameter being replaced
    else if (this._parameters && name in this._parameters) {
      if (value !== null) {
        throw new TypeError(
          `Cannot assign '${value?.constructor?.name || typeof value}' as parameter '${name}' ` +
            "(Parameter or null expected)",
        );
      }
      this.registerParameter(name, value);
    }
    // Handle Module assignment
    else if (value instanceof Module) {
      if (!this._modules) {
        throw new Error("Cannot assign module before Module constructor call");
      }
      removeFrom(
        this,
        this._parameters,
        this._buffers,
        this._nonPersistentBuffersSet,
      );
      this.addModule(name, value);
    }
    // Handle existing Module being replaced
    else if (this._modules && name in this._modules) {
      if (value !== null) {
        throw new TypeError(
          `Cannot assign '${value?.constructor?.name || typeof value}' as child module '${name}' ` +
            "(Module or null expected)",
        );
      }
      this.addModule(name, value);
    }
    // Handle Buffer or existing Buffer being replaced
    else if (
      value instanceof Buffer ||
      (this._buffers && name in this._buffers)
    ) {
      if (value !== null && !(value instanceof Buffer)) {
        throw new TypeError(
          `Cannot assign '${value?.constructor?.name || typeof value}' as buffer '${name}' ` +
            "(Buffer or null expected)",
        );
      }
      const persistent =
        value instanceof Buffer
          ? value.persistent
          : !this._nonPersistentBuffersSet.has(name);
      this.registerBuffer(name, value, persistent);
    }
    // Handle regular attribute
    else {
      (this as unknown as Record<string, unknown>)[name] = value;
    }

    return true;
  }

  registerParameter(
    name: string,
    param: Parameter | null,
  ): Module<Input, Output> {
    if (param !== null && !(Parameter._unwrap(param) instanceof Tensor_wasm)) {
      throw new TypeError(
        `Cannot assign '${(param as unknown)?.constructor?.name || typeof param}' as parameter '${name}'`,
      );
    }

    // Remove any existing attribute
    delete (this as unknown as Record<string, unknown>)[name];

    if (param === null) {
      delete this._parameters[name];
    } else {
      this._parameters[name] = param;
    }

    (param as Parameter).scope = [...(this.scope || [])];
    (param as Parameter).nameOnParent = name;
    return this;
  }

  registerBuffer(
    name: string,
    buffer: Buffer | null,
    persistent: boolean = true,
  ): Module<Input, Output> {
    if (buffer !== null && !(buffer instanceof Buffer)) {
      throw new TypeError(
        `Cannot assign '${(buffer as unknown)?.constructor?.name || typeof buffer}' as buffer '${name}'`,
      );
    }

    // Remove any existing attribute
    delete (this as unknown as Record<string, unknown>)[name];

    if (buffer === null) {
      delete this._buffers[name];
      if (this._nonPersistentBuffersSet.has(name)) {
        this._nonPersistentBuffersSet.delete(name);
      }
    } else {
      this._buffers[name] = buffer;
      if (!persistent) {
        this._nonPersistentBuffersSet.add(name);
      } else if (this._nonPersistentBuffersSet.has(name)) {
        this._nonPersistentBuffersSet.delete(name);
      }
    }

    (buffer as Buffer).data.scope = [...(this.scope || [])];
    (buffer as Buffer).data.nameOnParent = name;
    return this;
  }

  /**
   * Add a child module to the current module.
   *
   * The module can be accessed as an attribute using the given name.
   *
   * @param name - name of the child module. The child module can be accessed from this module using the given name
   * @param module - child module to be added to the module.
   */
  addModule<SubmoduleInput, SubmoduleOutput>(
    name: string,
    module: Module<SubmoduleInput, SubmoduleOutput> | null,
  ): Module<Input, Output> {
    if (module !== null && !(module instanceof Module)) {
      throw new Error(
        `${(module as unknown)?.constructor?.name || typeof module} is not a Module`,
      );
    } else if (typeof name !== "string") {
      throw new Error(`module name should be a string. Got ${typeof name}`);
    } else if (name in this && !(name in this._modules)) {
      throw new Error(`attribute '${name}' already exists`);
    } else if (name.includes(".")) {
      throw new Error(`module name can't contain ".", got: ${name}`);
    } else if (name === "") {
      throw new Error('module name can\'t be empty string ""');
    }

    this._modules[name] = module as Module<unknown, unknown>;
    (module!.scope![module!.scope!.length - 1] as ModuleScopeItem).name = name;
    const thisScope = this.scope || [];
    module?.parameters().forEach((param) => {
      (param as Parameter).scope = [...thisScope, ...(param.scope ?? [])];
    });
    module?.buffers().forEach((buf) => {
      (buf as Buffer).data.scope = [...thisScope, ...(buf.data.scope ?? [])];
    });
    module?.modules().forEach((mod) => {
      (mod as Module<unknown, unknown>).scope = [
        ...thisScope,
        ...(mod.scope ?? []),
      ];
    });
    return this;
  }

  /**
   * Helper method that returns an iterator over named members of the module.
   */
  private *_namedMembers<SubmoduleInput, SubmoduleOutput, MemberType>(
    getMembersFn: (
      module: Module<SubmoduleInput, SubmoduleOutput>,
    ) => Iterable<[string, MemberType]>,
    prefix: string = "",
    recurse: boolean = true,
    removeDuplicate: boolean = true,
    memo: Set<MemberType> = new Set(),
  ): Generator<[string, MemberType]> {
    // Get members from this module
    for (const [name, item] of getMembersFn(
      this as unknown as Module<SubmoduleInput, SubmoduleOutput>,
    )) {
      if (item !== null) {
        if (removeDuplicate && memo.has(item)) {
          continue;
        }
        memo.add(item);
        yield [prefix + (prefix ? "." : "") + name, item];
      }
    }

    // Recursively get members from child modules
    if (recurse) {
      for (const [name, module] of Object.entries(this._modules)) {
        if (module === null) {
          continue;
        }

        const submodulePrefix = prefix + (prefix ? "." : "") + name;

        // Create a generator from the child module's named members
        const submodule_gen = module._namedMembers(
          getMembersFn,
          submodulePrefix,
          recurse,
          removeDuplicate,
          memo,
        );

        // Yield all elements from the submodule generator
        yield* submodule_gen;
      }
    }
  }

  /**
   * Return an iterator over module parameters.
   *
   * @param recurse - If true, recurses into child modules
   */
  *parametersIter(recurse: boolean = true): Generator<Parameter> {
    for (const [, param] of this.namedParametersIter("", recurse)) {
      yield param;
    }
  }

  /**
   * Return an array of module parameters.
   *
   * @param recurse - If true, recurses into child modules
   */
  parameters(recurse: boolean = true): Parameter[] {
    return Array.from(this.parametersIter(recurse));
  }

  /**
   * Return an iterator over named module parameters.
   *
   * @param prefix - Prefix to prepend to all parameter names
   * @param recurse - If true, recurses into child modules
   * @param remove_duplicate - If true, removes duplicate parameters
   */
  *namedParametersIter(
    prefix: string = "",
    recurse: boolean = true,
    remove_duplicate: boolean = true,
  ): Generator<[string, Parameter]> {
    const gen = this._namedMembers<unknown, unknown, Parameter>(
      (module) => Object.entries(module._parameters),
      prefix,
      recurse,
      remove_duplicate,
    );

    yield* gen;
  }

  /**
   * Return an array of named module parameters.
   *
   * @param prefix - Prefix to prepend to all parameter names
   * @param recurse - If true, recurses into child modules
   * @param remove_duplicate - If true, removes duplicate parameters
   */
  namedParameters(
    prefix: string = "",
    recurse: boolean = true,
    remove_duplicate: boolean = true,
  ): Array<[string, Parameter]> {
    return Array.from(
      this.namedParametersIter(prefix, recurse, remove_duplicate),
    );
  }

  /**
   * Return an iterator over module buffers.
   *
   * @param recurse - If true, recurses into child modules
   */
  *buffersIter(recurse: boolean = true): Generator<Buffer> {
    for (const [, buf] of this.namedBuffersIter("", recurse)) {
      yield buf;
    }
  }

  /**
   * Return an array of module buffers.
   *
   * @param recurse - If true, recurses into child modules
   */
  buffers(recurse: boolean = true): Buffer[] {
    return Array.from(this.buffersIter(recurse));
  }

  /**
   * Return an iterator over named module buffers.
   *
   * @param prefix - Prefix to prepend to all buffer names
   * @param recurse - If true, recurses into child modules
   * @param remove_duplicate - If true, removes duplicate buffers
   */
  *namedBuffersIter(
    prefix: string = "",
    recurse: boolean = true,
    remove_duplicate: boolean = true,
  ): Generator<[string, Buffer]> {
    const gen = this._namedMembers<unknown, unknown, Buffer>(
      (module) => Object.entries(module._buffers),
      prefix,
      recurse,
      remove_duplicate,
    );

    yield* gen;
  }

  /**
   * Return an array of named module buffers.
   *
   * @param prefix - Prefix to prepend to all buffer names
   * @param recurse - If true, recurses into child modules
   * @param remove_duplicate - If true, removes duplicate buffers
   */
  namedBuffers(
    prefix: string = "",
    recurse: boolean = true,
    remove_duplicate: boolean = true,
  ): Array<[string, Buffer]> {
    return Array.from(this.namedBuffersIter(prefix, recurse, remove_duplicate));
  }

  /**
   * Return an iterator over immediate children modules.
   */
  *childrenIter(): Generator<Module<unknown, unknown>> {
    for (const [, module] of this.namedChildrenIter()) {
      yield module;
    }
  }

  /**
   * Return an array of immediate children modules.
   */
  children(): Module<unknown, unknown>[] {
    return Array.from(this.childrenIter());
  }

  /**
   * Return an iterator over immediate named children modules.
   */
  *namedChildrenIter(): Generator<[string, Module]> {
    const memo = new Set<Module>();

    for (const [name, module] of Object.entries(this._modules)) {
      if (module !== null && !memo.has(module)) {
        memo.add(module);
        yield [name, module];
      }
    }
  }

  /**
   * Return an array of immediate named children modules.
   */
  namedChildren(): Array<[string, Module]> {
    return Array.from(this.namedChildrenIter());
  }

  /**
   * Return an iterator over all modules in the network.
   */
  *modulesIter(): Generator<Module> {
    for (const [, module] of this.namedModulesIter()) {
      yield module;
    }
  }

  /**
   * Return an array of all modules in the network.
   */
  modules(): Module[] {
    return Array.from(this.modulesIter());
  }

  /**
   * Return an iterator over all named modules in the network.
   *
   * @param memo - Set used to track already seen modules
   * @param prefix - Prefix to prepend to all module names
   * @param remove_duplicate - If true, removes duplicate modules
   */
  *namedModulesIter(
    memo: Set<Module> = new Set<Module>(),
    prefix: string = "",
    remove_duplicate: boolean = true,
  ): Generator<[string, Module]> {
    if (!memo.has(this as unknown as Module) || !remove_duplicate) {
      if (remove_duplicate) {
        memo.add(this as unknown as Module);
      }

      yield [prefix, this as unknown as Module];

      for (const [name, module] of Object.entries(this._modules)) {
        if (module === null) {
          continue;
        }

        const submodulePrefix = prefix + (prefix ? "." : "") + name;

        // Get generator from submodule's named modules
        yield* module.namedModulesIter(memo, submodulePrefix, remove_duplicate);
      }
    }
  }

  /**
   * Return an array of all named modules in the network.
   *
   * @param memo - Set used to track already seen modules
   * @param prefix - Prefix to prepend to all module names
   * @param remove_duplicate - If true, removes duplicate modules
   */
  namedModules(
    memo: Set<Module> = new Set<Module>(),
    prefix: string = "",
    remove_duplicate: boolean = true,
  ): Array<[string, Module]> {
    return Array.from(this.namedModulesIter(memo, prefix, remove_duplicate));
  }

  /**
   * Set the module in training mode.
   *
   * @param mode - Whether to set training mode (true) or evaluation mode (false)
   */
  train(mode: boolean = true): Module {
    if (typeof mode !== "boolean") {
      throw new TypeError("Training mode is expected to be boolean");
    }

    this.training = mode;

    for (const module of this.children()) {
      module.train(mode);
    }

    return this as unknown as Module;
  }

  /**
   * Set the module in evaluation mode (equivalent to train(false)).
   */
  eval(): Module {
    return this.train(false);
  }

  /**
   * Return state dictionary containing parameters and persistent buffers.
   */
  stateDict(): Record<string, Parameter | Buffer> {
    const state: Record<string, Parameter | Buffer> = {};

    // Add parameters
    for (const [name, param] of this.namedParameters()) {
      if (param !== null) {
        state[name] = param;
      }
    }

    // Add persistent buffers (exclude those in _nonPersistentBuffersSet)
    for (const [name, buf] of this.namedBuffers()) {
      if (buf !== null && !this._nonPersistentBuffersSet.has(name)) {
        state[name] = buf;
      }
    }

    return state;
  }

  /**
   * Registers a forward pre-hook which will be called before forward is called.
   *
   * @param hook - Function taking module and input arguments, returning modified input or undefined
   * @returns A handle that can be used to remove the hook
   */
  registerForwardPreHook(
    hook: (module: Module<Input, Output>, inputs: Input) => Input | undefined,
  ): RemovableHandle {
    const handle = new RemovableHandle(this._forwardPreHooks);
    this._forwardPreHooks.set(handle.id, hook);
    return handle;
  }

  /**
   * Registers a forward hook which will be called after forward is called.
   *
   * @param hook - Function taking module, input and output arguments, returning modified output or undefined
   * @returns A handle that can be used to remove the hook
   */
  registerForwardHook(
    hook: (
      module: Module<Input, Output>,
      inputs: Input,
      output: Output,
    ) => Output | undefined,
  ): RemovableHandle {
    const handle = new RemovableHandle(this._forwardHooks);
    this._forwardHooks.set(handle.id, hook);
    return handle;
  }

  /**
   * Abstract method that should be implemented by subclasses to define the
   * forward pass
   *
   * @param input - The input to the module
   * @returns The output from the module
   */
  // @ts-expect-error - This is an abstract method that will be implemented by
  // subclasses
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  forward(...input: Input): Output {
    throw new Error("Subclasses must implement the forward method");
  }
}

/**
 * ModuleList container for a sequence of modules
 * Modules can be added as ordered members of the ModuleList.
 * ModuleList can be indexed like an array to access contained modules.
 * This is similar to torch.nn.ModuleList in PyTorch.
 */
export class ModuleList<Input = unknown, Output = unknown> extends Module<
  Input,
  Output
> {
  private _modules_list: Module<Input, Output>[];
  private _proxy: ModuleList<Input, Output>;
  [key: number]: Module<Input, Output>;

  constructor(modules: Module<Input, Output>[] = []) {
    super();
    this._modules_list = [];

    // Add modules if provided in constructor
    if (modules && modules.length > 0) {
      for (let i = 0; i < modules.length; i++) {
        this.append(modules[i]);
      }
    }

    // Create a proxy to allow array-like indexing
    this._proxy = new Proxy(this, {
      get: (
        target: ModuleList<Input, Output>,
        prop: string | symbol,
      ): unknown => {
        // Forward numeric indices and string numeric indices to at()
        if (typeof prop === "string" && /^\d+$/.test(prop)) {
          return target.at(parseInt(prop, 10));
        } else if (typeof prop === "number") {
          return target.at(prop);
        }

        // Forward everything else to the target
        return Reflect.get(target, prop);
      },
      set: (
        target: ModuleList<Input, Output>,
        prop: string | symbol,
        value: unknown,
      ): boolean => {
        // Handle numeric indices and string numeric indices
        if (
          typeof prop === "string" &&
          /^\d+$/.test(prop) &&
          value instanceof Module
        ) {
          const index = parseInt(prop, 10);
          if (index >= 0 && index < target.length) {
            // Set using the set method
            target.set(index, value);
            return true;
          }
        }

        // Default behavior for other properties
        return Reflect.set(target, prop, value);
      },
    });
  }

  /**
   * Returns the proxy version of this ModuleList for array-like indexing
   */
  valueOf(): unknown {
    return this._proxy;
  }

  /**
   * Appends a module to the ModuleList
   *
   * @param module - Module to append
   * @returns The ModuleList instance for chaining
   */
  append(module: Module<Input, Output>): ModuleList<Input, Output> {
    const idx = this._modules_list.length;
    this.addModule(idx.toString(), module);
    this._modules_list.push(module);
    return this;
  }

  /**
   * Extends the ModuleList with an iterable of modules
   *
   * @param modules - Iterable of modules to add
   * @returns The ModuleList instance for chaining
   */
  extend(modules: Iterable<Module<Input, Output>>): ModuleList<Input, Output> {
    for (const module of modules) {
      this.append(module);
    }
    return this;
  }

  /**
   * Insert a module at a specific position
   *
   * @param index - Position to insert the module
   * @param module - Module to insert
   * @returns The ModuleList instance for chaining
   */
  insert(
    index: number,
    module: Module<Input, Output>,
  ): ModuleList<Input, Output> {
    if (index < 0) {
      index = Math.max(0, this._modules_list.length + index + 1);
    }

    // Insert the module in our list
    this._modules_list.splice(index, 0, module);

    // Re-register all modules with updated indices
    this._recreateModules();

    return this;
  }

  /**
   * Remove the first occurrence of a specific module
   *
   * @param module - Module to remove
   * @returns The ModuleList instance for chaining
   */
  remove(module: Module<Input, Output>): ModuleList<Input, Output> {
    const index = this._modules_list.indexOf(module);
    if (index !== -1) {
      this._modules_list.splice(index, 1);
      // Re-register all modules with updated indices
      this._recreateModules();
    }
    return this;
  }

  /**
   * Clear all modules from the ModuleList
   *
   * @returns The ModuleList instance for chaining
   */
  clear(): ModuleList<Input, Output> {
    this._modules_list = [];
    // Get all registered module keys using named modules iterator
    const moduleKeys = Array.from(this.namedChildrenIter()).map(([key]) => key);
    for (const key of moduleKeys) {
      this.addModule(key, null);
    }
    return this;
  }

  /**
   * Get the number of modules in the ModuleList
   */
  get length(): number {
    return this._modules_list.length;
  }

  /**
   * Access a module by index
   *
   * @param index - Index of the module to retrieve
   * @returns The module at the specified index
   */
  at(index: number): Module<Input, Output> | undefined {
    if (index < 0) {
      index = this._modules_list.length + index;
    }
    return this._modules_list[index];
  }

  /**
   * Set a module at a specific index
   *
   * @param index - Index where to set the module
   * @param module - Module to set at the index
   * @returns The ModuleList instance for chaining
   */
  set(index: number, module: Module<Input, Output>): ModuleList<Input, Output> {
    if (index < 0) {
      index = this._modules_list.length + index;
    }

    if (index >= 0 && index < this._modules_list.length) {
      // Remove old module registration
      this.addModule(index.toString(), null);
      // Add new module
      this.addModule(index.toString(), module);
      // Update in the array
      this._modules_list[index] = module;
    } else if (index === this._modules_list.length) {
      // If appending to the end, use append
      this.append(module);
    } else {
      throw new Error("Index out of bounds");
    }

    return this;
  }

  /**
   * Helper method to recreate the numbered module references after insertion or deletion
   * @private
   */
  private _recreateModules(): void {
    // Get all registered module keys using named modules iterator
    const moduleKeys = Array.from(this.namedChildrenIter()).map(([key]) => key);

    // Clear existing numbered modules
    for (const key of moduleKeys) {
      this.addModule(key, null);
    }

    // Re-add all modules with correct indices
    for (let i = 0; i < this._modules_list.length; i++) {
      this.addModule(i.toString(), this._modules_list[i]);
    }
  }

  /**
   * Array-like indexing
   *
   * @param index - Index to access
   * @returns The module at the specified index
   */
  get(index: number): Module<Input, Output> | undefined {
    return this.at(index);
  }

  /**
   * Get the list of modules in the ModuleList
   */
  get list(): Module<Input, Output>[] {
    return this._modules_list;
  }

  /**
   * Direct access to the ModuleList, using the key as a property
   */
  [Symbol.toPrimitive](_hint: number): ModuleList<Input, Output> {
    return this._proxy;
  }
}

/**
 * ModuleDict container for storing modules by name
 * Modules can be added to the ModuleDict with a name as key.
 * ModuleDict can be accessed like an object to retrieve contained modules.
 * This is similar to torch.nn.ModuleDict in PyTorch.
 */
export class ModuleDict<T extends Record<string, Module>> extends Module {
  private _proxy: ModuleDict<T>;

  constructor(modules: T = {} as T) {
    super();

    // Add modules if provided in constructor
    if (modules && Object.keys(modules).length > 0) {
      for (const [key, module] of Object.entries(modules)) {
        this.update({ [key]: module });
      }
    }

    // Create a proxy to allow object-like access
    this._proxy = new Proxy(this, {
      get: (target: ModuleDict<T>, prop: string | symbol): unknown => {
        // Forward string properties to get()
        if (
          typeof prop === "string" &&
          Object.prototype.hasOwnProperty.call(target._modules, prop)
        ) {
          return target.get(prop);
        }

        // Forward everything else to the target
        return Reflect.get(target, prop);
      },
      set: (
        target: ModuleDict<T>,
        prop: string | symbol,
        value: unknown,
      ): boolean => {
        // Handle string properties as modules
        if (
          typeof prop === "string" &&
          !prop.startsWith("_") &&
          value instanceof Module
        ) {
          target.set(prop, value);
          return true;
        }

        // Default behavior for other properties
        return Reflect.set(target, prop, value);
      },
    });
  }

  /**
   * Returns the proxy version of this ModuleDict for object-like access
   */
  valueOf(): ModuleDict<T> {
    return this._proxy;
  }

  /**
   * Add a module to the ModuleDict
   *
   * @param key - Key for the module
   * @param module - Module to add
   * @returns The ModuleDict instance for chaining
   */
  set(key: string, module: Module): ModuleDict<T> {
    if (typeof key !== "string") {
      throw new TypeError(`Key must be a string, got ${typeof key}`);
    }

    // Register module with the Module system
    this.addModule(key, module);

    // Keep track in our map
    this._modules[key] = module;

    return this;
  }

  /**
   * Update the ModuleDict with entries from another mapping
   *
   * @param modules - Object with module mappings to add
   * @returns The ModuleDict instance for chaining
   */
  update(modules: Record<string, Module>): ModuleDict<T> {
    for (const [key, module] of Object.entries(modules)) {
      this.set(key, module);
    }
    return this;
  }

  /**
   * Get a module by key
   *
   * @param key - Key of the module to retrieve
   * @returns The module with the specified key or undefined if not found
   */
  get(key: string): Module {
    return this._modules?.[key];
  }

  /**
   * Check if the ModuleDict contains a module with the specified key
   *
   * @param key - Key to check
   * @returns True if the key exists, false otherwise
   */
  has(key: string): boolean {
    return Object.prototype.hasOwnProperty.call(this._modules, key) ?? false;
  }

  /**
   * Remove a module by key
   *
   * @param key - Key of the module to remove
   * @returns The ModuleDict instance for chaining
   */
  delete(key: string): ModuleDict<T> {
    if (Object.prototype.hasOwnProperty.call(this._modules, key)) {
      // Remove from Module system
      this.addModule(key, null);

      // Remove from our map
      delete this._modules[key];
    }
    return this;
  }

  /**
   * Clear all modules from the ModuleDict
   *
   * @returns The ModuleDict instance for chaining
   */
  clear(): ModuleDict<T> {
    // Get all keys
    const keys = Object.keys(this._modules);

    // Remove all modules
    for (const key of keys) {
      this.delete(key);
    }

    return this;
  }

  /**
   * Get all keys in the ModuleDict
   *
   * @returns Array of keys
   */
  keys(): string[] {
    return Object.keys(this._modules);
  }

  /**
   * Get all values (modules) in the ModuleDict
   *
   * @returns Array of modules
   */
  values(): Module[] {
    return Object.values(this._modules) as Module[];
  }

  /**
   * Get all entries in the ModuleDict
   *
   * @returns Array of [key, module] pairs
   */
  entries(): Array<[string, Module]> {
    return Object.entries(this._modules) as Array<[string, Module]>;
  }

  /**
   * Get the number of modules in the ModuleDict
   */
  get length(): number {
    return Object.keys(this._modules).length;
  }

  /**
   * Iterate over the keys of the ModuleDict
   */
  *[Symbol.iterator](): IterableIterator<string> {
    yield* Object.keys(this._modules);
  }

  /**
   * Get the dictionary of modules in the ModuleDict
   */
  get dict(): T {
    return this._modules as T;
  }
}
