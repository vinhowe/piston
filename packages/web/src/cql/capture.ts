import type { Module } from "@/nn/module";

import { PistonFunctionMode } from "@/function";
import { Tensor } from "@/tensor";
import { cloneReplaceTensorsDeep, forEachTensorDeep, RemovableHandle } from "@/utils";
import { Tensor_wasm } from "@/wasm";

import { DiagnosticError } from "./error";
import { createModuleAdapter } from "./moduleAdapter";
import { ModuleSelector } from "./moduleSelector";
import { parse } from "./parser";
import { createQueryContext, moduleFacetHook, parameterHook, tensorOpHook } from "./tensorHooks";
import {
  CaptureResult,
  LintDiagnostic,
  ModuleQueryContext,
  ModuleSelectionResult,
  OpQueryContext,
  ParameterQueryContext,
  QueryContext,
  QueryMatch,
  TensorQuery,
} from "./types";

export interface CaptureDiagnostics {
  parsing: LintDiagnostic[];
  selection: LintDiagnostic[];
}

interface ModuleRuntimeState {
  opContexts: WeakRef<OpQueryContext>[];
  inputFacetContexts: WeakRef<ModuleQueryContext>[];
  outputFacetContexts: WeakRef<ModuleQueryContext>[];
  parameterContexts: WeakRef<ParameterQueryContext>[];
}

// Minimal plan state marker; currently we only need membership to install hooks.
type ModulePlanState = true;

export class CapturePlan {
  private tensorQueries: TensorQuery[] = [];
  private activeModule: Module | null = null;
  private diagnostics: CaptureDiagnostics = { parsing: [], selection: [] };

  private _selectionResults: ModuleSelectionResult[] = [];
  private _planStatesByModule: WeakMap<Module, ModulePlanState> = new WeakMap();
  private _modulePathMap: Map<Module, string> = new Map();

  private hookedModules: WeakSet<Module> = new WeakSet();
  private forwardHookHandles: RemovableHandle[] = [];
  private activeSession: CaptureSession | null = null;

  parseScript(script: string): this {
    this.diagnostics.parsing = [];
    try {
      this.tensorQueries = parse(script);
    } catch (error) {
      if (error instanceof DiagnosticError) {
        this.diagnostics.parsing.push(error.toLintDiagnostic());
      } else {
        throw error;
      }
    }
    return this;
  }

  get queries(): TensorQuery[] {
    return this.tensorQueries;
  }

  clearQueries(): this {
    this.tensorQueries = [];
    return this;
  }

  hookModule(module: Module): this {
    this.activeModule = module;
    if (!this.tensorQueries.length) {
      throw new Error("No commands parsed. Call parseScript() before hooking a module.");
    }

    const { nodeAdapter, modelAdapter } = createModuleAdapter(this.activeModule);
    const moduleSelector = new ModuleSelector(modelAdapter);
    this._selectionResults = this.tensorQueries.map((command) =>
      moduleSelector.selectModules(command, nodeAdapter),
    );
    this.diagnostics.selection = this._selectionResults.flatMap((r) => r.diagnostics);

    // Build per-module plan membership
    this._planStatesByModule = new WeakMap();
    for (const result of this._selectionResults) {
      for (const node of result.matchedModules) {
        const mod = (node as { getModule?: () => Module }).getModule?.();
        if (!mod) continue;
        if (!this._planStatesByModule.has(mod)) {
          this._planStatesByModule.set(mod, true);
        }
      }
    }

    // Install hooks on modules that participate in any query
    const setupHooks = (name: string, mod: Module) => {
      if (this.hookedModules.has(mod)) return;

      const pre = mod.registerForwardPreHook((m, inputs) => {
        const s = this.activeSession;
        if (s) s.onPreForward(m, inputs);
      });

      const post = mod.registerForwardHook((m, _inputs, output) => {
        const s = this.activeSession;
        if (s) s.onPostForward(m, output);
      });

      this.forwardHookHandles.push(pre, post);
      this.hookedModules.add(mod);
      this._modulePathMap.set(mod, name);
    };

    for (const [name, mod] of this.activeModule.namedModules()) {
      if (this._planStatesByModule.get(mod)) setupHooks(name, mod);
    }

    return this;
  }

  createSession(): CaptureSession {
    if (!this.activeModule) {
      throw new Error("No active module hooked. Call hookModule() before creating a session.");
    }
    return new CaptureSession(this);
  }

  getDiagnostics(): CaptureDiagnostics {
    return this.diagnostics;
  }

  get selectionResults(): ModuleSelectionResult[] {
    return this._selectionResults;
  }

  get planStatesByModule(): WeakMap<Module, ModulePlanState> {
    return this._planStatesByModule;
  }

  get modulePathMap(): Map<Module, string> {
    return this._modulePathMap;
  }

  /**
   * @internal
   */
  setActiveSession(session: CaptureSession | null): void {
    this.activeSession = session;
  }

  [Symbol.dispose](): void {
    for (const handle of this.forwardHookHandles) {
      handle.remove();
    }
    this.forwardHookHandles.length = 0;
    this.hookedModules = new WeakSet();
  }
}

export class CaptureSession extends PistonFunctionMode {
  private plan: CapturePlan;
  private results: CaptureResult[] = [];
  private runtimeContexts: QueryContext[] = [];
  private runtimeStateByModule: WeakMap<Module, ModuleRuntimeState> = new WeakMap();
  private moduleStack: Module[] = [];
  private stateStack: ModuleRuntimeState[] = [];
  private paused: boolean = false;

  constructor(plan: CapturePlan) {
    super();
    this.plan = plan;

    const selection = this.plan.selectionResults;
    this.runtimeContexts = selection.map((r, idx) =>
      createQueryContext(r.query, r.matchedModules, idx),
    );

    for (let i = 0; i < selection.length; i++) {
      const r = selection[i];
      const ctx = this.runtimeContexts[i];
      for (const node of r.matchedModules) {
        const mod = (node as { getModule?: () => Module }).getModule?.();
        if (!mod) continue;

        let state = this.runtimeStateByModule.get(mod);
        if (!state) {
          state = {
            opContexts: [],
            inputFacetContexts: [],
            outputFacetContexts: [],
            parameterContexts: [],
          } satisfies ModuleRuntimeState;
          this.runtimeStateByModule.set(mod, state);
        }

        const target = r.query.target;
        if (target.kind === "op") {
          state.opContexts.push(new WeakRef(ctx as OpQueryContext));
        } else if (target.kind === "module") {
          if (target.site === "input") {
            state.inputFacetContexts.push(new WeakRef(ctx));
          } else {
            state.outputFacetContexts.push(new WeakRef(ctx));
          }
        } else if (target.kind === "parameter") {
          state.parameterContexts.push(new WeakRef(ctx as ParameterQueryContext));
        }
      }
    }

    this.plan.setActiveSession(this);
  }

  /**
   * @internal
   */
  onPreForward(module: Module, inputs: unknown): void {
    if (this.paused) {
      return;
    }

    this.moduleStack.push(module);

    const s =
      this.runtimeStateByModule.get(module) ??
      ({
        opContexts: [],
        inputFacetContexts: [],
        outputFacetContexts: [],
        parameterContexts: [],
      } satisfies ModuleRuntimeState);
    this.stateStack.push(s);

    for (const ctxWeak of s.inputFacetContexts) {
      const ctx = ctxWeak.deref();
      if (!ctx) {
        console.warn("Input facet context not found");
        continue;
      }
      hookWithRetainedGrads(() => moduleFacetHook(module, inputs, ctx), ctx, ctx.query);
    }

    for (const ctxWeak of s.parameterContexts) {
      // Parameters should already have gradient tracking
      const ctx = ctxWeak.deref();
      if (!ctx) {
        console.warn("Parameter context not found");
        continue;
      }
      parameterHook(module, ctx);
    }
  }

  /**
   * @internal
   */
  onPostForward(module: Module, output: unknown): void {
    if (this.paused) {
      return;
    }

    const s = this.runtimeStateByModule.get(module);
    if (s) {
      for (const ctxWeak of s.outputFacetContexts) {
        const ctx = ctxWeak.deref();
        if (!ctx) {
          console.warn("Output facet context not found");
          continue;
        }
        hookWithRetainedGrads(() => moduleFacetHook(module, output, ctx), ctx, ctx.query);
      }
    }

    this.stateStack.pop();
    this.moduleStack.pop();
  }

  /**
   * @internal
   */
  _pistonFunction<T>(
    func: <FT>(...args: unknown[]) => FT | Promise<FT>,
    _types: unknown[],
    args: unknown[],
    kwargs: Record<string, unknown>,
  ): Promise<T> | T {
    if (this.paused) {
      return func(...args, kwargs);
    }

    const after = (result: T) => {
      if (result instanceof Tensor_wasm) {
        const currentModule = this.moduleStack[this.moduleStack.length - 1];
        const activeState = this.stateStack[this.stateStack.length - 1];
        if (currentModule && activeState && activeState.opContexts.length > 0) {
          for (const ctxWeak of activeState.opContexts) {
            const ctx = ctxWeak.deref();
            if (!ctx) {
              console.warn("Op context not found");
              continue;
            }
            const matchState = ctx.matchStates?.get(currentModule);
            if (matchState) {
              hookWithRetainedGrads(
                () => tensorOpHook(currentModule, result, matchState, ctx),
                ctx,
                ctx.query,
              );
            }
          }
        }
      }
      return result;
    };

    const ret = func(...args, kwargs);
    if (ret instanceof Promise) {
      return ret.then(after);
    }
    return after(ret as T);
  }

  finalize(): CaptureResult[] {
    const results: CaptureResult[] = [];
    for (const ctx of this.runtimeContexts) {
      const moduleMatchGroups = Object.fromEntries(
        ctx.matches
          .entries()
          .map(([module, matches]) => [
            this.plan.modulePathMap.get(module)!,
            matches.flat() as QueryMatch[],
          ]),
      );

      if (ctx.query.gradient) {
        const handleGrad = (tensor: Tensor) => {
          const grad = tensor.grad;
          // If we're the one who requested the gradient, then the tensor will refer to it, which
          // refers back to the tensor as part of its construction graph, creating a reference cycle
          // and permanently leaking the op. So we prevent that here.
          if (tensor.retainsGrad) {
            tensor.grad = null;
          }
          return grad;
        };

        for (const [key, matches] of Object.entries(moduleMatchGroups)) {
          // Access gradient lazily at finalize time
          const grad = matches
            .map(({ tensorForGrad, transformationForGrad, ...rest }) => {
              let bufferTensor: Tensor | Tensor[] | undefined = undefined;
              if (tensorForGrad != null) {
                if (transformationForGrad) {
                  const replaced =
                    tensorForGrad instanceof Tensor
                      ? handleGrad(tensorForGrad)
                      : cloneReplaceTensorsDeep(tensorForGrad, handleGrad);
                  bufferTensor = transformationForGrad(replaced);
                } else {
                  bufferTensor = Array.isArray(tensorForGrad)
                    ? tensorForGrad.map((t) => handleGrad(t)).filter((t) => t !== undefined)
                    : handleGrad(tensorForGrad as Tensor);
                }
              }
              return {
                ...rest,
                bufferTensor,
              };
            })
            .filter(({ bufferTensor }) => bufferTensor !== undefined);
          if (grad) {
            moduleMatchGroups[key] = grad;
          } else {
            delete moduleMatchGroups[key];
          }
        }
      }

      if (Object.keys(moduleMatchGroups).length > 0) {
        results.push({ matches: moduleMatchGroups, source: ctx.query });
      }
    }
    this.results = results;
    return this.results;
  }

  pause(): void {
    this.paused = true;
  }

  resume(): void {
    this.paused = false;
  }

  getResults(): CaptureResult[] {
    return this.results;
  }

  filterResults(predicate: (result: CaptureResult) => boolean): CaptureResult[] {
    return this.results.filter(predicate);
  }

  [Symbol.dispose](): void {
    super[Symbol.dispose]();
    this.plan.setActiveSession(null);
  }
}

const hookWithRetainedGrads = (hook: () => void, ctx: QueryContext, parsedQuery: TensorQuery) => {
  if (!parsedQuery.gradient) {
    hook();
    return;
  }

  const original = ctx.matches;
  ctx.matches = new Map();
  hook();
  ctx.matches.forEach((t) =>
    t.forEach((obj) => {
      const { tensorForGrad } = obj;
      if (tensorForGrad != null) {
        forEachTensorDeep(tensorForGrad, (t: Tensor) => {
          if (t.isLeaf) {
            return;
          }
          t.retainGrad();
        });
      }
    }),
  );
  const merged = new Map(original);
  for (const [module, matches] of ctx.matches.entries()) {
    merged.set(module, [...(merged.get(module) ?? []), ...matches]);
  }
  ctx.matches = merged;
};
