import { _popFunctionMode, _popMarkStepMode, _pushFunctionMode, _pushMarkStepMode } from "@/wasm";

export abstract class PistonFunctionMode {
  constructor() {
    _pushFunctionMode(this);
  }

  [Symbol.dispose]() {
    _popFunctionMode();
  }

  abstract _pistonFunction<T>(
    func: <FT>(...args: unknown[]) => FT | Promise<FT>,
    types: unknown[],
    args: unknown[],
    kwargs: Record<string, unknown>,
  ): T | Promise<T>;
}

export class BasePistonFunctionMode extends PistonFunctionMode {
  _pistonFunction<T>(
    func: <FT>(...args: unknown[]) => FT | Promise<FT>,
    _types: unknown[],
    args: unknown[],
    kwargs: Record<string, unknown>,
  ): T | Promise<T> {
    if (!kwargs) {
      kwargs = {};
    }
    return func(...args, kwargs);
  }
}

export class FunctionModeGuard {
  public readonly mode: PistonFunctionMode;

  constructor() {
    this.mode = _popFunctionMode();
  }

  [Symbol.dispose]() {
    if (this.mode) {
      _pushFunctionMode(this.mode);
    }
  }
}

export abstract class PistonMarkStepMode {
  constructor() {
    _pushMarkStepMode(this);
  }

  [Symbol.dispose]() {
    _popMarkStepMode();
  }

  // If returns undefined/null, default mark_step is executed.
  abstract _pistonMarkStep(original: () => Promise<void>): void | Promise<void>;
}

export class BasePistonMarkStepMode extends PistonMarkStepMode {
  _pistonMarkStep(original: () => Promise<void>): void | Promise<void> {
    return original();
  }
}
