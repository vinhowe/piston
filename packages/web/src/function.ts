import { _popFunctionMode, _pushFunctionMode } from "@/wasm";

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
    _pushFunctionMode(this.mode);
  }
}
