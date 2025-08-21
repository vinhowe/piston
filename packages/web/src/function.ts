import { _popFunctionMode, _pushFunctionMode } from "@/wasm";
export abstract class PistonFunctionMode {
  constructor() {
    _pushFunctionMode(this);
  }

  [Symbol.dispose]() {
    _popFunctionMode();
  }

  abstract _pistonFunction(
    func: (...args: unknown[]) => unknown,
    types: unknown[],
    args: unknown[],
    kwargs: Record<string, unknown>,
  ): unknown;
}

export class BasePistonFunctionMode extends PistonFunctionMode {
  _pistonFunction(
    func: (...args: unknown[]) => unknown,
    _types: unknown[],
    args: unknown[],
    kwargs: Record<string, unknown>,
  ): unknown {
    if (!kwargs) {
      kwargs = {};
    }
    return func(...args, kwargs);
  }
}
