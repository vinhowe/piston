import * as globals from "@/globals";
import * as nn from "@/nn";
import * as optim from "@/optim";
import { Tensor } from "@/tensor";
import * as utils from "@/utils";
import * as wasm from "@/wasm";

// This collosal hack intentionally aggregates submodules that do NOT import from the top-level
// barrel to avoid circular initialization issues for expressions.

// Provide both nested namespaces (piston.optim.AdamW) and top-level names (piston.AdamW) matching
// the barrel export access pattern, without importing the barrel itself.

// Worst part about this whole setup is we need to manually sync it with any other changes we care
// about.

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const pistonForExpressions: any = new Proxy(
  {
    nn,
    optim,
    globals,
    wasm,
    utils,
    Tensor,
  },
  {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    get(target: Record<string, any>, prop: string | symbol): any {
      if (typeof prop === "symbol") {
        return (target as unknown as Record<symbol, unknown>)[prop];
      }
      if (prop in target) return (target as Record<string, unknown>)[prop];
      if (prop in optim) return (optim as unknown as Record<string, unknown>)[prop];
      if (prop in nn) return (nn as unknown as Record<string, unknown>)[prop];
      if (prop in globals) return (globals as unknown as Record<string, unknown>)[prop];
      if (prop in wasm) return (wasm as unknown as Record<string, unknown>)[prop];
      if (prop in utils) return (utils as unknown as Record<string, unknown>)[prop];
      return undefined;
    },
    // Preserve property existence checks
    has(_target, prop: string | symbol): boolean {
      if (typeof prop === "symbol") return false;
      return prop in optim || prop in nn || prop in globals || prop in wasm || prop in utils;
    },
  },
);
