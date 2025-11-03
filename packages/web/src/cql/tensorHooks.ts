import { Module } from "@/nn/module";
import { pistonForExpressions } from "@/pistonRuntime";
import { Tensor } from "@/tensor";

import { applyTransformations } from "./transformations";
import {
  CombinatorOpSelectorItem,
  ItemMatcher,
  ModelNode,
  ModuleQueryContext,
  ModuleQueryMatch,
  OpMatchState,
  OpQueryContext,
  OpQueryMatch,
  OpSelectorToken,
  ParameterQueryContext,
  ParameterQueryMatch,
  ParameterSelectorToken,
  ParsedSlice,
  QueryContext,
  SiblingChainOpMatchState,
  TensorQuery,
} from "./types";

export function createJsTransformation(expr: string): (input: unknown) => Tensor | Tensor[] {
  const fn = new Function("it", "piston", expr) as (
    input: unknown,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    _p: any,
  ) => Tensor | Tensor[];
  return function (this: Module | undefined, input: unknown) {
    return fn.call(this as unknown, input, pistonForExpressions);
  };
}

export function createOutputTransformation(
  slice: ParsedSlice | null,
  transformation: ((input: unknown) => Tensor | Tensor[]) | null,
): (input: unknown) => Tensor | Tensor[] {
  return function (this: Module | undefined, input: unknown) {
    return applyTransformations(this, input, slice, transformation);
  };
}

function createItemMatcher(token: OpSelectorToken): ItemMatcher {
  if (token.type === "name") {
    return {
      type: "name",
      name: token.value,
      match: null,
    };
  }
  if (token.type === "wildcard") {
    return {
      type: "wildcard",
      match: null,
    };
  }
  if (token.type === "regex") {
    return {
      type: "regex",
      regex: token.value,
      match: null,
    };
  }
  throw new Error("Invalid token type");
}

function validateOpSelector(opSelector: OpSelectorToken[]): void {
  if (opSelector.length === 0) {
    throw new Error("Op selector cannot be empty");
  }

  // Valid shapes:
  //  - single token (name, wildcard, or regex)
  //  - odd-length sequence alternating item and combinator (>=3)
  if (opSelector.length === 1) {
    const only = opSelector[0];
    if (!["name", "wildcard", "regex"].includes(only.type)) {
      throw new Error("Single-token selector must be a name, wildcard, or regex");
    }
    return;
  }

  if (opSelector.length % 2 === 0) {
    throw new Error(
      `Invalid tensor selector with ${opSelector.length} tokens (must be 1 or odd-length)`,
    );
  }

  for (let i = 0; i < opSelector.length; i++) {
    const tok = opSelector[i];
    if (i % 2 === 0) {
      if (!["name", "wildcard", "regex"].includes(tok.type)) {
        throw new Error("Expected name, wildcard, or regex at even positions in op selector");
      }
    } else {
      if (tok.type !== "combinator") {
        throw new Error("Expected combinator at odd positions in op selector");
      }
      if (!(tok.kind === "next-sibling" || tok.kind === "subsequent-sibling")) {
        // For ops, we currently support forward sibling combinators for matching
        throw new Error(
          `Unsupported op combinator '${tok.kind}' (only + and ~ are supported for ops)`,
        );
      }
    }
  }
}

function createMatchStateForModule(module: Module, opSelector: OpSelectorToken[]): OpMatchState {
  if (opSelector.length === 1) {
    return {
      module,
      type: "simple",
      current: createItemMatcher(opSelector[0]),
      matches: new Map(),
    };
  }

  // Build chain from tokens: item (0), [comb (1), item (2)]*
  const items: ItemMatcher[] = [];
  const relations: ("next-sibling" | "subsequent-sibling")[] = [];
  for (let i = 0; i < opSelector.length; i++) {
    const tok = opSelector[i];
    if (i % 2 === 0) {
      items.push(createItemMatcher(tok));
    } else {
      // i is odd => combinator
      relations.push((tok as CombinatorOpSelectorItem).kind);
    }
  }
  const chain: SiblingChainOpMatchState = {
    module,
    type: "sibling-chain",
    items,
    relations,
    progresses: [],
    matches: new Map(),
  };
  return chain;
}

function createMatchStates(
  opSelector: OpSelectorToken[],
  modules: Module[],
): Map<Module, OpMatchState> {
  validateOpSelector(opSelector);

  const matchStates = new Map<Module, OpMatchState>();
  modules.forEach((mod) => {
    matchStates.set(mod, createMatchStateForModule(mod, opSelector));
  });

  return matchStates;
}

export function createQueryContext(
  parsedQuery: TensorQuery,
  modules: ModelNode[],
  queryIndex: number,
): QueryContext {
  const jsTransformation = parsedQuery.jsPipe ? createJsTransformation(parsedQuery.jsPipe) : null;
  // const outputTransformation =
  //   parsedQuery.slice || jsTransformation
  //     ? createOutputTransformation(parsedQuery.slice, jsTransformation)
  //     : null;
  const outputTransformation = createOutputTransformation(parsedQuery.slice, jsTransformation);
  const matchStates =
    parsedQuery.target.kind === "op"
      ? createMatchStates(
          parsedQuery.target.selector,
          modules.map(
            (n) => (n as unknown as { getModule?: () => Module }).getModule?.() as Module,
          ),
        )
      : null;
  return {
    query: parsedQuery,
    queryIndex,
    transformation: outputTransformation,
    matchStates,
    matches: new Map(),
  };
}

function matchTensor(tensor: Tensor, matcher: ItemMatcher): boolean {
  if (matcher.type === "name") {
    return tensor.op().name === matcher.name;
  } else if (matcher.type === "wildcard") {
    return true;
  } else if (matcher.type === "regex") {
    return matcher.regex.test(tensor.op().name);
  }
  throw new Error("Invalid matcher type");
}

export function tensorOpHook(
  module: Module,
  rawTensor: Tensor,
  matchState: OpMatchState,
  context: OpQueryContext,
) {
  if (context.query.target.kind !== "op" || context.query.target.selector.length === 0) {
    throw new Error("Tensor hook must be called with an op selector");
  }

  if (!matchState) {
    throw new Error("Tensor hook must be called with a valid match state");
  }

  const tensor = Tensor._wrap(rawTensor);

  const pushMatch = () => {
    let bufferTensor: Tensor | unknown | undefined = undefined;
    let tensorForGrad: unknown | undefined = undefined;
    let transformationForGrad: ((input: unknown) => Tensor | Tensor[]) | null = null;

    if (!context.query.gradient) {
      if (context.transformation) {
        bufferTensor = context.transformation.call(module, tensor);
      } else {
        bufferTensor = Array.isArray(tensor)
          ? tensor.map((t) => t.debugTensor)
          : tensor.debugTensor;
      }
    } else {
      tensorForGrad = tensor;
      transformationForGrad = context.transformation
        ? (tensor: unknown) => context.transformation!.call(module, tensor)
        : null;
    }

    matches.push({
      op: tensor.op().name,
      queryIndex: context.queryIndex,
      bufferTensor: bufferTensor as Tensor | undefined,
      tensorForGrad,
      transformationForGrad,
      type: "op",
    });
  };

  const matches: OpQueryMatch[] = [];
  if (matchState.type === "simple") {
    if (matchTensor(tensor, matchState.current)) {
      pushMatch();
    }
  } else if (matchState.type === "next-sibling") {
    if (matchState.preceding.match && matchTensor(tensor, matchState.target)) {
      pushMatch();
      matchState.preceding.match = null;
    }

    if (matchTensor(tensor, matchState.preceding)) {
      matchState.preceding.match = tensor;
    } else {
      // This makes it "next" sibling. If we don't match on the next sibling, we reset the
      // preceding match.
      matchState.preceding.match = null;
    }
  } else if (matchState.type === "subsequent-sibling") {
    if (matchState.preceding.match && matchTensor(tensor, matchState.target)) {
      pushMatch();
      // We can match arbitrarily many subsequent siblings
      // matchState.preceding.match = null;
    }

    if (matchTensor(tensor, matchState.preceding)) {
      matchState.preceding.match = tensor;
    }
  } else if (matchState.type === "sibling-chain") {
    // Chain semantics: progress multiple active chains to support overlapping matches.
    const { items, relations } = matchState;

    const newProgresses: number[] = [];

    // 1) Advance existing progresses
    for (const currentIndex of matchState.progresses) {
      const expectingIndex = currentIndex + 1;
      if (expectingIndex >= items.length) {
        // Completed previously; don't carry forward
        continue;
      }

      const relation = relations[currentIndex];
      const expectingItem = items[expectingIndex];

      if (relation === "next-sibling") {
        // Must match immediately; if not matched, discard this progress
        if (matchTensor(tensor, expectingItem)) {
          newProgresses.push(expectingIndex);
        }
      } else {
        // subsequent-sibling: can skip; keep progress and also advance if match
        if (matchTensor(tensor, expectingItem)) {
          newProgresses.push(expectingIndex);
        } else {
          newProgresses.push(currentIndex);
        }
      }
    }

    // 2) Start new progress from head if current tensor matches head
    if (matchTensor(tensor, items[0])) {
      newProgresses.push(0);
    }

    matchState.progresses = newProgresses;

    // 3) Emit matches for any progresses reaching the final item; keep others for overlaps
    if (matchState.progresses.some((i) => i === items.length - 1)) {
      pushMatch();
      matchState.progresses = matchState.progresses.filter((i) => i < items.length - 1);
    }
  } else {
    throw new Error("Tensor hook must be called with a valid match state");
  }

  if (matches.length) {
    matchState.matches.set(module, [...(matchState.matches.get(module) ?? []), ...matches]);
    context.matches.set(module, [...(context.matches.get(module) ?? []), ...matches]);
  }
}

// Can be on the beginning or end of the module. Luckily which end is figured out by whatever
// actually assigns this as a hook.
export function moduleFacetHook(module: Module, input: unknown, context: ModuleQueryContext) {
  if (context.query.target.kind !== "module") {
    throw new Error("Module facet hook must be called with a module target");
  }

  const isGrad = context.query.gradient;

  let bufferTensor: (Tensor | Tensor[]) | unknown | undefined = undefined;
  let tensorForGrad: unknown | undefined = undefined;
  let transformationForGrad: ((input: unknown) => Tensor | Tensor[]) | null = null;

  if (!isGrad) {
    bufferTensor = input;
    if (context.transformation) {
      bufferTensor = context.transformation.call(module, bufferTensor);
    }
  } else {
    tensorForGrad = input;
    transformationForGrad = context.transformation
      ? (tensor: unknown) => context.transformation!.call(module, tensor)
      : null;
  }

  // TODO: Implement lazy gradient retrieval (will be some annoying thinking; we could do a Promise-
  // -type thing, but idk)
  // if (matchState.parameter === "gradient") {
  //   output = applyToTensorLike(output, (tensor) => {
  //     tensor.grad;
  //   });
  // }
  // TODO: Turn them into some sort of TrackingTensor-like wrapper (especially because that might
  // be the way to get debugTensors)
  // context.matches.set(module, Array.isArray(output) ? (output as Tensor[]) : [output as Tensor]);

  // if (Array.isArray(bufferTensor)) {
  //   for (const t of bufferTensor) {
  //     if (!(t instanceof Tensor)) {
  //       // TODO: Make this more granular, and show up in the linter
  //       // throw Error("Module facet hook must be called with a tensor");
  //       console.warn("Module facet hook must be called with a tensor");
  //       return;
  //     }
  //   }
  // } else {
  //   if (!(bufferTensor instanceof Tensor)) {
  //     // TODO: Make this more granular, and show up in the linter
  //     // throw Error("Module facet hook must be called with a tensor");
  //     console.warn("Module facet hook must be called with a tensor");
  //     return;
  //   }
  // }

  context.matches.set(module, [
    ...(context.matches.get(module) ?? []),
    {
      queryIndex: context.queryIndex,
      bufferTensor: bufferTensor as Tensor | Tensor[] | undefined,
      tensorForGrad,
      transformationForGrad,
      type: "module",
      site: context.query.target.site,
    } as ModuleQueryMatch,
  ]);
}

function matchesParameterSelector(selector: ParameterSelectorToken, name: string): boolean {
  if (selector.type === "wildcard") return true;
  if (selector.type === "name") return selector.value === name;
  if (selector.type === "regex") return selector.value.test(name);
  return false;
}

export function parameterHook(module: Module, context: ParameterQueryContext): void {
  if (context.query.target.kind !== "parameter") {
    throw new Error("Parameter hook must be called with a parameter target");
  }

  const isGrad = context.query.gradient;

  const selector = context.query.target.selector;

  const matches: ParameterQueryMatch[] = [];

  const pushMatch = (name: string, tensor: Tensor | Tensor[], type: "parameter" | "buffer") => {
    if (tensor && matchesParameterSelector(selector, name)) {
      matches.push({
        parameter: name,
        queryIndex: context.queryIndex,
        tensorForGrad: isGrad ? tensor : undefined,
        transformationForGrad: isGrad
          ? context.transformation
            ? (tensor: unknown) => context.transformation!.call(module, tensor)
            : null
          : null,
        bufferTensor: isGrad
          ? undefined
          : context.transformation
            ? context.transformation.call(module, tensor)
            : tensor,
        // A little bit gross that we can have parameterType: "parameter" and "buffer"
        parameterType: type,
        type: "parameter",
      });
    }
  };

  // Immediate parameters only (no recursion)
  for (const [name, param] of module.namedParameters("", false)) {
    pushMatch(name, param, "parameter");
  }
  // Immediate buffers only (no recursion)
  for (const [name, buffer] of module.namedBuffers("", false)) {
    pushMatch(name, buffer, "buffer");
  }

  context.matches.set(module, [...(context.matches.get(module) ?? []), ...matches]);
}
