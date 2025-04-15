import { Module } from "@/nn/module";
import { Optimizer, ParamGroup } from "@/optim";
import { Parameter } from "@/parameter";
import { OpDescription, Tensor } from "@/tensor";

export interface BaseScopeItem {
  type:
    | "module"
    | "optimizer"
    | "optimizer-param-group"
    | "optimizer-param-update";
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

export interface OptimizerParamGroupScopeItem extends BaseScopeItem {
  type: "optimizer-param-group";
  optimizer: Optimizer;
  paramGroup: ParamGroup;
}

export interface OptimizerParamUpdateScopeItem extends BaseScopeItem {
  type: "optimizer-param-update";
  optimizer: Optimizer;
  parameter: Parameter;
}

export type ScopeItem =
  | ModuleScopeItem
  | OptimizerScopeItem
  | OptimizerParamGroupScopeItem
  | OptimizerParamUpdateScopeItem;

export const tensorScopeStack: ScopeItem[] = [];

interface TrackState {
  stack: TrackedTensor[];
  alreadyTracked: Set<number>;
}

export const trackStacks: Record<number, TrackState> = {};

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
}

export interface TensorTrackOptions {
  dependency?: boolean;
}

export function trackTensor(tensor: Tensor, options?: TensorTrackOptions) {
  const { dependency = false } = options ?? {};

  for (const stackId in trackStacks) {
    const id = tensor.id();
    if (trackStacks[stackId].alreadyTracked.has(id)) {
      continue;
    }
    trackStacks[stackId].stack.push({
      id,
      op: tensor.op(),
      shape: tensor.shape,
      srcIds: tensor.srcIds(),
      nameOnParent: tensor.nameOnParent,
      scope: tensor.scope,
      // This presumably breaks any non-explicit inplacing, which makes
      // observing inplacing storage patterns difficult.
      tensor: tensor,
      // tensor: tensor.requiresGrad() ? tensor : undefined,
      debugTensor: tensor.debugTensor(),
      dependency,
    });
    trackStacks[stackId].alreadyTracked.add(id);
  }
  return tensor;
}

export function trackTensors(tensors: Tensor[], options?: TensorTrackOptions) {
  const { dependency = false } = options ?? {};
  for (const tensor of tensors) {
    trackTensor(tensor, { dependency });
  }
}

export interface WithScopeOptions {
  replace?: boolean;
}

export function withScope<T>(
  scope: ScopeItem[] | ((scope: ScopeItem[]) => ScopeItem[]),
  f: () => T,
  options: WithScopeOptions = {},
): T {
  const { replace = false } = options;
  const originalScope = [...tensorScopeStack];
  const newScope =
    typeof scope === "function" ? scope(tensorScopeStack) : scope;
  if (replace) {
    tensorScopeStack.length = 0;
  }
  tensorScopeStack.push(...newScope);
  try {
    return f();
  } finally {
    tensorScopeStack.length = 0;
    tensorScopeStack.push(...originalScope);
  }
}

export function nameFromScope(scope: ScopeItem[]) {
  return (
    scope
      .map((item) => (item.name?.includes(".") ? `[${item.name}]` : item.name))
      .join(".") ?? "."
  );
}

export function tensorName(tensor: Tensor) {
  return (
    nameFromScope(tensor.scope ?? []) +
    (tensor.nameOnParent ? `.${tensor.nameOnParent}` : "")
  );
}

let stackCounter = 0;

export function track<Output>(
  module: Module<unknown, Output>,
  ...args: Parameters<typeof module.forward>
): [Output, TrackedTensor[]] {
  const stackId = stackCounter++;
  const tensors: TrackedTensor[] = [];
  trackStacks[stackId] = {
    stack: tensors,
    alreadyTracked: new Set(),
  };
  try {
    const result = module.forward(...args);
    return [result, tensors];
  } finally {
    delete trackStacks[stackId];
  }
}

export async function trackOptimizerStep(optimizer: Optimizer) {
  const stackId = stackCounter++;
  const tensors: TrackedTensor[] = [];
  trackStacks[stackId] = {
    stack: tensors,
    alreadyTracked: new Set(),
  };
  try {
    await optimizer.step();
  } finally {
    delete trackStacks[stackId];
  }
  return tensors;
}
