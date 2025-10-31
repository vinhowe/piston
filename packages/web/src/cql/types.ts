import { Tensor } from "@/tensor";

// Types for the selector tokens
export type ModuleSelectorItemType =
  | "name"
  | "type"
  | "wildcard"
  | "name-regex"
  | "type-regex"
  | "combinator";

export interface SourceReferencedItem {
  from: number;
  to: number;
  source: string;
}

type OptionallySourceReferencedItem = Partial<SourceReferencedItem>;

interface BaseModuleSelectorItem extends SourceReferencedItem {
  type: ModuleSelectorItemType;
}

export interface Indexable {
  index: number | null;
}

export interface NameModuleSelectorItem extends BaseModuleSelectorItem, Indexable {
  type: "name";
  value: string;
}

export interface TypeModuleSelectorItem extends BaseModuleSelectorItem, Indexable {
  type: "type";
  value: string;
}

export interface RegexModuleSelectorItem extends BaseModuleSelectorItem {
  type: "name-regex";
  value: RegExp;
}

export interface RegexTypeModuleSelectorItem extends BaseModuleSelectorItem {
  type: "type-regex";
  value: RegExp;
}

export interface WildcardModuleSelectorItem extends BaseModuleSelectorItem {
  type: "wildcard";
}

export interface CombinatorModuleSelectorItem extends BaseModuleSelectorItem {
  type: "combinator";
  kind: "child" | "descendant" | "next-sibling" | "subsequent-sibling";
}

export type ModuleSelectorToken =
  | NameModuleSelectorItem
  | TypeModuleSelectorItem
  | RegexModuleSelectorItem
  | RegexTypeModuleSelectorItem
  | WildcardModuleSelectorItem
  | CombinatorModuleSelectorItem;

export type OpSelectorItemType = "name" | "wildcard" | "regex" | "combinator";

export interface BaseOpSelectorItem extends SourceReferencedItem {
  type: OpSelectorItemType;
}

export interface NameOpSelectorItem extends BaseOpSelectorItem {
  type: "name";
  value: string;
}

export interface WildcardOpSelectorItem extends BaseOpSelectorItem {
  type: "wildcard";
}

export interface RegexOpSelectorItem extends BaseOpSelectorItem {
  type: "regex";
  value: RegExp;
}

export interface CombinatorOpSelectorItem extends BaseOpSelectorItem {
  type: "combinator";
  kind: "next-sibling" | "subsequent-sibling";
}

export type OpSelectorToken =
  | NameOpSelectorItem
  | WildcardOpSelectorItem
  | RegexOpSelectorItem
  | CombinatorOpSelectorItem;

// Types for slice expressions
export type SliceArgument =
  | {
      start: number | null;
      stop: number | null;
      step: number | null;
      isSingleIndex: boolean;
    }
  | "ellipsis";

export type ParsedSlice = {
  items: SliceArgument[];
} & SourceReferencedItem;

// Parsed query structure
export type ParameterSelectorItemType = "name" | "wildcard" | "regex";

export interface BaseParameterSelectorItem extends SourceReferencedItem {
  type: ParameterSelectorItemType;
}

export interface NameParameterSelectorItem extends BaseParameterSelectorItem {
  type: "name";
  value: string;
}

export interface WildcardParameterSelectorItem extends BaseParameterSelectorItem {
  type: "wildcard";
}

export interface RegexParameterSelectorItem extends BaseParameterSelectorItem {
  type: "regex";
  value: RegExp;
}

export type ParameterSelectorToken =
  | NameParameterSelectorItem
  | WildcardParameterSelectorItem
  | RegexParameterSelectorItem;

export interface ModuleSelectionTarget extends OptionallySourceReferencedItem {
  kind: "module";
  site: "input" | "output";
}

export interface OpSelectionTarget extends SourceReferencedItem {
  kind: "op";
  selector: OpSelectorToken[];
}

export interface ParameterSelectionTarget extends SourceReferencedItem {
  kind: "parameter";
  selector: ParameterSelectorToken;
}

export type SelectionTarget = ModuleSelectionTarget | OpSelectionTarget | ParameterSelectionTarget;

export interface ParsedScale extends SourceReferencedItem {
  value: number;
  unit: "percent" | "absolute";
}

export interface ParsedGradient extends SourceReferencedItem {
  value: boolean;
}

export interface ParsedNorm extends SourceReferencedItem {
  value: boolean;
}

export interface ParsedLabel extends SourceReferencedItem {
  value: string;
}

export interface ParsedFacets extends SourceReferencedItem {
  gradient?: ParsedGradient;
  norm?: ParsedNorm;
  scale?: ParsedScale;
  label?: ParsedLabel;
}

export interface ParsedJsPipe extends SourceReferencedItem {
  value: string;
}

export interface ParsedQuery extends SourceReferencedItem {
  moduleSelector: ModuleSelectorToken[];
  target?: SelectionTarget;
  facets?: ParsedFacets;
  slice?: ParsedSlice;
  jsPipe?: ParsedJsPipe;
}

export interface TensorQuery {
  parsedQuery: ParsedQuery;
  moduleSelector: ModuleSelectorToken[];
  target: SelectionTarget;
  gradient: boolean;
  norm: boolean;
  scale: number | null;
  label?: string | undefined;
  slice: ParsedSlice | null;
  jsPipe: string | null;
}

// Model adapter interface for evaluating the DSL against actual models
export interface ModelNode<ParameterType = Tensor> {
  name: string;
  typeName: string;
  parent: ModelNode<ParameterType> | null;
  getChildren(): ModelNode<ParameterType>[];
  getChild(nameOrIndex: string | number): ModelNode<ParameterType> | undefined;
  getParameters(): Record<string, ParameterType>;
  matchesRegex(regex: RegExp): boolean;
  path(): string[];
}

export interface ModelAdapter<ParameterType = Tensor> {
  getRootModules(): ModelNode<ParameterType>[];
  getParameter(node: ModelNode<ParameterType>, name: string): ParameterType | undefined;
  getChild(node: ModelNode<ParameterType>, name: string): ModelNode<ParameterType> | undefined;
  findDescendants(
    node: ModelNode<ParameterType>,
    predicate: (node: ModelNode<ParameterType>) => boolean,
    results: ModelNode<ParameterType>[],
    maxDepth?: number,
  ): void;
}

export interface ModuleSelectorOptions {
  includeDescendants?: boolean;
  maxDepth?: number;
}

export interface ModuleSelectionResult<ParameterType = Tensor> {
  matchedModules: ModelNode<ParameterType>[];
  query: TensorQuery;
  diagnostics: LintDiagnostic[];
  context?: {
    matchCount?: number;
    [key: string]: unknown;
  };
}

/**
 * Lint diagnostic for reporting errors with source locations
 */
export interface LintDiagnostic {
  from: number;
  to: number;
  message: string;
  severity: "error" | "warning" | "info";
  source?: string;
  code?: string | number;
}
