import type { SyntaxNode } from "@lezer/common";
import type { EditorView } from "codemirror";

import {
  type Completion,
  type CompletionContext,
  insertCompletionText,
  pickedCompletion,
  snippetCompletion,
} from "@codemirror/autocomplete";
import { syntaxTree } from "@codemirror/language";

import type { LintDiagnostic, ModuleSelectorToken, OpSelectorToken, TensorQuery } from "@/types";

import type { ParameterSelectorToken } from "./types";
import type { ModelNode } from "./types";

import {
  CaptureIndexModelAdapter,
  CaptureIndexModelNode,
  type IndexOperation,
  type IndexParameter,
  type IndexState,
} from "./captureIndex";
import { ModuleSelector } from "./moduleSelector";
import { flattenModules, flattenOpSelectors } from "./parser";

function formatChildrenCount(childrenCount: number): string {
  if (childrenCount === 0) return "";
  if (childrenCount === 1) return " (1 child)";
  return ` (${childrenCount} children)`;
}

function moduleAutocompleteQuery(
  state: IndexState,
  kind: "descendant" | "child" | "next-sibling" | "subsequent-sibling",
  selector: ModuleSelectorToken[],
): Completion[] {
  const modelAdapter = new CaptureIndexModelAdapter(state);

  const newSelector = [...selector];

  const wildcard = {
    type: "wildcard",
  } as ModuleSelectorToken;

  if (kind === "descendant") {
    if (newSelector.length > 0) {
      newSelector.push({
        type: "combinator",
        kind: "descendant",
      } as ModuleSelectorToken);
    }
    newSelector.push(wildcard);
  } else if (kind === "child") {
    newSelector.push(
      {
        type: "combinator",
        kind: "child",
      } as ModuleSelectorToken,
      wildcard,
    );
  } else if (kind === "next-sibling") {
    newSelector.push(
      {
        type: "combinator",
        kind: "next-sibling",
      } as ModuleSelectorToken,
      wildcard,
    );
  } else if (kind === "subsequent-sibling") {
    newSelector.push(
      {
        type: "combinator",
        kind: "subsequent-sibling",
      } as ModuleSelectorToken,
      wildcard,
    );
  }

  const query = { moduleSelector: newSelector } as TensorQuery;

  const buildModuleNameOptions = (name: string, module: ModelNode<IndexParameter>): Completion => ({
    label: name,
    type: "property",
    detail: `${module.typeName}${formatChildrenCount(module.getChildren().length)}`,
    boost: 2,
  });

  const handlePossiblyIndexedModuleName = (module: ModelNode<IndexParameter>): Completion[] => {
    if (module.typeName === "ModuleList") {
      return module
        .getChildren()
        .map((child) => buildModuleNameOptions(`${module.name}[${child.name}]`, child));
    }
    return [buildModuleNameOptions(module.name, module)];
  };

  const selectResult = new ModuleSelector(modelAdapter).selectModules(query, [
    new CaptureIndexModelNode(state.root),
  ]);

  const options = selectResult.matchedModules
    .filter((m) => m.name !== "" && !m.name.match(/^\d+$/))
    .flatMap((m) => [
      ...handlePossiblyIndexedModuleName(m),
      {
        label: `.${m.typeName}`,
        type: "class",
      },
    ]);

  if (selectResult.matchedModules.length > 0) {
    options.unshift({
      label: "*",
      type: "keyword",
      boost: 4,
      detail: `(${selectResult.matchedModules.length})`,
    });
  }

  return options;
}

function matchesOpSelectorToken(operation: IndexOperation, token: OpSelectorToken): boolean {
  if (token.type === "wildcard") return true;
  if (token.type === "name") {
    return token.value === operation.name;
  }
  if (token.type === "regex") {
    return token.value.test(operation.name);
  }
  return false;
}

function filterOpsBySelector(
  state: IndexState,
  selector: OpSelectorToken[] | null,
  initialOps: IndexOperation[],
): { matched: IndexOperation[]; failedAt: OpSelectorToken | null } {
  const order = state.opExecutionOrder;
  if (!selector || selector.length === 0) {
    return { matched: initialOps, failedAt: null };
  }

  const items: OpSelectorToken[] = [];
  const relations: ("next-sibling" | "subsequent-sibling")[] = [];
  for (const token of selector) {
    if (token.type === "combinator") {
      relations.push(token.kind);
    } else {
      items.push(token);
    }
  }

  const seedIndices = new Set<number>(initialOps.map((o) => o.executionIndex));
  let failedAt: OpSelectorToken | null = null;

  let current = new Set<number>();
  for (const idx of seedIndices) {
    const op = order[idx];
    if (op && matchesOpSelectorToken(op, items[0])) current.add(idx);
  }
  if (current.size === 0) {
    failedAt = items[0] ?? null;
  }

  for (let i = 0; i < relations.length; i++) {
    const relation = relations[i];
    const nextItem = items[i + 1];
    const nextSet = new Set<number>();
    for (const idx of current) {
      if (relation === "next-sibling") {
        const j = idx + 1;
        const op = order[j];
        if (op && matchesOpSelectorToken(op, nextItem)) nextSet.add(j);
      } else {
        for (let j = idx + 1; j < order.length; j++) {
          const op = order[j];
          if (matchesOpSelectorToken(op, nextItem)) nextSet.add(j);
        }
      }
    }
    current = nextSet;
    if (current.size === 0) {
      failedAt = nextItem ?? null;
      break;
    }
  }

  const matched = Array.from(current)
    .sort((a, b) => a - b)
    .map((i) => order[i]);
  return { matched, failedAt };
}

function opAutocompleteQuery(
  state: IndexState,
  kind: "op" | "op-next-sibling" | "op-subsequent-sibling",
  moduleSelector: ModuleSelectorToken[],
  opSelector: OpSelectorToken[] | null,
): Completion[] {
  const modelAdapter = new CaptureIndexModelAdapter(state);
  const query = { moduleSelector } as TensorQuery;

  const initialOps: IndexOperation[] = [];
  new ModuleSelector(modelAdapter)
    .selectModules(query, [new CaptureIndexModelNode(state.root)])
    .matchedModules.forEach((m) => {
      initialOps.push(...(m as CaptureIndexModelNode).getOperations());
    });

  let effectiveSelector: OpSelectorToken[] | null = opSelector ? [...opSelector] : null;
  if (kind === "op-next-sibling") {
    effectiveSelector = [
      ...(effectiveSelector || []),
      { type: "combinator", kind: "next-sibling" } as OpSelectorToken,
      { type: "wildcard" } as OpSelectorToken,
    ];
  } else if (kind === "op-subsequent-sibling") {
    effectiveSelector = [
      ...(effectiveSelector || []),
      { type: "combinator", kind: "subsequent-sibling" } as OpSelectorToken,
      { type: "wildcard" } as OpSelectorToken,
    ];
  } else if (kind !== "op") {
    throw new Error(`Invalid op autocomplete kind: ${kind}`);
  }

  const { matched } = filterOpsBySelector(state, effectiveSelector, initialOps);
  const grouped = new Map<string, IndexOperation[]>();
  for (const op of matched) {
    grouped.set(op.name, [...(grouped.get(op.name) || []), op]);
  }

  const options: Completion[] = Array.from(grouped.entries()).map(([name, operations]) => ({
    label: name,
    type: "function",
    detail: operations.length > 1 ? `(${operations.length})` : undefined,
  }));

  if (matched.length > 0) {
    options.unshift({ label: "*", type: "keyword", boost: 4, detail: `(${matched.length})` });
  }

  return options;
}

function parameterAutocompleteQuery(
  state: IndexState,
  selector: ModuleSelectorToken[],
): Completion[] {
  const modelAdapter = new CaptureIndexModelAdapter(state);
  const query = { moduleSelector: selector } as TensorQuery;

  let paramCount = 0;
  const consolidatedParameters = new Map<string, IndexParameter[]>();

  new ModuleSelector(modelAdapter)
    .selectModules(query, [new CaptureIndexModelNode(state.root)])
    .matchedModules.forEach((m) => {
      const parameters = (m as CaptureIndexModelNode).getParameters();
      Object.keys(parameters).forEach((p) => {
        consolidatedParameters.set(p, [...(consolidatedParameters.get(p) || []), parameters[p]]);
      });
      paramCount++;
    });

  const options: Completion[] = Array.from(consolidatedParameters.entries()).map(
    ([name, parameters]) => ({
      label: name,
      type: "property",
      detail: parameters.length > 1 ? `(${parameters.length})` : undefined,
    }),
  );

  if (paramCount > 0) {
    options.unshift({ label: "*", type: "keyword", boost: 4, detail: `(${paramCount})` });
  }

  return options;
}

function matchesParameterSelectorToken(name: string, token: ParameterSelectorToken): boolean {
  if (token.type === "wildcard") return true;
  if (token.type === "name") return token.value === name;
  if (token.type === "regex") return token.value.test(name);
  return false;
}

export function validateScriptAgainstIndex(
  script: TensorQuery[],
  state: IndexState,
): LintDiagnostic[] {
  const diagnostics: LintDiagnostic[] = [];

  const modelAdapter = new CaptureIndexModelAdapter(state);
  const selector = new ModuleSelector(modelAdapter);
  const root = new CaptureIndexModelNode(state.root);

  for (const q of script) {
    const selection = selector.selectModules(q, [root]);
    if (selection.diagnostics?.length || selection.matchedModules.length === 0) {
      const d = selection.diagnostics?.[0];
      if (d) {
        diagnostics.push({ ...d, severity: "warning" });
      } else if (q.moduleSelector?.[0]) {
        const t = q.moduleSelector[0];
        diagnostics.push({
          from: t.from,
          to: t.to,
          source: t.source,
          message: "No modules matched",
          severity: "warning",
        });
      }
      continue;
    }

    // Target is module, and we have modules
    if (q.target?.kind === "module") {
      continue;
    }

    // Get modules that matched for op/parameter check
    const matchedModules = selection.matchedModules as CaptureIndexModelNode[];

    if (q.target?.kind === "parameter") {
      const token = q.target.selector;
      let any = false;
      for (const m of matchedModules) {
        const params = m.getParameters();
        for (const name of Object.keys(params)) {
          if (matchesParameterSelectorToken(name, token)) {
            any = true;
            break;
          }
        }
        if (any) break;
      }
      if (!any) {
        diagnostics.push({
          from: token.from,
          to: token.to,
          source: token.source,
          message: `No parameters matched '${token.source}'`,
          severity: "warning",
        });
      }
      continue;
    }

    if (q.target?.kind === "op") {
      const initialOps: IndexOperation[] = [];
      for (const m of matchedModules) {
        initialOps.push(...m.getOperations());
      }
      const { matched, failedAt } = filterOpsBySelector(state, q.target.selector, initialOps);
      if (matched.length === 0) {
        const t = failedAt ?? q.target.selector?.[0] ?? q.moduleSelector?.[0];
        if (t) {
          diagnostics.push({
            from: t.from,
            to: t.to,
            source: t.source,
            message: `No operations matched '${t.source}'`,
            severity: "warning",
          });
        }
      }
      continue;
    }
  }

  return diagnostics;
}

function findWithParent(node: SyntaxNode, type: string): SyntaxNode | null {
  while (node.parent && node.parent.name !== type) {
    if (!node.parent) return null;
    node = node.parent;
  }
  if (node.parent?.name !== type) {
    return null;
  }
  return node;
}

export function completeCQL(context: CompletionContext, modelIndex: IndexState) {
  const nodeBefore = syntaxTree(context.state).resolveInner(context.pos, -1);

  let nExtraSpaces = 0;

  let moduleSelector: ModuleSelectorToken[] | null = null;
  let opSelector: OpSelectorToken[] | null = null;
  let autocompleteType:
    | "descendant"
    | "child"
    | "next-sibling"
    | "subsequent-sibling"
    | "op"
    | "op-next-sibling"
    | "op-subsequent-sibling"
    | "parameter"
    | "facet"
    | "module-facet"
    | null = null;

  if (nodeBefore.name === "Script" || nodeBefore.name === "SelectorLine") {
    const nodeBefore2 =
      context.pos > 1 ? syntaxTree(context.state).resolveInner(context.pos - 1, -1) : nodeBefore;
    const targetSelector = nodeBefore2 ? findWithParent(nodeBefore2, "ModuleSelector") : null;
    autocompleteType = "descendant";
    moduleSelector = [];
    if (targetSelector) {
      flattenModules(targetSelector.cursor(), context.state.doc.toString(), moduleSelector);
      if (moduleSelector.length === 0) return null;
    } else if (!nodeBefore2) {
      return null;
    }
  } else if (nodeBefore.name.match(/^(Child|Sibling)(Selector|Op)$/)) {
    let targetSelectorNode: SyntaxNode | null = nodeBefore;
    const isSelector = nodeBefore.name.endsWith("Selector");
    if (isSelector) {
      targetSelectorNode = nodeBefore;
    } else {
      nExtraSpaces = 1;
      targetSelectorNode = nodeBefore.parent;
    }
    // If immediate parent of nodeBefore is OpSelector, then we use a different path to get the
    // module selector.
    if (targetSelectorNode && targetSelectorNode.name.match(/^(Sibling|Child)Selector$/)) {
      const isOpSelector = targetSelectorNode.parent?.name === "OpSelector";
      if (nodeBefore.name.startsWith("Sibling")) {
        const opNode = isSelector
          ? syntaxTree(context.state).resolveInner(context.pos - 1, -1)
          : nodeBefore;
        const opText = context.state.sliceDoc(opNode.from, opNode.to);
        if (opText === "+") {
          autocompleteType = "next-sibling";
        } else if (opText === "~") {
          autocompleteType = "subsequent-sibling";
        }
      } else {
        autocompleteType = "child";
      }
      let moduleSelectorNode: SyntaxNode | null = targetSelectorNode;
      if (isOpSelector) {
        autocompleteType = `op-${autocompleteType}` as "op-next-sibling" | "op-subsequent-sibling";
        moduleSelectorNode =
          moduleSelectorNode.parent?.parent?.parent?.getChild("ModuleSelector") ?? null;
        // If there is no module selector node, then we assume we're in a base-op-selector
        // situation.
        if (!targetSelectorNode.firstChild) return null;
        opSelector = [];
        flattenOpSelectors(
          targetSelectorNode.firstChild.cursor(),
          context.state.doc.toString(),
          opSelector,
        );
        if (opSelector.length === 0) return null;
        if (moduleSelectorNode && moduleSelectorNode.firstChild) {
          moduleSelector = [];
          flattenModules(
            moduleSelectorNode?.firstChild.cursor(),
            context.state.doc.toString(),
            moduleSelector,
          );
          if (moduleSelector.length === 0) return null;
        } else {
          moduleSelector = [{ type: "wildcard" } as ModuleSelectorToken];
        }
      } else {
        // Module-level child/sibling autocomplete: build moduleSelector from surrounding
        // ModuleSelector
        const moduleRoot = moduleSelectorNode?.parent;
        if (moduleRoot && moduleRoot.name === "ModuleSelector" && moduleRoot.firstChild) {
          moduleSelector = [];
          flattenModules(
            moduleRoot.firstChild.cursor(),
            context.state.doc.toString(),
            moduleSelector,
          );
          if (moduleSelector.length === 0) return null;
        } else {
          moduleSelector = [{ type: "wildcard" } as ModuleSelectorToken];
        }
      }
    }
  } else if (nodeBefore.name === "OpSpec" || nodeBefore.name === "@") {
    const targetSelector =
      nodeBefore.name === "OpSpec"
        ? nodeBefore.parent?.firstChild
        : nodeBefore.parent?.parent?.parent?.firstChild;
    if (nodeBefore.name === "@") {
      nExtraSpaces = 1;
    }
    // A little risky to do this unconditionally, but we do want to support queries like a bar
    // `@ Add` etc.
    autocompleteType = "op";
    moduleSelector = [];
    if (targetSelector && targetSelector.name === "ModuleSelector" && targetSelector.firstChild) {
      flattenModules(
        targetSelector.firstChild.cursor(),
        context.state.doc.toString(),
        moduleSelector,
      );
      if (moduleSelector.length === 0) return null;
    }
  } else if (nodeBefore.name === "ParamSpec" || nodeBefore.name === "#") {
    const targetSelector =
      nodeBefore.name === "ParamSpec"
        ? nodeBefore.parent?.firstChild
        : nodeBefore.parent?.parent?.parent?.firstChild;
    if (nodeBefore.name === "#") {
      nExtraSpaces = 1;
    }
    if (targetSelector && targetSelector.name === "ModuleSelector" && targetSelector.firstChild) {
      autocompleteType = "parameter";
      moduleSelector = [];
      flattenModules(
        targetSelector.firstChild.cursor(),
        context.state.doc.toString(),
        moduleSelector,
      );
      if (moduleSelector.length === 0) return null;
    }
  } else if (nodeBefore.name === ":" && nodeBefore.parent?.name.endsWith("Facet")) {
    autocompleteType = nodeBefore.parent?.parent?.parent?.getChild("ModuleSelector")
      ? "module-facet"
      : "facet";
  }

  if (!autocompleteType || (!moduleSelector && !autocompleteType.endsWith("facet"))) return null;

  let options: Completion[] = [];
  if (moduleSelector) {
    if (
      autocompleteType === "descendant" ||
      autocompleteType === "child" ||
      autocompleteType === "next-sibling" ||
      autocompleteType === "subsequent-sibling"
    ) {
      options = moduleAutocompleteQuery(modelIndex, autocompleteType, moduleSelector);

      if (autocompleteType === "descendant") {
        if (options.length === 0) {
          const nextSiblingOptions = moduleAutocompleteQuery(
            modelIndex,
            "next-sibling",
            moduleSelector,
          ).map((o) => ({ ...o, label: `+ ${o.label}`, boost: 2 }));
          const subsequentSiblingOptions = moduleAutocompleteQuery(
            modelIndex,
            "subsequent-sibling",
            moduleSelector,
          ).map((o) => ({ ...o, label: `~ ${o.label}`, boost: 2 }));
          const childOptions = moduleAutocompleteQuery(modelIndex, "child", moduleSelector).map(
            (o) => ({ ...o, label: `> ${o.label}`, boost: 3 }),
          );
          const opOptions = opAutocompleteQuery(modelIndex, "op", moduleSelector, opSelector).map(
            (o) => ({ ...o, label: `@ ${o.label}`, boost: 2 }),
          );
          const parameterOptions = parameterAutocompleteQuery(modelIndex, moduleSelector).map(
            (o) => ({ ...o, label: `# ${o.label}`, boost: 2 }),
          );
          options.push(
            ...nextSiblingOptions,
            ...subsequentSiblingOptions,
            ...childOptions,
            ...opOptions,
            ...parameterOptions,
          );
        }
      }
    } else if (
      autocompleteType === "op" ||
      autocompleteType === "op-next-sibling" ||
      autocompleteType === "op-subsequent-sibling"
    ) {
      options = opAutocompleteQuery(
        modelIndex,
        autocompleteType,
        moduleSelector.length > 0 ? moduleSelector : [{ type: "wildcard" } as ModuleSelectorToken],
        opSelector,
      );
    } else if (autocompleteType === "parameter") {
      options = parameterAutocompleteQuery(modelIndex, moduleSelector);
    }
  }

  if (autocompleteType === "facet" || autocompleteType === "module-facet") {
    options = [
      { label: "grad", type: "keyword" },
      snippetCompletion("scale(${100}%)", { label: "scale%", type: "keyword" }),
      snippetCompletion("scale(${1.0})", { label: "scale", type: "keyword" }),
      snippetCompletion('label("${value}")', { label: "label", type: "keyword" }),
    ];
    if (autocompleteType === "module-facet") {
      options.push(snippetCompletion("input", { label: "input", type: "keyword" }));
      options.push(snippetCompletion("output", { label: "output", type: "keyword" }));
    }
  }

  let apply = undefined;

  if (nExtraSpaces) {
    apply = (view: EditorView, completion: Completion, from: number, to: number) => {
      view.dispatch({
        ...insertCompletionText(view.state, " ".repeat(nExtraSpaces) + completion.label, from, to),
        annotations: pickedCompletion.of(completion),
      });
    };
  }

  return {
    from: context.pos,
    options: apply ? options.map((o) => ({ ...o, apply })) : options,
    validFor: /^(\.?\w*%?)?$/,
  };
}
