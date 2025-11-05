import { Tensor } from "@/tensor";

import type {
  LintDiagnostic,
  ModelAdapter,
  ModelNode,
  ModuleSelectionResult,
  ModuleSelectorOptions,
  TensorQuery,
} from "./types";

import { formatSelectorToken } from "./formatting";

/**
 * Class that handles module selection based on selector tokens.
 */
export class ModuleSelector<ParameterType = Tensor> {
  private modelAdapter: ModelAdapter<ParameterType>;
  private options: ModuleSelectorOptions;

  constructor(modelAdapter: ModelAdapter<ParameterType>, options: ModuleSelectorOptions = {}) {
    this.modelAdapter = modelAdapter;
    this.options = {
      includeDescendants: false,
      ...options,
    };
  }

  /**
   * Select modules based on selector tokens.
   *
   * @param selectorTokens Array of selector tokens
   * @param initialNodes Starting node(s) for selection
   * @returns Result containing matched modules and context
   */
  public selectModules(
    query: TensorQuery,
    initialNodes: ModelNode<ParameterType> | ModelNode<ParameterType>[],
  ): ModuleSelectionResult<ParameterType> {
    let currentContext: ModelNode<ParameterType>[] = Array.isArray(initialNodes)
      ? initialNodes
      : [initialNodes];

    const diagnostics: LintDiagnostic[] = [];

    const selectorTokens = query.moduleSelector;

    for (let i = 0; i < selectorTokens.length; i++) {
      const token = selectorTokens[i];
      const nextContext: ModelNode<ParameterType>[] = [];

      if (token.type === "combinator") {
        // Combinators prepare for the next path token; actual traversal happens there.
        continue;
      }

      const prevToken = i > 0 ? selectorTokens[i - 1] : null;
      const combinator =
        prevToken && prevToken.type === "combinator" ? prevToken.kind : "descendant"; // Default to descendant

      // Determine if this is the first selector (not preceded by a combinator)
      const isFirstSelector: boolean =
        i === 0 ||
        (i === 1 &&
          prevToken !== null &&
          prevToken.type === "combinator" &&
          prevToken.kind === "descendant");

      for (const node of currentContext) {
        switch (token.type) {
          case "name":
            if (token.index !== undefined && token.index !== null) {
              findByNameIndexed(
                node,
                token.value,
                token.index,
                combinator,
                this.modelAdapter,
                nextContext,
                isFirstSelector,
              );
            } else {
              findByName(
                node,
                token.value,
                combinator,
                this.modelAdapter,
                nextContext,
                isFirstSelector,
              );
            }
            break;
          case "type":
            findByType(node, token.value, combinator, this.modelAdapter, nextContext);
            break;
          case "wildcard":
            findByWildcard(node, combinator, this.modelAdapter, nextContext, isFirstSelector);
            break;
          case "name-regex":
            findByRegex(node, token.value, combinator, this.modelAdapter, nextContext);
            break;
          case "type-regex":
            findByTypeRegex(node, token.value, combinator, this.modelAdapter, nextContext);
            break;
        }
      }

      currentContext = dedupeByPath(nextContext);
      if (currentContext.length === 0 && i < selectorTokens.length - 1) {
        // No matches found midway
        diagnostics.push({
          from: token.from ?? 0,
          to: token.to ?? 0,
          // TODO: Make this a better error message, with the selector up to this point
          message: `No modules found matching "${formatSelectorToken(token)}". The selector failed at this point.`,
          severity: "warning",
          source: "ModuleSelector",
        });
        return {
          matchedModules: [],
          query,
          diagnostics: diagnostics,
          context: { matchCount: 0 },
        };
      }
    }

    // Include descendants if requested
    if (this.options.includeDescendants) {
      const withDescendants: ModelNode<ParameterType>[] = [...currentContext];
      for (const node of currentContext) {
        const descendants: ModelNode<ParameterType>[] = [];
        this.modelAdapter.findDescendants(node, () => true, descendants, this.options.maxDepth);
        withDescendants.push(...descendants);
      }
      currentContext = dedupeByPath(withDescendants);
    }

    // Add warning if no modules matched the complete selector
    if (currentContext.length === 0 && selectorTokens.length > 0) {
      const lastToken = selectorTokens[selectorTokens.length - 1];
      diagnostics.push({
        from: lastToken.from ?? 0,
        to: lastToken.to ?? lastToken.from ?? 0,
        message: `No modules found matching '${lastToken.source}'. The selector failed at this point.`,
        severity: "warning",
        source: lastToken.source,
      });
    }

    return {
      matchedModules: currentContext,
      query,
      diagnostics: diagnostics,
      context: { matchCount: currentContext.length },
    };
  }
}

function dedupeByPath<ParameterType>(
  nodes: ModelNode<ParameterType>[],
): ModelNode<ParameterType>[] {
  const seen = new Set<string>();
  const out: ModelNode<ParameterType>[] = [];
  for (const n of nodes) {
    const key = n.path().join("/");
    if (!seen.has(key)) {
      seen.add(key);
      out.push(n);
    }
  }
  return out;
}

/**
 * Normalize Python-style negative indices based on a node's children length.
 * Returns undefined if the normalized index is out of bounds.
 */
function getChildByIntegerIndex<ParameterType>(
  node: ModelNode<ParameterType>,
  index: number,
): ModelNode<ParameterType> | undefined {
  let normalized = index;
  if (index < 0) {
    const length = node.getChildren().length;
    normalized = length + index;
  }
  if (normalized < 0) return undefined;
  return node.getChild(normalized);
}

/**
 * Find nodes by name.
 */
function findByName<ParameterType>(
  startNode: ModelNode<ParameterType>,
  name: string,
  combinator: string,
  modelAdapter: ModelAdapter<ParameterType>,
  results: ModelNode<ParameterType>[],
  isFirstSelector: boolean = false,
): void {
  if (combinator === "child") {
    const child = modelAdapter.getChild(startNode, name);
    if (child) {
      results.push(child);
    }
  } else if (combinator === "next-sibling" || combinator === "subsequent-sibling") {
    findBySibling<ParameterType>(
      startNode,
      (node) => node.name === name,
      modelAdapter,
      results,
      combinator,
    );
  } else {
    // descendant
    // Special case for the first selector matching the root node
    // This is necessary because in CSS-like selectors, the first token should match
    // the root node itself, not just its descendants
    if (isFirstSelector && startNode.name === name) {
      results.push(startNode);
      return; // Return early as we don't need to search descendants
    }

    const foundNodes: ModelNode<ParameterType>[] = [];
    modelAdapter.findDescendants(startNode, (node) => node.name === name, foundNodes);
    results.push(...foundNodes);
  }
}

/**
 * Find nodes by indexed name.
 */
function findByNameIndexed<ParameterType>(
  startNode: ModelNode<ParameterType>,
  name: string,
  index: number,
  combinator: string,
  modelAdapter: ModelAdapter<ParameterType>,
  results: ModelNode<ParameterType>[],
  isFirstSelector: boolean = false,
): void {
  if (combinator === "child") {
    if (name === "") {
      // Direct index selector [n] - get child by index only
      const child = getChildByIntegerIndex(startNode, index);
      if (child) results.push(child);
    } else {
      const child =
        modelAdapter.getChild(startNode, `${name}[${index}]`) ||
        (() => {
          const parent = modelAdapter.getChild(startNode, name);
          if (!parent) return undefined;
          return getChildByIntegerIndex(parent, index);
        })();
      if (child) results.push(child);
    }
  } else if (combinator === "next-sibling" || combinator === "subsequent-sibling") {
    findBySibling(
      startNode,
      (node) => {
        if (name === "") {
          // Check if sibling has a child at the index
          const indexedChild = getChildByIntegerIndex(node, index);
          return !!indexedChild;
        }
        return (
          node.name === `${name}[${index}]` ||
          (node.name === name && !!getChildByIntegerIndex(node, index))
        );
      },
      modelAdapter,
      results,
      combinator,
    );
  } else {
    // descendant
    // Special case for the first selector matching the root node
    if (isFirstSelector && startNode.name === name) {
      const indexedChild = getChildByIntegerIndex(startNode, index);
      if (indexedChild) {
        results.push(indexedChild);
      }
      return;
    }

    // For descendant, find all nodes with the name, then check for indexed children
    const foundNodes: ModelNode<ParameterType>[] = [];

    if (name === "") {
      // For direct index selector [n], need to check all nodes for children at that index
      modelAdapter.findDescendants(
        startNode,
        (node) => {
          const indexedChild = getChildByIntegerIndex(node, index);
          if (indexedChild) {
            results.push(indexedChild);
          }
          return false; // Don't add the parent to results
        },
        foundNodes,
      );
    } else {
      modelAdapter.findDescendants(
        startNode,
        (node) =>
          node.name === `${name}[${index}]` ||
          (node.name === name && !!getChildByIntegerIndex(node, index)),
        foundNodes,
      );

      // For nodes that match name but not indexed notation, try to get their indexed child
      for (const node of foundNodes) {
        if (node.name === name) {
          const indexedChild = getChildByIntegerIndex(node, index);
          if (indexedChild) results.push(indexedChild);
        } else {
          results.push(node);
        }
      }
    }
  }
}

/**
 * Find sibling nodes.
 */
function findBySibling<ParameterType>(
  startNode: ModelNode<ParameterType>,
  predicate: (node: ModelNode<ParameterType>) => boolean,
  _modelAdapter: ModelAdapter<ParameterType>,
  results: ModelNode<ParameterType>[],
  combinator: string,
): void {
  if (!startNode.parent) {
    return; // No siblings if no parent
  }

  const siblings = startNode.parent.getChildren();
  // Wrapper instances differ across calls; match by path when identity fails
  const startPath = startNode.path().join("/");
  const currentIndex = siblings.findIndex(
    (n) => n === startNode || n.path().join("/") === startPath,
  );

  if (currentIndex < 0) {
    return; // Cannot determine sibling position reliably
  }

  if (combinator === "next-sibling") {
    // Find only the immediate next sibling that matches the predicate
    for (let i = currentIndex + 1; i < siblings.length; i++) {
      const sibling = siblings[i];
      if (predicate(sibling)) {
        results.push(sibling);
        break; // Only the immediate next matching sibling
      }
    }
  } else if (combinator === "subsequent-sibling") {
    // Find all subsequent siblings that match the predicate
    for (let i = currentIndex + 1; i < siblings.length; i++) {
      const sibling = siblings[i];
      if (predicate(sibling)) {
        results.push(sibling);
        // Don't break - continue to find all subsequent matching siblings
      }
    }
  }
}

/**
 * Find all child nodes (wildcard selector).
 */
function findByWildcard<ParameterType>(
  startNode: ModelNode<ParameterType>,
  combinator: string,
  modelAdapter: ModelAdapter<ParameterType>,
  results: ModelNode<ParameterType>[],
  isFirstSelector: boolean = false,
): void {
  if (combinator === "child") {
    results.push(...startNode.getChildren());
  } else if (combinator === "next-sibling" || combinator === "subsequent-sibling") {
    findBySibling(startNode, () => true, modelAdapter, results, combinator);
  } else {
    // For descendant combinator
    // If this is the first selector, include the starting node
    if (isFirstSelector) {
      results.push(startNode);
    }
    // Always include all descendants
    const allDescendants: ModelNode<ParameterType>[] = [];
    modelAdapter.findDescendants(startNode, () => true, allDescendants);
    results.push(...allDescendants);
  }
}

/**
 * Find nodes by type.
 */
function findByType<ParameterType>(
  startNode: ModelNode<ParameterType>,
  typeName: string,
  combinator: string,
  modelAdapter: ModelAdapter<ParameterType>,
  results: ModelNode<ParameterType>[],
): void {
  if (combinator === "child") {
    const children = startNode.getChildren();
    for (const child of children) {
      if (child.typeName === typeName) {
        results.push(child);
      }
    }
  } else if (combinator === "next-sibling" || combinator === "subsequent-sibling") {
    findBySibling<ParameterType>(
      startNode,
      (node) => node.typeName === typeName,
      modelAdapter,
      results,
      combinator,
    );
  } else {
    // Find all descendants with matching type
    const foundNodes: ModelNode<ParameterType>[] = [];
    modelAdapter.findDescendants(startNode, (node) => node.typeName === typeName, foundNodes);
    results.push(...foundNodes);
  }
}

/**
 * Find nodes by type regex.
 */
function findByTypeRegex<ParameterType>(
  startNode: ModelNode<ParameterType>,
  pattern: RegExp,
  combinator: string,
  modelAdapter: ModelAdapter<ParameterType>,
  results: ModelNode<ParameterType>[],
): void {
  if (combinator === "child") {
    const children = startNode.getChildren();
    for (const child of children) {
      if (pattern.test(child.typeName)) {
        results.push(child);
      }
    }
  } else if (combinator === "next-sibling" || combinator === "subsequent-sibling") {
    findBySibling<ParameterType>(
      startNode,
      (node) => pattern.test(node.typeName),
      modelAdapter,
      results,
      combinator,
    );
  } else {
    const foundNodes: ModelNode<ParameterType>[] = [];
    modelAdapter.findDescendants(startNode, (node) => pattern.test(node.typeName), foundNodes);
    results.push(...foundNodes);
  }
}

/**
 * Find nodes by regex pattern.
 */
function findByRegex<ParameterType>(
  startNode: ModelNode<ParameterType>,
  pattern: RegExp,
  combinator: string,
  modelAdapter: ModelAdapter<ParameterType>,
  results: ModelNode<ParameterType>[],
): void {
  if (combinator === "child") {
    const children = startNode.getChildren();
    for (const child of children) {
      if (child.matchesRegex(pattern)) {
        results.push(child);
      }
    }
  } else if (combinator === "next-sibling" || combinator === "subsequent-sibling") {
    findBySibling(
      startNode,
      (node) => node.matchesRegex(pattern),
      modelAdapter,
      results,
      combinator,
    );
  } else {
    // Find all descendants with matching regex
    const foundNodes: ModelNode<ParameterType>[] = [];
    modelAdapter.findDescendants(startNode, (node) => node.matchesRegex(pattern), foundNodes);
    results.push(...foundNodes);
  }
}

/**
 * Utility function to select modules using a selector.
 * This is a convenience wrapper around the ModuleSelector class.
 *
 * @param selectorTokens Array of selector tokens
 * @param initialNodes Starting node(s) for selection
 * @param modelAdapter Model adapter instance
 * @param options Optional configuration options
 * @returns Result containing matched modules
 */
export function selectModules<ParameterType = Tensor>(
  query: TensorQuery,
  initialNodes: ModelNode<ParameterType> | ModelNode<ParameterType>[],
  modelAdapter: ModelAdapter<ParameterType>,
  options: ModuleSelectorOptions = {},
): ModuleSelectionResult<ParameterType> {
  const selector = new ModuleSelector(modelAdapter, options);
  return selector.selectModules(query, initialNodes);
}
