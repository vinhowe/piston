import { SyntaxNode, TreeCursor } from "@lezer/common";

// import { printTree } from "./__mocks__/print-lezer-tree";
import { DiagnosticError } from "./error";
import { parser } from "./lezer-parser";
import {
  ModuleSelectorToken,
  OpSelectorToken,
  ParameterSelectorToken,
  ParsedGradient,
  ParsedJsPipe,
  ParsedLabel,
  ParsedNorm,
  ParsedQuery,
  ParsedScale,
  ParsedSlice,
  SelectionTarget,
  SliceArgument,
  SourceReferencedItem,
  TensorQuery,
} from "./types";

function sourceReferencedItem(node: SyntaxNode, src: string): SourceReferencedItem {
  return {
    from: node.from,
    to: node.to,
    source: src.slice(node.from, node.to),
  };
}

// Helper to parse an Integer node
function parseInteger(integerNode: SyntaxNode, src: string): number {
  return parseInt(src.slice(integerNode.from, integerNode.to).replace(/_/g, ""), 10);
}

// Helper to parse a SliceItem node
function parseSliceItemNode(itemNode: SyntaxNode, src: string): SliceArgument {
  const children: SyntaxNode[] = [];
  let child = itemNode.firstChild;
  while (child) {
    children.push(child);
    child = child.nextSibling;
  }

  // Case 1: Single Integer or Ellipsis (e.g., [0] or [...])
  if (children.length === 1) {
    if (children[0].name === "Integer") {
      return {
        start: parseInteger(children[0], src),
        stop: null,
        step: null,
        isSingleIndex: true,
      };
    } else if (children[0].name === "Ellipsis") {
      return "ellipsis";
    }
  }

  // Case 2: Colon-based slice (e.g., [:], [1:], [1:2], [1:2:3])
  // Max 3 components: start, stop, step
  const components: (number | null)[] = [null, null, null];
  let componentIndex = 0;

  for (const node of children) {
    if (node.name === "Integer") {
      components[componentIndex] = parseInteger(node, src);
    } else if (node.name === ":") {
      // Anonymous literal for colon
      componentIndex++;
      if (componentIndex > 2) {
        throw new DiagnosticError(
          `Too many colons in SliceItem: ${src.slice(itemNode.from, itemNode.to)}`,
          {
            from: itemNode.from,
            to: itemNode.to,
            source: src,
          },
        );
      }
    }
    // Other node types (like ErrorNode) are ignored, assuming valid AST from Lezer.
  }
  return {
    start: components[0],
    stop: components[1],
    step: components[2],
    isSingleIndex: components[0] !== null && components[1] === null && components[2] === null,
  };
}

function parseIndexable(indexableNode: SyntaxNode, src: string): number | null {
  const child = indexableNode.getChild("Integer");
  if (child) {
    return parseInteger(child, src);
  }
  return null;
}

// Helper to parse a Slice (sliceexpression) node
function parseSliceExpressionNode(exprNode: SyntaxNode, src: string): ParsedSlice {
  const parsedSlice: ParsedSlice = {
    items: [],
    ...sourceReferencedItem(exprNode, src),
  };
  const sliceItems = exprNode.getChildren("SliceItem");
  for (const item of sliceItems) {
    parsedSlice.items.push(parseSliceItemNode(item, src));
  }
  return parsedSlice;
}

function parseRegExpLiteral(literal: string): RegExp {
  // Expect formats like /pattern/flags
  if (!literal.startsWith("/")) {
    throw new DiagnosticError(`Invalid RegExp literal: ${literal}`, {
      from: 0,
      to: 0,
      source: literal,
    });
  }
  const lastSlash = literal.lastIndexOf("/");
  const pattern = literal.slice(1, lastSlash);
  const flags = literal.slice(lastSlash + 1);
  return new RegExp(pattern, flags as unknown as undefined);
}

function parseParameterSelector(paramNameNode: SyntaxNode, src: string): ParameterSelectorToken {
  const idNode = paramNameNode.getChild("Identifier");
  if (idNode) {
    return {
      type: "name",
      value: src.slice(idNode.from, idNode.to),
      ...sourceReferencedItem(idNode, src),
    };
  }
  const wildcardNode = paramNameNode.getChild("UniversalSelector");
  if (wildcardNode) {
    return {
      type: "wildcard",
      ...sourceReferencedItem(wildcardNode, src),
    };
  }
  const regexNode = paramNameNode.getChild("RegExp");
  if (regexNode) {
    const literal = src.slice(regexNode.from, regexNode.to);
    return {
      type: "regex",
      value: parseRegExpLiteral(literal),
      ...sourceReferencedItem(regexNode, src),
    };
  }
  throw new DiagnosticError("Invalid ParamName: expected Identifier, '*' or /regex/", {
    from: paramNameNode.from,
    to: paramNameNode.to,
    source: src,
  });
}

// Helper function to extract name and slice from IndexableIdentifier
function extractNameAndIndex(
  cur: TreeCursor,
  src: string,
  type: "name" | "type",
): ModuleSelectorToken {
  let value = "";
  let index: number | null = null;
  const selectorNode = cur.node;
  const iiNode = selectorNode.getChild("IndexableIdentifier");

  if (iiNode) {
    const idNode = iiNode.getChild("Identifier");
    if (idNode) {
      // The Identifier node itself spans the text of its child 'identifier' (token)
      value = src.slice(idNode.from, idNode.to);
    } else {
      throw new DiagnosticError(
        "Identifier not found within IndexableIdentifier; invalid grammar",
        {
          from: selectorNode.from,
          to: selectorNode.to,
          source: src,
        },
      );
    }
    const indexNode = iiNode.getChild("Index");
    if (indexNode) {
      index = parseIndexable(indexNode, src);
    }
  } else {
    throw new DiagnosticError(
      `IndexableIdentifier not found in ${selectorNode.name}; invalid grammar`,
      {
        from: selectorNode.from,
        to: selectorNode.to,
        source: src,
      },
    );
  }
  return {
    type,
    value,
    index,
    ...sourceReferencedItem(selectorNode, src),
  };
}

const moduleLeafHandlers: Record<string, (cur: TreeCursor, src: string) => ModuleSelectorToken> = {
  NameSelector: (c, s) => extractNameAndIndex(c, s, "name"),
  TypeSelector: (c, s) => extractNameAndIndex(c, s, "type"),
};

const isModuleSelectorLike = (n: string) =>
  n in moduleLeafHandlers ||
  n === "DescendantSelector" ||
  n === "ChildSelector" ||
  n === "SiblingSelector" ||
  n === "UniversalSelector" ||
  n === "RegExpNameSelector" ||
  n === "RegExpTypeSelector";

export function flattenModules(cur: TreeCursor, src: string, out: ModuleSelectorToken[]): void {
  const { name } = cur.node;

  // Universal selector (*)
  if (name === "UniversalSelector") {
    out.push({
      type: "wildcard",
      ...sourceReferencedItem(cur.node, src),
    });
    return;
  }

  // Descendant (whitespace)
  if (name === "DescendantSelector") {
    if (!cur.firstChild()) return;
    let first = true;
    do {
      if (isModuleSelectorLike(cur.node.name)) {
        if (!first)
          out.push({
            type: "combinator",
            kind: "descendant",
            ...sourceReferencedItem(cur.node, src),
          });
        first = false;
      }
      flattenModules(cur, src, out);
    } while (cur.nextSibling());
    cur.parent();
    return;
  }

  // Sibling selector
  if (name === "SiblingSelector") {
    cur.firstChild(); // left side (can be null)
    flattenModules(cur, src, out);

    while (cur.nextSibling()) {
      if (cur.node.name === "SiblingOp") {
        const opText = src.slice(cur.node.from, cur.node.to);
        let kind: "next-sibling" | "subsequent-sibling" = "subsequent-sibling";
        if (opText === "+") {
          kind = "next-sibling";
        } else if (opText === "~") {
          kind = "subsequent-sibling";
        }
        out.push({
          type: "combinator",
          kind,
          ...sourceReferencedItem(cur.node, src),
        });
      } else {
        flattenModules(cur, src, out);
      }
    }
    cur.parent();
    return;
  }

  // Child selector
  if (name === "ChildSelector") {
    cur.firstChild(); // left side
    flattenModules(cur, src, out);

    while (cur.nextSibling()) {
      // op + right side
      if (cur.node.name === "ChildOp") {
        out.push({
          type: "combinator",
          kind: "child",
          ...sourceReferencedItem(cur.node, src),
        });
      } else {
        flattenModules(cur, src, out);
      }
    }
    cur.parent();
    return;
  }

  // Leaf selectors
  const leaf = moduleLeafHandlers[name];
  if (leaf) {
    out.push(leaf(cur, src));
    return;
  }

  // Regex selector for modules
  if (name === "RegExpNameSelector" || name === "RegExpTypeSelector") {
    if (cur.firstChild()) {
      // Children include RegExp token
      do {
        if (cur.node.name === "RegExp") {
          const literal = src.slice(cur.node.from, cur.node.to);
          out.push({
            type: name === "RegExpNameSelector" ? "name-regex" : "type-regex",
            value: parseRegExpLiteral(literal),
            ...sourceReferencedItem(cur.node, src),
          });
          break;
        }
      } while (cur.nextSibling());
      cur.parent();
    }
    return;
  }
}

export function flattenOpSelectors(cur: TreeCursor, src: string, out: OpSelectorToken[]): void {
  const { name } = cur.node;

  // Sibling selector
  if (name === "SiblingSelector") {
    cur.firstChild(); // left side (can be null)
    flattenOpSelectors(cur, src, out);

    while (cur.nextSibling()) {
      if (cur.node.name === "SiblingOp") {
        const opText = src.slice(cur.node.from, cur.node.to);
        let kind: "next-sibling" | "subsequent-sibling" = "subsequent-sibling";
        if (opText === "+") {
          kind = "next-sibling";
        } else if (opText === "~") {
          kind = "subsequent-sibling";
        }
        out.push({
          type: "combinator",
          kind,
          ...sourceReferencedItem(cur.node, src),
        });
      } else {
        flattenOpSelectors(cur, src, out);
      }
    }
    cur.parent();
    return;
  }

  // Universal selector (*)
  if (name === "NameSelector") {
    cur.firstChild();
    if (cur.node.name === "Identifier") {
      // The Identifier node itself spans the text of its child 'identifier' (token)
      out.push({
        type: "name",
        value: src.slice(cur.node.from, cur.node.to),
        ...sourceReferencedItem(cur.node, src),
      });
    } else if (cur.node.name === "UniversalSelector") {
      out.push({
        type: "wildcard",
        ...sourceReferencedItem(cur.node, src),
      });
    } else if (cur.node.name === "RegExp") {
      const literal = src.slice(cur.node.from, cur.node.to);
      out.push({
        type: "regex",
        value: parseRegExpLiteral(literal),
        ...sourceReferencedItem(cur.node, src),
      });
    }
    cur.parent();
    return;
  }
}

// Parse a complete SelectorLine node
function parseSelectorLine(lineNode: SyntaxNode, src: string): ParsedQuery {
  let child: SyntaxNode | null = lineNode.firstChild;

  const sourcedReferencedQuery = sourceReferencedItem(lineNode, src);
  const moduleSelector: ModuleSelectorToken[] = [];
  let target: SelectionTarget | undefined = undefined;
  let gradient: ParsedGradient | undefined = undefined;
  let norm: ParsedNorm | undefined = undefined;
  let scale: ParsedScale | undefined = undefined;
  let label: ParsedLabel | undefined = undefined;
  let slice: ParsedSlice | undefined = undefined;
  let jsPipe: ParsedJsPipe | undefined = undefined;
  let facetsSourceReferencedItem: SourceReferencedItem | undefined = undefined;

  // Parse the selector part
  if (child && child.name === "ModuleSelector") {
    const cursor = child.firstChild!.cursor();
    flattenModules(cursor, src, moduleSelector);
    child = child.nextSibling;
  }

  // Look for one of OpSpec, ParamSpec, ModuleSpec
  if (
    child &&
    (child.name === "OpSpec" || child.name === "ParamSpec" || child.name === "ModuleSpec")
  ) {
    // Common optional nodes
    const readFacetsAndJs = (container: SyntaxNode) => {
      let n: SyntaxNode | null = null;
      const facetsStart = container.from;
      let facetsEnd = container.from;
      if ((n = container.getChild("GradientFacet"))) {
        gradient = {
          value: true,
          ...sourceReferencedItem(n, src),
        };
        facetsEnd = Math.max(n.to, facetsEnd);
      }
      if ((n = container.getChild("NormFacet"))) {
        norm = {
          value: true,
          ...sourceReferencedItem(n, src),
        };
        facetsEnd = Math.max(n.to, facetsEnd);
      }
      // Scale can appear in a facet chain
      const scaleFacet = container.getChild("ScaleFacet");
      if (scaleFacet) {
        const scaleArg = scaleFacet.getChild("ScaleArg");
        if (scaleArg) {
          const percentNode = scaleArg.getChild("Percent");
          const floatNode = scaleArg.getChild("Float");
          const integerNode = scaleArg.getChild("Integer");
          const numericNode = floatNode ?? integerNode ?? percentNode;
          if (numericNode) {
            const text = src.slice(numericNode.from, numericNode.to).replace(/_/g, "");
            const val = parseFloat(text);
            if (!isNaN(val)) {
              scale = {
                value: percentNode ? val / 100 : val,
                unit: percentNode ? "percent" : "absolute",
                ...sourceReferencedItem(scaleFacet, src),
              };
            }
          }
        }
        facetsEnd = Math.max(scaleFacet.to, facetsEnd);
      }

      const labelFacet = container.getChild("LabelFacet");
      if (labelFacet) {
        const strNode = labelFacet.getChild("String");
        if (strNode) {
          const raw = src.slice(strNode.from, strNode.to);
          // Basic unquoting; String token may be missing closing quote, but lezer gives range.
          // Support both single and double quoted strings.
          let unquoted = raw;
          if (
            (unquoted.startsWith('"') && unquoted.endsWith('"')) ||
            (unquoted.startsWith("'") && unquoted.endsWith("'"))
          ) {
            unquoted = unquoted.slice(1, -1);
          } else if (unquoted.startsWith('"') || unquoted.startsWith("'")) {
            unquoted = unquoted.slice(1);
          }
          // Decode simple escapes for readability
          unquoted = unquoted
            .replace(/\\n/g, "\n")
            .replace(/\\t/g, "\t")
            .replace(/\\r/g, "\r")
            .replace(/\\"/g, '"')
            .replace(/\\'/g, "'");
          label = {
            value: unquoted,
            ...sourceReferencedItem(labelFacet, src),
          };
        }
        facetsEnd = Math.max(labelFacet.to, facetsEnd);
      }

      facetsSourceReferencedItem = {
        from: facetsStart,
        to: facetsEnd,
        source: src.slice(facetsStart, facetsEnd),
      };

      // Slice can appear directly as 'Slice', or under 'SliceMod'
      if ((n = container.getChild("Slice"))) {
        slice = parseSliceExpressionNode(n, src);
      } else {
        const sliceMod = container.getChild("SliceMod");
        if (sliceMod && (n = sliceMod.getChild("Slice"))) {
          slice = parseSliceExpressionNode(n, src);
        }
      }
      if ((n = container.getChild("JsPipe"))) {
        const jsChild = n.firstChild;
        if (jsChild && (jsChild.name === "JsBlock" || jsChild.name === "JsStatement")) {
          jsPipe = {
            value: src.slice(jsChild.from, jsChild.to).trim(),
            ...sourceReferencedItem(jsChild, src),
          };
        }
      }
    };

    if (child.name === "OpSpec") {
      const os = child.getChild("OpSelector");
      if (os) {
        const cursor = os.cursor();
        // OpSpec child structure: OpSelector is a node whose children form the selector
        // Move to first child inside OpSelector for flattening
        if (cursor.firstChild()) {
          const selector: OpSelectorToken[] = [];
          flattenOpSelectors(cursor, src, selector);
          target = { kind: "op", selector, ...sourceReferencedItem(os, src) };
        }
      } else {
        throw new DiagnosticError("Empty op selector", {
          from: child.from,
          to: child.to,
          source: src,
        });
      }
      readFacetsAndJs(child);
    } else if (child.name === "ParamSpec") {
      const pn = child.getChild("ParamName");
      if (pn) {
        const selector = parseParameterSelector(pn, src);
        target = { kind: "parameter", selector, ...sourceReferencedItem(pn, src) };
      } else {
        throw new DiagnosticError("Empty parameter selector", {
          from: child.from,
          to: child.to,
          source: src,
        });
      }
      readFacetsAndJs(child);
    } else if (child.name === "ModuleSpec") {
      const moduleFacet = child.getChild("ModuleFacet");
      if (moduleFacet) {
        const selectNode = moduleFacet.getChild("ModuleSelectInput");
        target = {
          kind: "module",
          site: selectNode ? "input" : "output",
          ...sourceReferencedItem(moduleFacet, src),
        };
      }
      readFacetsAndJs(child);
    }

    child = child.nextSibling;
  }

  return {
    moduleSelector,
    target,
    facets: facetsSourceReferencedItem
      ? { gradient, norm, scale, label, ...(facetsSourceReferencedItem as SourceReferencedItem) }
      : undefined,
    slice,
    jsPipe,
    ...sourcedReferencedQuery,
  };
}

function createTensorQuery(parsedQuery: ParsedQuery): TensorQuery {
  return {
    parsedQuery,
    moduleSelector: parsedQuery.moduleSelector,
    target: parsedQuery.target ?? { kind: "module", site: "output" },
    gradient: parsedQuery.facets?.gradient?.value ?? false,
    norm: parsedQuery.facets?.norm?.value ?? false,
    scale: parsedQuery.facets?.scale?.value ?? 1,
    label: parsedQuery.facets?.label?.value ?? undefined,
    jsPipe: parsedQuery.jsPipe?.value ?? null,
    slice: parsedQuery.slice ?? null,
  };
}

export function parse(input: string): TensorQuery[] {
  const result: TensorQuery[] = [];
  const strictParser = parser.configure({ strict: true });
  const tree = strictParser.parse(input);
  const children = tree.topNode.getChildren("Statement");
  if (!children) {
    throw new DiagnosticError("No SelectorLine node found", {
      from: tree.topNode.from,
      to: tree.topNode.to,
      source: input,
    });
  }
  for (const child of children) {
    result.push(createTensorQuery(parseSelectorLine(child.getChild("SelectorLine")!, input)));
  }
  return result;
}
