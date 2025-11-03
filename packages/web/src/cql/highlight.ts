import { styleTags, tags as t } from "@lezer/highlight";

// Highlighting rules for the Capture Query Language (CQL) grammar.
// This closely mirrors the approach used by the official Lezer CSS support package. The selectors
// on the left-hand side reference token or node names that come out of `cql.grammar`. The values
// on the right-hand side are `@lezer/highlight` tags that drive theming in CodeMirror 6.

export const cqlHighlighting = styleTags({
  // Core language atoms
  Comment: t.lineComment,
  String: t.string,
  Integer: t.number,
  RegExp: t.regexp,

  // Selectors & identifiers
  UniversalSelector: t.definitionOperator,
  "Identifier IndexableIdentifier": t.variableName,
  // Module/type selectors & facets
  TypeSelector: t.className,
  "NameSelector TensorSelector": t.tagName,
  "ModuleSelectInput ModuleSelectOutput GradientFacetIdent NormFacetIdent ScaleFacetIdent LabelFacetIdent":
    t.constant(t.className),

  // Operators & combinators
  "ChildOp SiblingOp descendantOp": t.logicOperator,
  "=": t.compareOperator,

  // Slicing and punctuation
  "[ ]": t.squareBracket,
  "( )": t.paren,
  "{ }": t.brace,
  ":": t.punctuation,
  "#": t.derefOperator,
  "...": t.punctuation,
  ", ;": t.separator,

  // Pipe into JavaScript
  "|": t.operatorKeyword,

  // Tensor spec sigil
  "@": t.definitionOperator,

  // Fallback punctuation for dots and at-symbols
  ".": t.punctuation,

  // Any remaining child/parent combinators already handled above
});
