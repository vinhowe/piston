import { describe, expect, it } from "vitest";

import { parse } from "./parser";
import {
  CombinatorModuleSelectorItem,
  CombinatorOpSelectorItem,
  ModuleSelectorToken,
  OpSelectorToken,
  ParsedSlice,
  TensorQuery,
} from "./types";

// Utility function to remove 'from', 'to', and 'source' fields from objects for cleaner test
// comparisons
function stripPositionFieldsAndQuery<T>(obj: T): T {
  if (obj === null || obj === undefined) {
    return obj;
  }

  // Preserve RegExp instances as-is for deep equality comparisons
  if (obj instanceof RegExp) {
    return obj;
  }

  if (Array.isArray(obj)) {
    return obj.map((item) => stripPositionFieldsAndQuery(item)) as T;
  }

  if (typeof obj === "object") {
    const result: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(obj)) {
      // Skip 'from' and 'to' fields if both exist in the object
      if ((key === "from" || key === "to") && "from" in obj && "to" in obj) {
        continue;
      }
      // Skip 'source' field if it exists in the object
      if (key === "source") {
        continue;
      }
      if (key === "parsedQuery") {
        continue;
      }
      result[key] = stripPositionFieldsAndQuery(value);
    }
    return result as T;
  }

  return obj;
}

type SourcelessModuleSelectorToken = {
  type: ModuleSelectorToken["type"];
  value?: string | RegExp;
  kind?: CombinatorModuleSelectorItem["kind"];
  index?: number | null;
};

type SourcelessOpSelectorToken = {
  type: OpSelectorToken["type"];
  value?: string | RegExp;
  kind?: CombinatorOpSelectorItem["kind"];
};

type SourcelessTarget =
  | { kind: "module"; site: "input" | "output" }
  | { kind: "op"; selector: SourcelessOpSelectorToken[] }
  | {
      kind: "parameter";
      selector: { type: "name" | "wildcard" | "regex"; value?: string | RegExp };
    };

type SourcelessQuery = Omit<
  TensorQuery,
  "moduleSelector" | "slice" | "from" | "to" | "source" | "target" | "parsedQuery"
> & {
  moduleSelector: SourcelessModuleSelectorToken[];
  slice?: Omit<ParsedSlice, "from" | "to" | "source"> | null;
  target: SourcelessTarget;
};

describe("CQL Parser", () => {
  it("parses module selector with regex", () => {
    const input = "/Lin.*/";
    const result = parse(input);

    const expected: SourcelessQuery[] = [
      {
        moduleSelector: [{ type: "name-regex", value: /Lin.*/ }],
        target: { kind: "module", site: "output" },
        slice: null,
        jsPipe: null,
        gradient: false,
        norm: false,
        scale: 1,
      },
    ];

    expect(stripPositionFieldsAndQuery(result)).toEqual(expected);
  });

  it("parses type selector with regex", () => {
    const input = "/^Lin/ ./(Norm|Layer)/";
    const result = parse(input);

    const expected: SourcelessQuery[] = [
      {
        moduleSelector: [
          { type: "name-regex", value: /^Lin/ },
          { type: "combinator", kind: "descendant" },
          { type: "type-regex", value: /(Norm|Layer)/ },
        ],
        target: { kind: "module", site: "output" },
        slice: null,
        jsPipe: null,
        gradient: false,
        norm: false,
        scale: 1,
      },
    ];

    expect(stripPositionFieldsAndQuery(result)).toEqual(expected);
  });

  it("parses op selector with regex", () => {
    const input = "@/(Add|Mul)/";
    const result = parse(input);

    const expected: SourcelessQuery[] = [
      {
        moduleSelector: [],
        target: { kind: "op", selector: [{ type: "regex", value: /(Add|Mul)/ }] },
        slice: null,
        jsPipe: null,
        gradient: false,
        norm: false,
        scale: 1,
      },
    ];

    expect(stripPositionFieldsAndQuery(result)).toEqual(expected);
  });

  it("parses parameter selector with regex", () => {
    const input = ".Linear #/(weight|bias)/";
    const result = parse(input);

    const expected: SourcelessQuery[] = [
      {
        moduleSelector: [{ type: "type", value: "Linear", index: null }],
        target: { kind: "parameter", selector: { type: "regex", value: /(weight|bias)/ } },
        slice: null,
        jsPipe: null,
        gradient: false,
        norm: false,
        scale: 1,
      },
    ];

    expect(stripPositionFieldsAndQuery(result)).toEqual(expected);
  });
  it("parses module selector with child and sibling combinators", () => {
    const input = "model > transformer + .h @ReLU";

    const result = parse(input);

    const expected: SourcelessQuery[] = [
      {
        moduleSelector: [
          { type: "name", value: "model", index: null },
          { type: "combinator", kind: "child" },
          { type: "name", value: "transformer", index: null },
          { type: "combinator", kind: "next-sibling" },
          { type: "type", value: "h", index: null },
        ],
        target: { kind: "op", selector: [{ type: "name", value: "ReLU" }] },
        slice: null,
        jsPipe: null,
        scale: 1,
        gradient: false,
        norm: false,
      },
    ];

    expect(stripPositionFieldsAndQuery(result)).toEqual(expected);
  });

  it("parses selector with :label string", () => {
    const input = '.Linear :label("My Layer")';
    const result = parse(input);

    const expected: SourcelessQuery[] = [
      {
        moduleSelector: [{ type: "type", value: "Linear", index: null }],
        target: { kind: "module", site: "output" },
        slice: null,
        jsPipe: null,
        gradient: false,
        norm: false,
        scale: 1,
        label: "My Layer",
      },
    ];

    expect(stripPositionFieldsAndQuery(result)).toEqual(expected);
  });

  it("parses selector with :label and :scale and :gradient", () => {
    const input = ".Linear :label('LN-1') :scale(150%) :gradient";
    const result = parse(input);

    const expected: SourcelessQuery[] = [
      {
        moduleSelector: [{ type: "type", value: "Linear", index: null }],
        target: { kind: "module", site: "output" },
        slice: null,
        jsPipe: null,
        gradient: true,
        norm: false,
        scale: 1.5,
        label: "LN-1",
      },
    ];

    expect(stripPositionFieldsAndQuery(result)).toEqual(expected);
  });

  it("parses selector with :label empty string", () => {
    const input = '.Linear :label("")';
    const result = parse(input);

    const expected: SourcelessQuery[] = [
      {
        moduleSelector: [{ type: "type", value: "Linear", index: null }],
        target: { kind: "module", site: "output" },
        slice: null,
        jsPipe: null,
        gradient: false,
        norm: false,
        scale: 1,
        label: "",
      },
    ];

    expect(stripPositionFieldsAndQuery(result)).toEqual(expected);
  });

  it("parses selector with :label double-quoted escapes", () => {
    const input = '.Linear :label("Line\\nTab\\tQuote\\" Apos\'")';
    const result = parse(input);

    const expected: SourcelessQuery[] = [
      {
        moduleSelector: [{ type: "type", value: "Linear", index: null }],
        target: { kind: "module", site: "output" },
        slice: null,
        jsPipe: null,
        gradient: false,
        norm: false,
        scale: 1,
        label: "Line\nTab\tQuote\" Apos'",
      },
    ];

    expect(stripPositionFieldsAndQuery(result)).toEqual(expected);
  });

  it("parses selector with :label single-quoted escapes", () => {
    const input = ".Linear :label('It\\'s ok\\nnext line')";
    const result = parse(input);

    const expected: SourcelessQuery[] = [
      {
        moduleSelector: [{ type: "type", value: "Linear", index: null }],
        target: { kind: "module", site: "output" },
        slice: null,
        jsPipe: null,
        gradient: false,
        norm: false,
        scale: 1,
        label: "It's ok\nnext line",
      },
    ];

    expect(stripPositionFieldsAndQuery(result)).toEqual(expected);
  });

  it("parses selector with :label containing parentheses and punctuation", () => {
    const input = '.Linear :label("(alpha) [beta], {gamma}: done")';
    const result = parse(input);

    const expected: SourcelessQuery[] = [
      {
        moduleSelector: [{ type: "type", value: "Linear", index: null }],
        target: { kind: "module", site: "output" },
        slice: null,
        jsPipe: null,
        gradient: false,
        norm: false,
        scale: 1,
        label: "(alpha) [beta], {gamma}: done",
      },
    ];

    expect(stripPositionFieldsAndQuery(result)).toEqual(expected);
  });

  it("flattens selectors with indexes correctly", () => {
    const input = "array[10] .Klass[1]";
    const result = parse(input);

    const expected: SourcelessQuery[] = [
      {
        moduleSelector: [
          { type: "name", value: "array", index: 10 },
          { type: "combinator", kind: "descendant" },
          { type: "type", value: "Klass", index: 1 },
        ],
        target: { kind: "module", site: "output" },
        slice: null,
        jsPipe: null,
        gradient: false,
        norm: false,
        scale: 1,
      },
    ];
    expect(stripPositionFieldsAndQuery(result)).toEqual(expected);
  });

  it("flattens selector with slice", () => {
    const complexInput = "first .second :[1:10:2]";
    const result = parse(complexInput);

    const complexExpected: SourcelessQuery[] = [
      {
        moduleSelector: [
          { type: "name", value: "first", index: null },
          { type: "combinator", kind: "descendant" },
          { type: "type", value: "second", index: null },
        ],
        target: { kind: "module", site: "output" },
        slice: {
          items: [{ start: 1, stop: 10, step: 2, isSingleIndex: false }],
        },
        jsPipe: null,
        gradient: false,
        norm: false,
        scale: 1,
      },
    ];
    expect(stripPositionFieldsAndQuery(result)).toEqual(complexExpected);
  });

  it("flattens selector with ellipsis in slice", () => {
    const input = "first .second :[...,:2]";
    const result = parse(input);

    const expected: SourcelessQuery[] = [
      {
        moduleSelector: [
          { type: "name", value: "first", index: null },
          { type: "combinator", kind: "descendant" },
          { type: "type", value: "second", index: null },
        ],
        target: { kind: "module", site: "output" },
        slice: {
          items: ["ellipsis", { start: null, stop: 2, step: null, isSingleIndex: false }],
        },
        jsPipe: null,
        gradient: false,
        norm: false,
        scale: 1,
      },
    ];
    expect(stripPositionFieldsAndQuery(result)).toEqual(expected);
  });

  it("parses selector with :input module facet", () => {
    const input = "model :input [0:0]";
    const result = parse(input);

    const expected: SourcelessQuery[] = [
      {
        moduleSelector: [{ type: "name", value: "model", index: null }],
        target: { kind: "module", site: "input" },
        slice: {
          items: [{ start: 0, stop: 0, step: null, isSingleIndex: false }],
        },
        jsPipe: null,
        gradient: false,
        norm: false,
        scale: 1,
      },
    ];

    expect(stripPositionFieldsAndQuery(result)).toEqual(expected);
  });

  it("parses selector with :output module facet", () => {
    const input = "transformer :output";
    const result = parse(input);

    const expected: SourcelessQuery[] = [
      {
        moduleSelector: [{ type: "name", value: "transformer", index: null }],
        target: { kind: "module", site: "output" },
        slice: null,
        jsPipe: null,
        gradient: false,
        norm: false,
        scale: 1,
      },
    ];

    expect(stripPositionFieldsAndQuery(result)).toEqual(expected);
  });

  it("parses selector with parameter #weight selected", () => {
    const input = ".Linear #weight [0:0]";
    const result = parse(input);

    const expected: SourcelessQuery[] = [
      {
        moduleSelector: [{ type: "type", value: "Linear", index: null }],
        target: { kind: "parameter", selector: { type: "name", value: "weight" } },
        slice: {
          items: [{ start: 0, stop: 0, step: null, isSingleIndex: false }],
        },
        jsPipe: null,
        gradient: false,
        norm: false,
        scale: 1,
      },
    ];

    expect(stripPositionFieldsAndQuery(result)).toEqual(expected);
  });

  it("parses selector with universal parameter selector", () => {
    const input = ".Linear #*";
    const result = parse(input);

    const expected: SourcelessQuery[] = [
      {
        moduleSelector: [{ type: "type", value: "Linear", index: null }],
        target: { kind: "parameter", selector: { type: "wildcard" } },
        slice: null,
        label: undefined,
        jsPipe: null,
        gradient: false,
        norm: false,
        scale: 1,
      },
    ];

    expect(stripPositionFieldsAndQuery(result)).toEqual(expected);
  });

  it("parses selector with parameter #weight selected with :gradient facet", () => {
    const input = ".Linear #weight :gradient [0:0]";
    const result = parse(input);

    const expected: SourcelessQuery[] = [
      {
        moduleSelector: [{ type: "type", value: "Linear", index: null }],
        target: { kind: "parameter", selector: { type: "name", value: "weight" } },
        slice: {
          items: [{ start: 0, stop: 0, step: null, isSingleIndex: false }],
        },
        jsPipe: null,
        gradient: true,
        norm: false,
        scale: 1,
      },
    ];

    expect(stripPositionFieldsAndQuery(result)).toEqual(expected);
  });

  it("parses selector with :gradient facet", () => {
    const input = ".Linear :gradient [0:0]";
    const result = parse(input);

    const expected: SourcelessQuery[] = [
      {
        moduleSelector: [{ type: "type", value: "Linear", index: null }],
        target: { kind: "module", site: "output" },
        slice: {
          items: [{ start: 0, stop: 0, step: null, isSingleIndex: false }],
        },
        jsPipe: null,
        scale: 1,
        gradient: true,
        norm: false,
      },
    ];

    expect(stripPositionFieldsAndQuery(result)).toEqual(expected);
  });

  it("parses selector with JavaScript pipe", () => {
    const input = "model :output[0:0] | (console.log(it))";
    const result = parse(input);

    const expected: SourcelessQuery[] = [
      {
        moduleSelector: [{ type: "name", value: "model", index: null }],
        target: { kind: "module", site: "output" },
        slice: {
          items: [{ start: 0, stop: 0, step: null, isSingleIndex: false }],
        },
        jsPipe: "(console.log(it))",
        gradient: false,
        norm: false,
        scale: 1,
      },
    ];

    expect(stripPositionFieldsAndQuery(result)).toEqual(expected);
  });

  it("parses selector with all facets", () => {
    const input = "transformer :input :gradient [0:5] | (x => x * 2)";
    const result = parse(input);

    const expected: SourcelessQuery[] = [
      {
        moduleSelector: [
          {
            type: "name",
            value: "transformer",
            index: null,
          },
        ],
        target: { kind: "module", site: "input" },
        slice: {
          items: [{ start: 0, stop: 5, step: null, isSingleIndex: false }],
        },
        gradient: true,
        jsPipe: "(x => x * 2)",
        norm: false,
        scale: 1,
      },
    ];

    expect(stripPositionFieldsAndQuery(result)).toEqual(expected);
  });

  it("parses complex selector with module and parameter facets", () => {
    const input = "model > .Linear + LayerNorm :output [0:0]";
    const result = parse(input);

    const expected: SourcelessQuery[] = [
      {
        moduleSelector: [
          { type: "name", value: "model", index: null },
          { type: "combinator", kind: "child" },
          { type: "type", value: "Linear", index: null },
          { type: "combinator", kind: "next-sibling" },
          { type: "name", value: "LayerNorm", index: null },
        ],
        target: { kind: "module", site: "output" },
        slice: {
          items: [{ start: 0, stop: 0, step: null, isSingleIndex: false }],
        },
        jsPipe: null,
        gradient: false,
        norm: false,
        scale: 1,
      },
    ];

    expect(stripPositionFieldsAndQuery(result)).toEqual(expected);
  });

  it("parses universal selector (*)", () => {
    const input = "model > *";
    const result = parse(input);

    const expected: SourcelessQuery[] = [
      {
        moduleSelector: [
          { type: "name", value: "model", index: null },
          { type: "combinator", kind: "child" },
          { type: "wildcard" },
        ],
        target: { kind: "module", site: "output" },
        slice: null,
        jsPipe: null,
        gradient: false,
        norm: false,
        scale: 1,
      },
    ];

    expect(stripPositionFieldsAndQuery(result)).toEqual(expected);
  });

  it("parses sibling selector with universal selector", () => {
    const input = ".Linear + *";
    const result = parse(input);

    const expected: SourcelessQuery[] = [
      {
        moduleSelector: [
          { type: "type", value: "Linear", index: null },
          { type: "combinator", kind: "next-sibling" },
          { type: "wildcard" },
        ],
        target: { kind: "module", site: "output" },
        slice: null,
        jsPipe: null,
        gradient: false,
        norm: false,
        scale: 1,
      },
    ];

    expect(stripPositionFieldsAndQuery(result)).toEqual(expected);
  });

  it("parses tensor selector with name", () => {
    const input = "model > .Linear + LayerNorm @ReLU";
    const result = parse(input);

    const expected: SourcelessQuery[] = [
      {
        moduleSelector: [
          { type: "name", value: "model", index: null },
          { type: "combinator", kind: "child" },
          { type: "type", value: "Linear", index: null },
          { type: "combinator", kind: "next-sibling" },
          { type: "name", value: "LayerNorm", index: null },
        ],
        target: { kind: "op", selector: [{ type: "name", value: "ReLU" }] },
        slice: null,
        jsPipe: null,
        gradient: false,
        norm: false,
        scale: 1,
      },
    ];

    expect(stripPositionFieldsAndQuery(result)).toEqual(expected);
  });

  it("parses tensor selector with universal selector", () => {
    const input = "model > .Linear + LayerNorm @*";
    const result = parse(input);

    const expected: SourcelessQuery[] = [
      {
        moduleSelector: [
          { type: "name", value: "model", index: null },
          { type: "combinator", kind: "child" },
          { type: "type", value: "Linear", index: null },
          { type: "combinator", kind: "next-sibling" },
          { type: "name", value: "LayerNorm", index: null },
        ],
        target: { kind: "op", selector: [{ type: "wildcard" }] },
        slice: null,
        jsPipe: null,
        gradient: false,
        norm: false,
        scale: 1,
      },
    ];

    expect(stripPositionFieldsAndQuery(result)).toEqual(expected);
  });

  it("parses tensor selector with subsequent-sibling combinator and wildcard", () => {
    const input = "model > .Linear + LayerNorm @ReLU ~ *";
    const result = parse(input);

    const expected: SourcelessQuery[] = [
      {
        moduleSelector: [
          { type: "name", value: "model", index: null },
          { type: "combinator", kind: "child" },
          { type: "type", value: "Linear", index: null },
          { type: "combinator", kind: "next-sibling" },
          { type: "name", value: "LayerNorm", index: null },
        ],
        target: {
          kind: "op",
          selector: [
            { type: "name", value: "ReLU" },
            { type: "combinator", kind: "subsequent-sibling" },
            { type: "wildcard" },
          ],
        },
        slice: null,
        jsPipe: null,
        gradient: false,
        norm: false,
        scale: 1,
      },
    ];

    expect(stripPositionFieldsAndQuery(result)).toEqual(expected);
  });

  it("parses multiple sibling selectors", () => {
    const input = "model + .Linear + LayerNorm @ReLU ~ * ~ *";
    const result = parse(input);

    const expected: SourcelessQuery[] = [
      {
        moduleSelector: [
          { type: "name", value: "model", index: null },
          { type: "combinator", kind: "next-sibling" },
          { type: "type", value: "Linear", index: null },
          { type: "combinator", kind: "next-sibling" },
          { type: "name", value: "LayerNorm", index: null },
        ],
        target: {
          kind: "op",
          selector: [
            { type: "name", value: "ReLU" },
            { type: "combinator", kind: "subsequent-sibling" },
            { type: "wildcard" },
            { type: "combinator", kind: "subsequent-sibling" },
            { type: "wildcard" },
          ],
        },
        slice: null,
        jsPipe: null,
        gradient: false,
        norm: false,
        scale: 1,
      },
    ];

    expect(stripPositionFieldsAndQuery(result)).toEqual(expected);
  });

  it("parses empty module selector with present tensor selector", () => {
    const input = "@ReLU[...,:2]";
    const result = parse(input);

    const expected: SourcelessQuery[] = [
      {
        moduleSelector: [],
        target: { kind: "op", selector: [{ type: "name", value: "ReLU" }] },
        slice: {
          items: ["ellipsis", { start: null, stop: 2, step: null, isSingleIndex: false }],
        },
        jsPipe: null,
        gradient: false,
        norm: false,
        scale: 1,
      },
    ];

    expect(stripPositionFieldsAndQuery(result)).toEqual(expected);
  });

  it("parses selector with :norm facet", () => {
    const input = ".Linear :norm [0:0]";
    const result = parse(input);

    const expected: SourcelessQuery[] = [
      {
        moduleSelector: [{ type: "type", value: "Linear", index: null }],
        target: { kind: "module", site: "output" },
        slice: {
          items: [{ start: 0, stop: 0, step: null, isSingleIndex: false }],
        },
        jsPipe: null,
        gradient: false,
        norm: true,
        scale: 1,
      },
    ];

    expect(stripPositionFieldsAndQuery(result)).toEqual(expected);
  });

  it("parses selector with both :gradient and :norm facets in any order", () => {
    const inputs = [".Linear :gradient :norm [0:0]", ".Linear :norm :gradient [0:0]"];

    for (const input of inputs) {
      const result = parse(input);

      const expected: SourcelessQuery[] = [
        {
          moduleSelector: [{ type: "type", value: "Linear", index: null }],
          target: { kind: "module", site: "output" },
          slice: {
            items: [{ start: 0, stop: 0, step: null, isSingleIndex: false }],
          },
          jsPipe: null,
          gradient: true,
          norm: true,
          scale: 1,
        },
      ];

      expect(stripPositionFieldsAndQuery(result)).toEqual(expected);
    }
  });

  it("parses selector with :scale integer", () => {
    const input = ".Linear :scale(2) [0:0]";
    const result = parse(input);

    const expected: SourcelessQuery[] = [
      {
        moduleSelector: [{ type: "type", value: "Linear", index: null }],
        target: { kind: "module", site: "output" },
        slice: {
          items: [{ start: 0, stop: 0, step: null, isSingleIndex: false }],
        },
        jsPipe: null,
        gradient: false,
        norm: false,
        scale: 2,
      },
    ];

    expect(stripPositionFieldsAndQuery(result)).toEqual(expected);
  });

  it("parses selector with :scale float", () => {
    const input = ".Linear :scale(5.2)";
    const result = parse(input);

    const expected: SourcelessQuery[] = [
      {
        moduleSelector: [{ type: "type", value: "Linear", index: null }],
        target: { kind: "module", site: "output" },
        slice: null,
        jsPipe: null,
        gradient: false,
        norm: false,
        scale: 5.2,
      },
    ];

    expect(stripPositionFieldsAndQuery(result)).toEqual(expected);
  });

  it("parses selector with :scale percent", () => {
    const input = ".Linear :scale(304%)";
    const result = parse(input);

    const expected: SourcelessQuery[] = [
      {
        moduleSelector: [{ type: "type", value: "Linear", index: null }],
        target: { kind: "module", site: "output" },
        slice: null,
        jsPipe: null,
        gradient: false,
        norm: false,
        scale: 3.04,
      },
    ];

    expect(stripPositionFieldsAndQuery(result)).toEqual(expected);
  });
});
