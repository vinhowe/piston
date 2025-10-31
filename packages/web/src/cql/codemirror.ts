import { LanguageSupport, LRLanguage } from "@codemirror/language";
import { Extension } from "@codemirror/state";
import { parseMixed } from "@lezer/common";
import { parser as jsParser } from "@lezer/javascript";

import { parser } from "./lezer-parser";

const mixedCqlParser = parser.configure({
  wrap: parseMixed((node) => {
    if (node.name === "JsBlock" || node.name === "JsStatement") {
      return { parser: jsParser };
    }
    return null;
  }),
});

export const cqlLanguage = LRLanguage.define({
  parser: mixedCqlParser,
  languageData: {
    name: "cql",
    extensions: [".cql"],
    commentTokens: { line: "//" },
  },
});

export function cql(extension?: Extension) {
  return new LanguageSupport(cqlLanguage, extension);
}
