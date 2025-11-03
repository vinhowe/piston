import { parser as jsParser } from "@lezer/javascript";
import { ExternalTokenizer, InputStream } from "@lezer/lr";

import {
  blankLineStart,
  descendantOp,
  eof,
  Identifier,
  JsBlock,
  JsStatement,
  newline as newlineToken,
} from "./lezer-parser.terms";

const spaces = [
  9, 10, 11, 12, 13, 32, 133, 160, 5760, 8192, 8193, 8194, 8195, 8196, 8197, 8198, 8199, 8200, 8201,
  8202, 8232, 8233, 8239, 8287, 12288,
];
const underscore = 95,
  tab = 9,
  carriageReturn = 13,
  parenL = 40,
  braceL = 123,
  space = 32,
  dash = 45,
  period = 46,
  backslash = 92,
  newline = 10,
  slash = 47,
  asterisk = 42;

function isAlpha(ch: number) {
  return (ch >= 65 && ch <= 90) || (ch >= 97 && ch <= 122) || ch >= 161;
}

function isDigit(ch: number) {
  return ch >= 48 && ch <= 57;
}

function isLineBreak(ch: number) {
  return ch == newline || ch == carriageReturn;
}

export const identifiers = new ExternalTokenizer((input: InputStream) => {
  for (let inside = false, dashes = 0, i = 0; ; i++) {
    const { next } = input;
    if (isAlpha(next) || next == dash || next == underscore || (inside && isDigit(next))) {
      if (!inside && (next != dash || i > 0)) inside = true;
      if (dashes === i && next == dash) dashes++;
      input.advance();
    } else if (next == backslash && input.peek(1) != newline) {
      input.advance();
      if (input.next > -1) input.advance();
      inside = true;
    } else {
      if (inside) input.acceptToken(Identifier);
      break;
    }
  }
});

export const descendant = new ExternalTokenizer((input: InputStream) => {
  if (spaces.includes(input.peek(-1))) {
    const { next } = input;
    if (isAlpha(next) || next == underscore || next == period || next == asterisk || next == dash)
      input.acceptToken(descendantOp);
  }
});

export const newlines = new ExternalTokenizer(
  (input, stack) => {
    let prev;
    if (input.next < 0) {
      input.acceptToken(eof);
    } else if (
      ((prev = input.peek(-1)) < 0 || isLineBreak(prev)) &&
      stack.canShift(blankLineStart)
    ) {
      let spaces = 0;
      while (input.next == space || input.next == tab) {
        input.advance();
        spaces++;
      }
      // Treat start-of-line '//' as a blank-line start, but not '#'
      if (
        input.next == newline ||
        input.next == carriageReturn ||
        (input.next == slash && input.peek(1) == slash)
      )
        input.acceptToken(blankLineStart, -spaces);
    } else if (isLineBreak(input.next)) {
      input.acceptToken(newlineToken, 1);
    }
  },
  { contextual: true },
);

export const js = new ExternalTokenizer((input: InputStream) => {
  if (input.next < 0) return;

  while (input.next == space || input.next == tab) {
    input.advance();
  }
  if (input.next == braceL || input.next == parenL) {
    const isBlock = input.next == braceL;
    let text = "";
    let firstNewlinePosition = -1;

    for (let i = 0; i < 2 ** 16; i++) {
      const peek = input.peek(i);
      if (peek === -1) break;
      text += String.fromCharCode(peek);
      if (peek === newline && firstNewlinePosition === -1) {
        firstNewlinePosition = i;
      }
    }

    const tree = jsParser.parse(text);

    const cursor = tree.cursor();
    if (!cursor.firstChild()) {
      return;
    }
    if (isBlock) {
      if (cursor.name !== "Block") {
        return;
      }
    } else {
      if (cursor.name !== "ExpressionStatement") {
        return;
      }
      if (!cursor.firstChild()) {
        return;
      }
      // @ts-expect-error TS isn't happy with cursor changing its name in place
      if (cursor.name !== "ParenthesizedExpression") {
        return;
      }
    }
    const block = cursor.node;
    if (!cursor.lastChild()) {
      return;
    }
    if (cursor.type.isError) {
      return;
    }
    input.acceptToken(isBlock ? JsBlock : JsStatement, block.to - block.from);
  }
});
