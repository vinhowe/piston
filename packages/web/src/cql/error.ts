import { LintDiagnostic } from "./types";

export interface DiagnosticErrorOptions {
  from: number;
  to: number;
  source?: string;
  cause?: Error;
}

export class DiagnosticError extends Error {
  public readonly from: number;
  public readonly to: number;
  public readonly source?: string;
  public readonly cause?: Error;

  constructor(message: string, options: DiagnosticErrorOptions) {
    super(message);
    this.name = "DiagnosticError";
    this.from = options.from;
    this.to = options.to;
    this.source = options.source;
    this.cause = options.cause;

    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, DiagnosticError);
    }
  }

  get length(): number {
    return Math.max(0, this.to - this.from);
  }

  get excerpt(): string | undefined {
    if (!this.source) return undefined;
    return this.source.slice(this.from, this.to);
  }

  toString(): string {
    let result = `${this.name}: ${this.message} (at ${this.from}-${this.to})`;

    if (this.source && this.excerpt) {
      result += `\n  Source: "${this.excerpt}"`;
    }

    if (this.cause) {
      result += `\n  Caused by: ${this.cause.message}`;
    }

    return result;
  }

  toLintDiagnostic(): LintDiagnostic {
    return {
      from: this.from,
      to: this.to,
      message: this.message,
      severity: "error",
    };
  }
}
