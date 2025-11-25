/**
 * Profiler for Piston - similar to torch.profiler
 *
 * Provides functionality to:
 * - Define named scopes for profiling (like torch.profiler.record_function)
 * - Collect GPU kernel timings and buffer allocations
 * - Export to Chrome Trace Event format for visualization
 *
 * @example
 * ```typescript
 * import { profiler } from "piston";
 *
 * profiler.start();
 * const result = profiler.recordFunction("forward_pass", () => {
 *   return profiler.recordFunction("attention", () => {
 *     // tensor ops...
 *   });
 * });
 * await gpu.markStep();
 * profiler.stop();
 *
 * const trace = profiler.exportChromeTrace();
 * // Can be loaded into chrome://tracing or Perfetto
 * ```
 */

import {
  profilerClear,
  profilerExportChromeTrace,
  profilerExportEventsJson,
  profilerIsEnabled,
  profilerPopScope,
  profilerPushScope,
  profilerStart,
  profilerStop,
} from "@piston-ml/piston-web-wasm";

/** Category of a profile event */
export type ProfileCategory = "scope" | "kernel" | "allocation" | "deallocation";

/** A single profile event */
export interface ProfileEvent {
  /** Name of the operation or scope */
  name: string;
  /** Category of the event */
  category: ProfileCategory;
  /** Start timestamp in microseconds since profiling started */
  start_us: number;
  /** Duration in microseconds (0 for instant events) */
  duration_us: number;
  /** Additional metadata (shape, dtype, buffer size, workgroups, etc.) */
  metadata?: Record<string, string>;
  /** Scope stack at the time of the event */
  stack?: string[];
}

/** Chrome Trace Event format output */
export interface ChromeTraceOutput {
  traceEvents: ChromeTraceEvent[];
}

/** Chrome Trace Event format event */
export interface ChromeTraceEvent {
  name: string;
  cat: string;
  ph: string;
  ts: number;
  dur?: number;
  pid: number;
  tid: number;
  args?: Record<string, string>;
}

/**
 * Profiler class for collecting and exporting performance data.
 *
 * Use the singleton `profiler` export for convenience.
 */
export class Profiler {
  private static _instance: Profiler | null = null;

  /**
   * Get the singleton profiler instance.
   */
  static get instance(): Profiler {
    if (!Profiler._instance) {
      Profiler._instance = new Profiler();
    }
    return Profiler._instance;
  }

  /**
   * Start profiling - clears previous events and begins collection.
   */
  start(): void {
    profilerStart();
  }

  /**
   * Stop profiling - closes any open scopes and stops collection.
   */
  stop(): void {
    profilerStop();
  }

  /**
   * Check if profiling is currently enabled.
   */
  get isEnabled(): boolean {
    return profilerIsEnabled();
  }

  /**
   * Record a named function/scope.
   *
   * @param name - Name of the scope (e.g., "attention", "forward_pass")
   * @param fn - Function to execute within the scope
   * @returns The return value of the function
   *
   * @example
   * ```typescript
   * const result = profiler.recordFunction("matmul", () => {
   *   return a.matmul(b);
   * });
   * ```
   */
  recordFunction<T>(name: string, fn: () => T): T {
    profilerPushScope(name);
    try {
      return fn();
    } finally {
      profilerPopScope();
    }
  }

  /**
   * Record an async function/scope.
   *
   * @param name - Name of the scope
   * @param fn - Async function to execute within the scope
   * @returns Promise resolving to the function's return value
   *
   * @example
   * ```typescript
   * const result = await profiler.recordFunctionAsync("forward", async () => {
   *   const out = model.forward(input);
   *   await gpu.markStep();
   *   return out;
   * });
   * ```
   */
  async recordFunctionAsync<T>(name: string, fn: () => Promise<T>): Promise<T> {
    profilerPushScope(name);
    try {
      return await fn();
    } finally {
      profilerPopScope();
    }
  }

  /**
   * Create a scope guard that can be used with try/finally.
   *
   * @param name - Name of the scope
   * @returns A disposable scope object
   *
   * @example
   * ```typescript
   * using scope = profiler.scope("attention");
   * // ... operations ...
   * // scope automatically closes when the block exits
   * ```
   */
  scope(name: string): ProfilerScope {
    return new ProfilerScope(name);
  }

  /**
   * Clear all collected events.
   */
  clear(): void {
    profilerClear();
  }

  /**
   * Export events to Chrome Trace Event format JSON.
   *
   * The returned string can be loaded into:
   * - chrome://tracing
   * - Perfetto (https://ui.perfetto.dev)
   *
   * @returns JSON string in Chrome Trace Event format
   */
  exportChromeTrace(): string {
    return profilerExportChromeTrace();
  }

  /**
   * Export events as a structured JSON object.
   *
   * @returns Parsed profile events
   */
  exportEvents(): ProfileEvent[] {
    const json = profilerExportEventsJson();
    return JSON.parse(json) as ProfileEvent[];
  }

  /**
   * Export events as a JSON string.
   *
   * @returns JSON string of profile events
   */
  exportEventsJson(): string {
    return profilerExportEventsJson();
  }

  /**
   * Download the Chrome Trace as a file.
   *
   * @param filename - Name of the file (default: "piston-trace.json")
   */
  downloadChromeTrace(filename = "piston-trace.json"): void {
    const trace = this.exportChromeTrace();
    console.log('trace', trace);
    const blob = new Blob([trace], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }
}

/**
 * Scope guard for profiler - automatically pops the scope when disposed.
 *
 * @example
 * ```typescript
 * using scope = profiler.scope("attention");
 * // ... operations ...
 * // scope automatically closes when the block exits
 * ```
 */
export class ProfilerScope implements Disposable {
  constructor(name: string) {
    profilerPushScope(name);
  }

  [Symbol.dispose](): void {
    profilerPopScope();
  }
}

/**
 * The singleton profiler instance.
 *
 * @example
 * ```typescript
 * import { profiler } from "piston";
 *
 * profiler.start();
 * // ... tensor operations ...
 * profiler.stop();
 *
 * console.log(profiler.exportChromeTrace());
 * ```
 */
export const profiler = Profiler.instance;

