import {
  JsTraceConfig as WasmTraceConfig,
  traceSetPhase as wasmTraceSetPhase,
  traceStart as wasmTraceStart,
  traceStop as wasmTraceStop,
  traceToChrome as wasmTraceToChrome,
} from "@/wasm";

export type TraceConfig = {
  tensors?: boolean;
  allocations?: boolean;
  phases?: boolean;
};

// Shape mirrors the Rust TraceSession; keep loose for now.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type PistonTraceSession = any;

export function start(config: TraceConfig = {}): void {
  const { tensors = true, allocations = true, phases = true } = config;
  const wasmConfig = new WasmTraceConfig(tensors, allocations, phases);
  wasmTraceStart(wasmConfig);
}

export function stop(): PistonTraceSession | null {
  const value = wasmTraceStop();
  // wasmTraceStop returns null when no session was active
  return value === null ? null : (value as PistonTraceSession);
}

export function setPhase(phase: string | null): void {
  wasmTraceSetPhase(phase);
}

export function toChrome(trace: PistonTraceSession): unknown {
  return wasmTraceToChrome(trace);
}
