// Import the full wasm module namespace so Rust can reflect on its exports
import * as pistonWasmExports from "@piston-ml/piston-web-wasm";

import { PistonFunctionMode, PistonMarkStepMode } from "@/function";
import { initGlobals } from "@/globals";
import { Tensor } from "@/tensor";
import {
  _setFunctionModeConstructor,
  _setMarkStepModeConstructor,
  _setPistonWebModule,
  _setTensorConstructor,
  wasmInit,
} from "@/wasm";
export * as trace from "@/trace";

// Track initialization state
let isInitialized = false;

async function init(): Promise<void> {
  if (isInitialized) {
    console.warn("WASM module already initialized");
    return;
  }

  try {
    await wasmInit();
    isInitialized = true;
  } catch (e: unknown) {
    const error = e as Error;
    console.error("Failed to initialize piston WASM module:", error);
    console.error("Error details:", {
      message: error.message,
      cause: error.cause ? (error.cause as unknown as { message: string }).message : "unknown",
      stack: error.stack,
    });
    throw error;
  }

  // Setup references to JS objects we need from Rust
  _setPistonWebModule(pistonWasmExports as unknown as object);
  _setFunctionModeConstructor(PistonFunctionMode);
  _setMarkStepModeConstructor(PistonMarkStepMode);
  _setTensorConstructor(Tensor);

  await initGlobals();
}

export * from "@/cql";
export * from "@/function";
export * from "@/globals";
export * as nn from "@/nn";
export * from "@/nn";
export { debugBufferHook } from "@/nn/moduleUtils";
export { Buffer, Parameter } from "@/nn/parameter";
export * as optim from "@/optim";
export * from "@/optim";
export { load, save } from "@/serialization";
export { init };
export { Tensor } from "@/tensor";
export * from "@/utils";
export { Device, DType } from "@/wasm";
