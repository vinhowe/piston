import { initGlobals } from "@/globals";
import { wasmInit } from "@/wasm";

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

  await initGlobals();
}

export {
  arange,
  bernoulli,
  cat,
  cpu,
  float16,
  float32,
  full,
  gpu,
  int32,
  ones,
  onesLike,
  rand,
  randint,
  randn,
  randnMeanStd,
  seed,
  stack,
  tensor,
  uint32,
  zeros,
  zerosLike,
} from "@/globals";
export * as nn from "@/nn";
export * from "@/nn";
export { debugBufferHook } from "@/nn/moduleUtils";
export { Buffer, Parameter } from "@/nn/parameter";
export * as optim from "@/optim";
export * from "@/optim";
export { save } from "@/serialization";
export { init };
export { Tensor } from "@/tensor";
export * from "@/utils";
export { Device, DType } from "@/wasm";
