import { arange, cat, tensor } from "@/globals";
import { Module } from "@/nn/module";
import { Buffer } from "@/nn/parameter";
import { Tensor } from "@/tensor";
import { Device } from "@/wasm";

/**
 * Implements sinusoidal positional encodings as described in "Attention Is All You Need".
 */
export class SinusoidalEmbedding extends Module {
  private dim: number;
  private invFreq: Buffer;

  /**
   * Creates a SinusoidalEmbedding module.
   * @param dim The embedding dimension. Must be even.
   * @param device Optional device to create the internal tensors on.
   */
  constructor(dim: number, device?: Device | "cpu") {
    super();
    if (dim % 2 !== 0) {
      throw new Error("Embedding dimension must be even.");
    }
    this.dim = dim;

    const freqs = new Float32Array(dim / 2);
    for (let i = 0; i < dim / 2; i++) {
      freqs[i] = 1.0 / Math.pow(10000, (2 * i) / dim);
    }

    this.invFreq = new Buffer(tensor(freqs, { device }));
  }

  /**
   * Applies sinusoidal positional embedding to the input tensor.
   * @param input The input tensor, expected shape [batch?, seqLen, dim].
   * @param offset The starting position offset for the sequence. Defaults to 0.
   * @returns Tensor with positional embeddings added.
   */
  forward(input: Tensor, offset: number = 0): Tensor {
    // Ensure invFreq is on the same device as input
    if (this.invFreq.device !== input.device) {
      throw new Error("invFreq must be on the same device as input");
    }

    const device = input.device;

    const seqLen = input.shape[input.shape.length - 2]; // Get sequence length (second to last dim)

    // Create position sequence [offset, offset + 1, ..., offset + seqLen - 1]
    const posSeq = arange({ start: offset, end: offset + seqLen, device });

    // Compute outer product: [seqLen, 1] @ [1, dim/2] -> [seqLen, dim/2]
    const sinusoidInp = posSeq.unsqueeze(1).matmul(this.invFreq.unsqueeze(0));

    // Compute sin and cos: [seqLen, dim/2]
    const sin = sinusoidInp.sin();
    const cos = sinusoidInp.cos();

    // Interleave sin and cos: [seqLen, dim]
    const posEmb = cat([sin, cos], { dim: -1 }); // Concatenate along the last dimension

    // Add batch dimension if input has batch: [1, seqLen, dim] or keep as [seqLen, dim]
    const posEmbFinal = input.shape.length === 3 ? posEmb.unsqueeze(0) : posEmb;

    // Add the positional embeddings to the input
    return input.add(posEmbFinal);
  }
}
