import { Module } from "@/nn/module";
import { Tensor } from "@/tensor";

/**
 * Implements the rotary positional encoding.
 *
 * The traditional implementation rotates consecutive pairs of elements in the feature dimension
 * while the default implementation rotates pairs with stride half the feature dimensions for
 * efficiency.
 *
 * For more details see `RoFormer: Enhanced Transformer with Rotary Position Embedding
 * <https://arxiv.org/abs/2104.09864>`.
 */
export class RotaryEmbedding extends Module {
  private dim: number;
  private base: number;

  /**
   * Creates a RotaryEmbedding module.
   * @param dim The feature dimensions to be rotated. If the input feature
   *            is larger than dims then the rest is left unchanged.
   * @param base The base used to compute angular frequency for each dimension
   *             in the positional encodings. Default: 10000.
   */
  constructor(dim: number, base: number = 10000) {
    super();
    this.dim = dim;
    this.base = base;
  }

  /**
   * Applies rotary positional embedding to the input tensor.
   * @param input The input tensor to apply RoPE to.
   * @param offset The position offset for the sequence. Defaults to 0.
   * @returns Tensor with rotary positional embeddings applied.
   */
  forward(input: Tensor, offset: number = 0): Tensor {
    return input.rope(this.dim, this.base, offset);
  }
}
