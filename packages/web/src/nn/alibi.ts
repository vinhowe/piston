import { zeros } from "@/globals";
import { Module } from "@/nn/module";
import { Tensor } from "@/tensor";

/**
 * Implements ALiBi (Attention with Linear Biases) positional encoding.
 * 
 * ALiBi applies position-dependent biases to attention weights, allowing models to extrapolate to
 * longer sequences than they were trained on.
 * 
 * For more details see "Train Short, Test Long: Attention with Linear Biases Enables Input Length
 * Extrapolation" (https://arxiv.org/abs/2108.12409).
 */
export class AlibiEmbedding extends Module {
  private maxBias: number;

  /**
   * Creates an AlibiEmbedding module.
   * @param nHead The number of attention heads.
   * @param maxBias The maximum bias value. Default: 8.0.
   */
  constructor(maxBias: number = 8.0) {
    super();
    this.maxBias = maxBias;
  }

  /**
   * Applies ALiBi positional biases to the attention scores.
   * @param input The attention scores tensor to apply ALiBi to.
   * @returns Tensor with ALiBi biases applied.
   */
  forward(input: Tensor): Tensor {
    // Create zeros tensor with the input shape (up to 3 dimensions for broadcasting)
    const zerosShape = input.shape.slice(0, 3);
    const alibiBase = zeros(zerosShape, { device: input.device });
    
    // Apply ALiBi bias and add to input
    const alibiBias = alibiBase.alibi(this.maxBias).unsqueeze(3);
    return input.add(alibiBias);
  }
}