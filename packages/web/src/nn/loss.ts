import { float32, full, int32 } from "@/globals";
import { LogSoftmax } from "@/nn/activation";
import { Module } from "@/nn/module";
import { Tensor } from "@/tensor";

/**
 * Configuration options for CrossEntropyLoss
 * @interface CrossEntropyLossConfig
 */
export interface CrossEntropyLossConfig {
  /**
   * Index to ignore in the target tensor
   * @default -100
   */
  ignoreIndex?: number;

  /**
   * Specifies the reduction to apply to the output
   * @default 'mean'
   */
  reduction?: "none" | "sum" | "mean";

  /**
   * Smoothing factor for label smoothing regularization
   * @default 0.0
   */
  labelSmoothing?: number;
}

export class CrossEntropyLoss extends Module {
  private ignoreIndex: number;
  private reduction: "none" | "sum" | "mean";
  private labelSmoothing: number;
  private logSoftmax: LogSoftmax;

  constructor(config: CrossEntropyLossConfig = {}) {
    super();
    this.ignoreIndex = config.ignoreIndex ?? -100;
    this.reduction = config.reduction ?? "mean";
    this.labelSmoothing = config.labelSmoothing ?? 0.0;
    this.logSoftmax = new LogSoftmax(1);
  }

  forward(input: Tensor, target: Tensor): Tensor {
    // Validate shapes
    const inputShape = input.size();
    const targetShape = target.size();

    if (inputShape.length !== 2) {
      throw new Error(
        `CrossEntropyLoss: input must be rank-2 [batch_size, vocab_size], got ${inputShape}`,
      );
    }

    if (targetShape.length !== 1) {
      throw new Error(`CrossEntropyLoss: target must be [batch_size], got ${targetShape}`);
    }

    const [batchSize, vocabSize] = inputShape;
    const targetBatchSize = targetShape[0];

    if (batchSize !== targetBatchSize) {
      throw new Error(
        `CrossEntropyLoss: batch size mismatch between input (${batchSize}) and target (${targetBatchSize})`,
      );
    }

    // Apply log softmax to the input
    const logProbs = this.logSoftmax.forward(input);

    // Create mask for ignored indices
    const mask = target
      .ne(
        full(targetShape, this.ignoreIndex, {
          device: target.device,
          dtype: int32,
        }),
      )
      .cast(float32);

    // Gather the negative log-prob for the correct classes
    const nllGathered = logProbs.gather(1, target.unsqueeze(1)).mul(-1);

    // Mask out ignored tokens
    const nllMasked = nllGathered.mul(mask.unsqueeze(1));

    // At this point, nllMasked has shape [batchSize, 1]. We squeeze it to get a per-sample loss.
    const nllLossPerSample = nllMasked.squeeze({ dim: 1 }); // shape: [batchSize]

    // Compute final per-sample loss (without global normalization)
    let finalLossPerSample = nllLossPerSample;

    // If label smoothing is applied, compute the uniform loss per sample and combine.
    if (this.labelSmoothing > 0) {
      // Calculate uniform loss: average log probability over all classes for each sample.
      const allProbsMasked = logProbs.mul(mask.unsqueeze(1));
      const sumLogProbs = allProbsMasked.sum({ dim: 1 }); // shape: [batchSize]
      const uniformLossPerSample = sumLogProbs.mul(-1 / vocabSize);

      // Combine with the nll loss using the smoothing factor.
      finalLossPerSample = nllLossPerSample
        .mul(1 - this.labelSmoothing)
        .add(uniformLossPerSample.mul(this.labelSmoothing));
    }

    // Apply reduction according to the configuration.
    if (this.reduction === "none") {
      // Return the per-sample losses directly.
      return finalLossPerSample;
    } else if (this.reduction === "sum") {
      // Sum up all per-sample losses.
      return finalLossPerSample.sum();
    } else {
      // For "mean", average the sum of losses by the count of valid tokens.
      const validCount = mask.sum();
      return finalLossPerSample.sum().div(validCount);
    }
  }
}
