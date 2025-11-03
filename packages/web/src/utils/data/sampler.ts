import { MersenneTwister19937, Random } from "random-js";

// Sampler protocol - base interface for all samplers
export interface Sampler<T = number> extends Iterable<T> {
  readonly length?: number;
}

// Base abstract class for samplers (inheritance)
export abstract class BaseSampler<T = number> implements Sampler<T> {
  abstract [Symbol.iterator](): Iterator<T>;
  abstract readonly length?: number;
}

/**
 * Samples elements sequentially, always in the same order.
 *
 * @param dataSource - Dataset to sample from
 */
export class SequentialSampler extends BaseSampler<number> {
  private dataSource: { length: number };

  constructor(dataSource: { length: number }) {
    super();
    this.dataSource = dataSource;
  }

  *[Symbol.iterator](): Iterator<number> {
    for (let i = 0; i < this.dataSource.length; i++) {
      yield i;
    }
  }

  get length(): number {
    return this.dataSource.length;
  }
}

/**
 * Samples elements randomly. If without replacement, then sample from a shuffled dataset.
 * If with replacement, then user can specify numSamples to draw.
 *
 * @param dataSource - Dataset to sample from
 * @param options - Configuration options
 * @param options.replacement - Samples are drawn on-demand with replacement if true, default=false
 * @param options.numSamples - Number of samples to draw, default=length of dataset
 * @param options.generator - Generator used in sampling
 */
export class RandomSampler extends BaseSampler<number> {
  private dataSource: { length: number };
  private replacement: boolean;
  private _numSamples?: number;
  private generator?: Random;

  constructor(
    dataSource: { length: number },
    {
      replacement = false,
      numSamples,
      generator,
    }: {
      replacement?: boolean;
      numSamples?: number;
      generator?: Random;
    } = {},
  ) {
    super();
    this.dataSource = dataSource;
    this.replacement = replacement;
    this._numSamples = numSamples;
    this.generator = generator;

    if (
      this._numSamples !== undefined &&
      (!Number.isInteger(this._numSamples) || this._numSamples <= 0)
    ) {
      throw new Error(
        `numSamples should be a positive integer value, but got numSamples=${this._numSamples}`,
      );
    }

    if (typeof this.replacement !== "boolean") {
      throw new TypeError(
        `replacement should be a boolean value, but got replacement=${this.replacement}`,
      );
    }
  }

  get numSamples(): number {
    return this._numSamples ?? this.dataSource.length;
  }

  *[Symbol.iterator](): Iterator<number> {
    const n = this.dataSource.length;

    let generator: Random;
    if (!this.generator) {
      const seed = Math.floor(Math.random() * 2 ** 32);
      const mt = MersenneTwister19937.seed(seed);
      generator = new Random(mt);
      this.generator = generator;
    } else {
      generator = this.generator;
    }

    const numSamples = this.numSamples;

    if (this.replacement) {
      // Sample with replacement - generate one random index at a time
      for (let i = 0; i < numSamples; i++) {
        yield generator.integer(0, n - 1);
      }
    } else {
      // Sample without replacement. Equivalent, effectively, to repeated torch.randperm.
      const fullPerms = Math.floor(numSamples / n);
      const remainder = numSamples % n;

      // Pre-create the base index array once to avoid allocations in every loop
      const base = Array.from({ length: n }, (_, i) => i);
      const shuffled = [...base];

      const getShuffledSlice = (size: number): number[] => {
        generator.shuffle(shuffled);
        return shuffled.slice(0, size);
      };

      for (let p = 0; p < fullPerms; p++) {
        yield* getShuffledSlice(n);
      }
      if (remainder > 0) {
        yield* getShuffledSlice(remainder);
      }
    }
  }

  get length(): number {
    return this.numSamples;
  }
}

/**
 * Helper function to take a slice of an iterator (similar to itertools.islice)
 */
export function* islice<T>(iterator: Iterator<T>, n: number): Generator<T> {
  for (let count = 0; count < n; count++) {
    const result = iterator.next();
    if (result.done) break;
    yield result.value;
  }
}

/**
 * Helper function to convert an islice result to an array
 */
export function isliceToArray<T>(iterator: Iterator<T>, n: number): T[] {
  const result: T[] = [];
  const slice = islice(iterator, n);
  for (const item of slice) {
    result.push(item);
  }
  return result;
}

/**
 * Wraps another sampler to yield a mini-batch of indices.
 *
 * @param sampler - Base sampler. Can be any iterable object.
 * @param options - Configuration options
 * @param options.batchSize - Size of mini-batch.
 * @param options.dropLast - If true, the sampler will drop the last batch if its size would be less than
 * batchSize.
 *
 * @example
 * ```ts
 * // Returns [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
 * list(BatchSampler(SequentialSampler(range(10)), { batchSize: 3, dropLast: false }))
 *
 * // Returns [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
 * list(BatchSampler(SequentialSampler(range(10)), { batchSize: 3, dropLast: true }))
 * ```
 */
export class BatchSampler<T = number> implements Sampler<T[]> {
  private sampler: Sampler<T>;
  private batchSize: number;
  private dropLast: boolean;

  constructor(
    sampler: Sampler<T>,
    {
      batchSize,
      dropLast = false,
    }: {
      batchSize: number;
      dropLast?: boolean;
    },
  ) {
    if (!Number.isInteger(batchSize) || typeof batchSize === "boolean" || batchSize <= 0) {
      throw new Error(
        `batchSize should be a positive integer value, but got batchSize=${batchSize}`,
      );
    }
    if (typeof dropLast !== "boolean") {
      throw new Error(`dropLast should be a boolean value, but got dropLast=${dropLast}`);
    }
    this.sampler = sampler;
    this.batchSize = batchSize;
    this.dropLast = dropLast;
  }

  *[Symbol.iterator](): Iterator<T[]> {
    // Implemented based on the benchmarking in PyTorch
    const samplerIter = this.sampler[Symbol.iterator]();

    if (this.dropLast) {
      // Create multiple references to the same iterator (zip-like approach)
      while (true) {
        const batch: T[] = [];

        // Try to fill a complete batch
        for (let i = 0; i < this.batchSize; i++) {
          const result = samplerIter.next();
          if (result.done) {
            // If we can't fill a complete batch and dropLast is true, we're done
            return;
          }
          batch.push(result.value);
        }

        yield batch;
      }
    } else {
      // Use islice-like approach for non-drop_last case
      let batch = isliceToArray(samplerIter, this.batchSize);
      while (batch.length > 0) {
        yield batch;
        batch = isliceToArray(samplerIter, this.batchSize);
      }
    }
  }

  get length(): number {
    if (this.sampler.length === undefined) {
      throw new Error("Cannot determine length of BatchSampler when base sampler has no length");
    }

    if (this.dropLast) {
      return Math.floor(this.sampler.length / this.batchSize);
    } else {
      return Math.ceil(this.sampler.length / this.batchSize);
    }
  }
}
