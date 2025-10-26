import { Random } from "random-js";

import type { Tensor } from "@/tensor";

import { defaultCollate } from "./collate";
import { AsyncIterableDataset, Dataset, IterableDataset } from "./dataset";
import { BatchSampler, RandomSampler, Sampler, SequentialSampler } from "./sampler";

/**
 * Used as sampler for {@link IterableDataset}.
 */
class _InfiniteConstantSampler implements Sampler<number> {
  *[Symbol.iterator](): Iterator<number> {
    while (true) {
      yield 0;
    }
  }
}

class DataLoaderIterator<T, B> implements Iterator<B> {
  private dataset: Dataset<T>;
  private isIterableDataset: boolean;
  private iterableDatasetLenCalled?: number;
  private autoCollation: boolean;
  private dropLast: boolean;
  private indexSampler: Sampler<number | number[]>;
  private collateFn?: (batch: T[]) => B;
  private samplerIter: Iterator<number | number[]>;
  private numYielded: number;
  private datasetIter?: Iterator<T>;
  private ended: boolean;

  constructor(loader: DataLoader<T, B>) {
    this.dataset = loader.dataset;
    this.isIterableDataset = loader.isIterableDataset;
    this.iterableDatasetLenCalled = loader.iterableDatasetLenCalled;
    this.autoCollation = loader.autoCollation;
    this.dropLast = loader.dropLast!;
    this.indexSampler = loader.indexSampler;
    this.collateFn = loader.collateFn;
    this.samplerIter = this.indexSampler[Symbol.iterator]();
    this.numYielded = 0;
    this.ended = false;

    // Initialize dataset iterator for iterable datasets
    if (this.isIterableDataset) {
      this.datasetIter = (this.dataset as IterableDataset<T>)[Symbol.iterator]();
    }
  }

  [Symbol.iterator](): Iterator<B> {
    return this;
  }

  private nextData(): IteratorResult<B> {
    const nextIdx = this.samplerIter.next();
    if (nextIdx.done) {
      return { value: undefined, done: true };
    }

    const idxOrIndices = nextIdx.value as number | number[];

    // Handle fetch logic based on dataset kind
    if (!this.isIterableDataset) {
      // Map dataset fetch logic
      let data: T | T[];
      if (this.autoCollation) {
        data = (idxOrIndices as number[]).map((i) => this.dataset.getItem(i));
      } else {
        data = this.dataset.getItem(idxOrIndices as number);
      }
      const result = this.collateFn ? this.collateFn(data as T[]) : (data as B);
      return { value: result, done: false };
    } else {
      // Iterable dataset fetch logic
      if (this.ended) {
        return { value: undefined, done: true };
      }

      let data: T | T[];
      if (this.autoCollation) {
        const batch: T[] = [];
        for (let i = 0; i < (idxOrIndices as number[]).length; i++) {
          const nextVal = this.datasetIter!.next();
          if (nextVal.done) {
            this.ended = true;
            break;
          }
          batch.push(nextVal.value);
        }
        if (
          batch.length === 0 ||
          (this.dropLast && batch.length < (idxOrIndices as number[]).length)
        ) {
          return { value: undefined, done: true };
        }
        data = batch;
      } else {
        const nextVal = this.datasetIter!.next();
        if (nextVal.done) {
          this.ended = true;
          return { value: undefined, done: true };
        }
        data = nextVal.value;
      }
      const result = this.collateFn ? this.collateFn(data as T[]) : (data as B);
      return { value: result, done: false };
    }
  }

  next(): IteratorResult<B> {
    const result = this.nextData();
    if (result.done) {
      return result;
    }

    this.numYielded += 1;

    if (
      this.isIterableDataset &&
      this.iterableDatasetLenCalled !== undefined &&
      this.numYielded > this.iterableDatasetLenCalled
    ) {
      console.warn(
        `Length of IterableDataset ${this.dataset} was reported to be ${this.iterableDatasetLenCalled} ` +
          `(when accessing len(dataloader)), but ${this.numYielded} samples have been fetched.`,
      );
    }

    return result;
  }
}

class DataLoaderAsyncIterator<T, B> implements AsyncIterator<B> {
  private dataset: Dataset<T>;
  private autoCollation: boolean;
  private dropLast: boolean;
  private indexSampler: Sampler<number | number[]>;
  private collateFn?: (batch: T[]) => B;
  private samplerIter: Iterator<number | number[]>;
  private datasetAsyncIter?: AsyncIterator<T>;
  private ended: boolean;

  constructor(loader: DataLoader<T, B>) {
    this.dataset = loader.dataset;
    this.autoCollation = loader.autoCollation;
    this.dropLast = loader.dropLast!;
    this.indexSampler = loader.indexSampler;
    this.collateFn = loader.collateFn;
    this.samplerIter = this.indexSampler[Symbol.iterator]();
    this.ended = false;

    this.datasetAsyncIter = (this.dataset as AsyncIterableDataset<T>)[Symbol.asyncIterator]();
  }

  private async nextData(): Promise<IteratorResult<B>> {
    const nextIdx = this.samplerIter.next();
    if (nextIdx.done) {
      return { value: undefined, done: true };
    }

    const idxOrIndices = nextIdx.value as number | number[];

    if (!this.datasetAsyncIter || this.ended) {
      // Shouldn't happen: async iterator used with non-async dataset
      return { value: undefined, done: true };
    }

    let data: T | T[];
    if (this.autoCollation) {
      const batch: T[] = [];
      const count = (idxOrIndices as number[]).length;
      for (let i = 0; i < count; i++) {
        const nextVal = await this.datasetAsyncIter.next();
        if (nextVal.done) {
          this.ended = true;
          break;
        }
        batch.push(nextVal.value);
      }
      if (
        batch.length === 0 ||
        (this.dropLast && batch.length < (idxOrIndices as number[]).length)
      ) {
        return { value: undefined, done: true };
      }
      data = batch;
    } else {
      const nextVal = await this.datasetAsyncIter.next();
      if (nextVal.done) {
        this.ended = true;
        return { value: undefined, done: true };
      }
      data = nextVal.value;
    }

    const resultValue = this.collateFn
      ? await Promise.resolve(this.collateFn(data as T[]))
      : (data as B);
    return { value: resultValue, done: false };
  }

  async next(): Promise<IteratorResult<B>> {
    return this.nextData();
  }
}

export interface DataLoaderConfig<T, B> {
  batchSize?: number;
  shuffle?: boolean;
  sampler?: Sampler<number>;
  batchSampler?: Sampler<number[]>;
  collateFn?: (batch: T[]) => B;
  dropLast?: boolean;
  generator?: Random;
}

export class DataLoader<T, B = Tensor> implements Iterable<B>, AsyncIterable<B> {
  public dataset: Dataset<T>;
  public batchSize?: number;
  public dropLast?: boolean;
  public batchSampler?: Sampler<number[]>;
  public sampler?: Sampler<number>;
  public collateFn?: (batch: T[]) => B;
  /** @internal */
  public isIterableDataset: boolean;
  /** @internal */
  public isAsyncIterableDataset: boolean;
  public generator?: Random;
  /** @internal */
  public iterableDatasetLenCalled?: number;

  constructor(dataset: Dataset<T>, config: DataLoaderConfig<T, B> = {}) {
    this.dataset = dataset;

    // MAKE DEFAULTS; BATCH SIZE SHOULD BE 1
    const { generator } = config;
    let {
      dropLast = false,
      shuffle = undefined,
      sampler,
      batchSampler,
      collateFn,
    }: DataLoaderConfig<T, B> = config;
    let batchSize: number | undefined = "batchSize" in config ? config.batchSize : 1;

    // Check if this is an iterable or async-iterable dataset using instanceof
    this.isIterableDataset = dataset instanceof IterableDataset;
    this.isAsyncIterableDataset = dataset instanceof AsyncIterableDataset;

    if (this.isIterableDataset || this.isAsyncIterableDataset) {
      if (config.sampler !== undefined) {
        throw new Error(
          `DataLoader with IterableDataset: expected unspecified sampler option, but got sampler=${config.sampler}`,
        );
      } else if (config.batchSampler !== undefined) {
        throw new Error(
          `DataLoader with IterableDataset: expected unspecified ` +
            `batch_sampler option, but got batch_sampler=${config.batchSampler}`,
        );
      }
    } else {
      shuffle ??= false;
    }

    if (config.sampler !== undefined && config.shuffle) {
      throw new Error("sampler option is mutually exclusive with shuffle");
    }

    if (config.batchSampler !== undefined) {
      // batchSampler option is mutually exclusive with batchSize, shuffle, sampler, and dropLast
      if (batchSize !== 1 || shuffle || sampler !== undefined || dropLast) {
        throw new Error(
          "batchSampler option is mutually exclusive with " +
            "batchSize, shuffle, sampler, and dropLast",
        );
      }
      batchSize = undefined;
      dropLast = false;
    } else if (batchSize === undefined) {
      if (config.dropLast) {
        throw new Error(
          "batchSize=undefined disables auto-batching and is mutually exclusive with dropLast",
        );
      }
    }

    if (this.isIterableDataset || this.isAsyncIterableDataset) {
      if (shuffle) {
        console.warn(
          "shuffle=True with IterableDataset has no effect. " +
            "IterableDataset does not support shuffling",
        );
      }
      // For iterable datasets, use _InfiniteConstantSampler
      sampler = new _InfiniteConstantSampler();
    } else {
      if (config.sampler !== undefined) {
        sampler = config.sampler;
      } else {
        sampler = shuffle
          ? new RandomSampler(dataset as { length: number }, {
              replacement: false,
              numSamples: undefined,
              generator: config.generator,
            })
          : new SequentialSampler(dataset as { length: number });
      }
    }

    if (config.batchSampler === undefined && batchSize !== undefined) {
      batchSampler = new BatchSampler(sampler!, {
        batchSize,
        dropLast,
      });
    } else if (config.batchSampler !== undefined) {
      batchSampler = config.batchSampler;
    }

    this.batchSize = batchSize;
    this.dropLast = dropLast;
    this.sampler = sampler;
    this.batchSampler = batchSampler;
    this.generator = generator;

    if (collateFn === undefined) {
      if (this.autoCollation) {
        collateFn = defaultCollate as (batch: T[]) => B;
      } else {
        collateFn = undefined;
      }
    }
    this.collateFn = collateFn;

    this.iterableDatasetLenCalled = undefined;
  }

  /** @internal */
  public get autoCollation(): boolean {
    return this.batchSampler !== undefined;
  }

  /** @internal */
  public get indexSampler(): Sampler<number | number[]> {
    // The actual sampler used for generating indices for DataLoaderIterator to read data at
    // each time. This would be batchSampler if in auto-collation mode, and sampler otherwise.
    if (this.autoCollation) {
      return this.batchSampler as Sampler<number | number[]>;
    } else {
      return this.sampler as Sampler<number | number[]>;
    }
  }

  public get length(): number {
    if (this.isIterableDataset || this.isAsyncIterableDataset) {
      const length = (this.iterableDatasetLenCalled = this.dataset.length);
      if (length === undefined) {
        throw new Error("Dataset does not implement length");
      }
      if (this.batchSize !== undefined) {
        // IterableDataset doesn't allow custom sampler or batch_sampler
        if (this.dropLast) {
          return Math.floor(length / this.batchSize);
        } else {
          return Math.ceil(length / this.batchSize);
        }
      }
      return length;
    } else {
      return (this.indexSampler as Sampler<unknown>).length!;
    }
  }

  public *[Symbol.iterator](): Iterator<B> {
    if (this.isAsyncIterableDataset) {
      throw new Error(
        "DataLoader with AsyncIterableDataset must be consumed with an async iterator",
      );
    }
    const iterator = new DataLoaderIterator(this);
    let result = iterator.next();
    while (!result.done) {
      yield result.value;
      result = iterator.next();
    }
  }

  public [Symbol.asyncIterator](): AsyncIterator<B> {
    if (this.isAsyncIterableDataset) {
      return new DataLoaderAsyncIterator(this);
    }

    // Fallback: wrap sync iterator into an async iterator so for-await works universally
    const syncIterator = this[Symbol.iterator]();
    return {
      next: async () => {
        const result = syncIterator.next();
        return Promise.resolve(result);
      },
    } as AsyncIterator<B>;
  }
}

export { defaultCollate };