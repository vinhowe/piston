import { Tensor } from "@/tensor";

/**
 * An abstract class representing a Dataset.
 *
 * All datasets that represent a map from keys to data samples should subclass it.
 * All subclasses should overwrite {@link Dataset.getItem}, supporting fetching a data sample for
 * a given key.
 * Subclasses could also optionally overwrite {@link Dataset.length}, which is expected to return
 * the size of the dataset by many {@link Sampler} implementations and the default options of
 * {@link DataLoader}.
 * Subclasses could also optionally implement {@link Dataset.getItems}, for speedup batched samples
 * loading. This method accepts list of indices of samples of batch and returns list of samples.
 */
export abstract class Dataset<T> {
  /**
   * Subclasses must implement getItem to fetch a sample for a given index
   */
  public abstract getItem(index: number): T;

  /**
   * Optional: subclasses can implement length if known
   */
  public get length(): number {
    throw new Error("Dataset subclass should implement length getter");
  }
}

/**
 * An iterable {@link Dataset}.
 *
 * All datasets that represent an iterable of data samples should subclass it.
 * Such form of datasets is particularly useful when data come from a stream.
 *
 * All subclasses should overwrite {@link Symbol.iterator}, which would return an
 * iterator of samples in this dataset.
 */
export abstract class IterableDataset<T> extends Dataset<T> implements Iterable<T> {
  public abstract [Symbol.iterator](): Iterator<T>;

  public getItem(_index: number): T {
    throw new Error("IterableDataset does not support indexed access");
  }
}

/**
 * {@link Dataset} wrapping tensors.
 *
 * Each sample will be retrieved by indexing tensors along the first dimension.
 *
 * @param tensors - Tensors that have the same size of the first dimension
 */
export class TensorDataset extends Dataset<Tensor> {
  private data: Tensor[];

  constructor(data: Tensor[]) {
    super();
    this.data = [...data];
  }

  public getItem(index: number): Tensor {
    if (index < 0 || index >= this.data.length) {
      throw new Error(`Index ${index} out of range [0, ${this.data.length})`);
    }
    return this.data[index];
  }

  public get length(): number {
    return this.data.length;
  }
}
