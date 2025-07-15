import { Tensor } from "@/__mocks__/tensor";
import { MersenneTwister19937, Random } from "random-js";
import { describe, expect, it, vi } from "vitest";

import { DataLoader } from "./dataloader";
import { Dataset, IterableDataset } from "./dataset";
import { BatchSampler, RandomSampler, Sampler, SequentialSampler } from "./sampler";

// Helper function to convert iterator to array for easier testing
function toArray<T>(iterable: Iterable<T>): T[] {
  return Array.from(iterable);
}

// Mock Dataset implementation for testing
class MockDataset implements Dataset<number> {
  constructor(public data: number[]) {}

  get length(): number {
    return this.data.length;
  }

  getItem(index: number): number {
    if (index < 0 || index >= this.data.length) {
      throw new Error(`Index ${index} out of bounds for dataset of length ${this.data.length}`);
    }
    return this.data[index];
  }
}

// Mock IterableDataset implementation for testing
class MockIterableDataset extends IterableDataset<string> {
  constructor(
    private data: string[],
    protected reportedLength?: number,
  ) {
    super();
  }

  get length(): number {
    return this.reportedLength ?? this.data.length;
  }

  *[Symbol.iterator](): Iterator<string> {
    for (const item of this.data) {
      yield item;
    }
  }
}

// Infinite IterableDataset for testing
class InfiniteIterableDataset extends IterableDataset<number> {
  constructor(private maxValue: number = 100) {
    super();
  }

  get length(): number {
    return Number.MAX_SAFE_INTEGER;
  }

  *[Symbol.iterator](): Iterator<number> {
    let i = 0;
    while (true) {
      yield i % this.maxValue;
      i++;
    }
  }
}

// Custom sampler for testing
class CustomSampler implements Sampler<number> {
  constructor(private indices: number[]) {}

  get length(): number {
    return this.indices.length;
  }

  *[Symbol.iterator](): Iterator<number> {
    for (const index of this.indices) {
      yield index;
    }
  }
}

// Custom batch sampler for testing
class CustomBatchSampler implements Sampler<number[]> {
  constructor(private batches: number[][]) {}

  get length(): number {
    return this.batches.length;
  }

  *[Symbol.iterator](): Iterator<number[]> {
    for (const batch of this.batches) {
      yield batch;
    }
  }
}

describe("DataLoader", () => {
  describe("basic configuration", () => {
    it("should work with default configuration", () => {
      const dataset = new MockDataset([1, 2, 3, 4, 5]);
      const dataLoader = new DataLoader(dataset);

      expect(dataLoader.batchSize).toBe(1);
      expect(dataLoader.dropLast).toBe(false);
      expect(dataLoader.dataset).toBe(dataset);
      expect(dataLoader.autoCollation).toBe(true); // BatchSampler is used by default
    });

    it("should handle custom batch size with default collate", () => {
      const dataset = new MockDataset([1, 2, 3, 4, 5]);
      const dataLoader = new DataLoader<number, Tensor>(dataset, { batchSize: 3 });

      expect(dataLoader.batchSize).toBe(3);

      const batches = toArray(dataLoader);
      expect(batches).toHaveLength(2);

      // Default collate converts number arrays to Tensors
      expect(batches[0]).toBeInstanceOf(Tensor);
      expect(batches[1]).toBeInstanceOf(Tensor);
      expect(batches[0]._data).toEqual([1, 2, 3]);
      expect(batches[1]._data).toEqual([4, 5]);
    });

    it("should handle dropLast option", () => {
      const dataset = new MockDataset([1, 2, 3, 4, 5]);
      const dataLoader = new DataLoader<number, Tensor>(dataset, { batchSize: 3, dropLast: true });

      const batches = toArray(dataLoader);
      expect(batches).toHaveLength(1);
      expect(batches[0]).toBeInstanceOf(Tensor);
      expect(batches[0]._data).toEqual([1, 2, 3]);
    });

    it("should handle shuffle option", () => {
      const dataset = new MockDataset([1, 2, 3, 4, 5]);
      const generator = new Random(MersenneTwister19937.seed(42));
      const dataLoader = new DataLoader<number, Tensor>(dataset, { shuffle: true, generator });

      const batches = toArray(dataLoader);
      expect(batches).toHaveLength(5); // Same number of batches

      // With shuffle, the order should be different (but this is probabilistic)
      // Extract the underlying data from tensors
      const extractedData = batches.map((batch) => batch._data[0]);
      extractedData.sort((a, b) => a - b);
      expect(extractedData).toEqual([1, 2, 3, 4, 5]);
    });

    it("should work without auto-collation when batchSize is undefined", () => {
      const dataset = new MockDataset([1, 2, 3, 4, 5]);
      const dataLoader = new DataLoader<number, number>(dataset, { batchSize: undefined });

      expect(dataLoader.batchSize).toBeUndefined();
      expect(dataLoader.autoCollation).toBe(false);

      const items = toArray(dataLoader);
      expect(items).toEqual([1, 2, 3, 4, 5]);
    });
  });

  describe("dataset type handling", () => {
    describe("Map-style datasets", () => {
      it("should work with sequential sampling", () => {
        const dataset = new MockDataset([10, 20, 30, 40]);
        const dataLoader = new DataLoader<number, Tensor>(dataset, { batchSize: 2 });

        const batches = toArray(dataLoader);
        expect(batches).toHaveLength(2);
        expect(batches[0]).toBeInstanceOf(Tensor);
        expect(batches[1]).toBeInstanceOf(Tensor);
        expect(batches[0]._data).toEqual([10, 20]);
        expect(batches[1]._data).toEqual([30, 40]);
      });

      it("should work with random sampling", () => {
        const dataset = new MockDataset([10, 20, 30, 40]);
        const generator = new Random(MersenneTwister19937.seed(123));
        const dataLoader = new DataLoader<number, Tensor>(dataset, {
          batchSize: 2,
          shuffle: true,
          generator,
        });

        const batches = toArray(dataLoader);
        expect(batches).toHaveLength(2);
        expect(batches[0]).toBeInstanceOf(Tensor);
        expect(batches[1]).toBeInstanceOf(Tensor);

        // Extract all values and verify they're the original ones
        const allValues = batches.flatMap((batch) => batch._data).sort();
        expect(allValues).toEqual([10, 20, 30, 40]);
      });

      it("should report correct length", () => {
        const dataset = new MockDataset([1, 2, 3, 4, 5]);

        const dataLoader1 = new DataLoader<number, Tensor>(dataset, { batchSize: 2 });
        expect(dataLoader1.length).toBe(3); // ceil(5/2)

        const dataLoader2 = new DataLoader<number, Tensor>(dataset, {
          batchSize: 2,
          dropLast: true,
        });
        expect(dataLoader2.length).toBe(2); // floor(5/2)
      });
    });

    describe("Iterable datasets", () => {
      it("should work with iterable datasets", () => {
        const dataset = new MockIterableDataset(["a", "b", "c", "d"]);
        const dataLoader = new DataLoader<string, string[]>(dataset, { batchSize: 2 });

        const batches = toArray(dataLoader);
        expect(batches).toEqual([
          ["a", "b"],
          ["c", "d"],
        ]);
      });

      it("should handle incomplete final batch in iterable datasets", () => {
        const dataset = new MockIterableDataset(["a", "b", "c"]);
        const dataLoader = new DataLoader<string, string[]>(dataset, { batchSize: 2 });

        const batches = toArray(dataLoader);
        expect(batches).toEqual([["a", "b"], ["c"]]);
      });

      it("should drop incomplete final batch when dropLast is true", () => {
        const dataset = new MockIterableDataset(["a", "b", "c"]);
        const dataLoader = new DataLoader<string, string[]>(dataset, {
          batchSize: 2,
          dropLast: true,
        });

        const batches = toArray(dataLoader);
        expect(batches).toEqual([["a", "b"]]);
      });

      it("should report correct length for iterable datasets", () => {
        const dataset = new MockIterableDataset(["a", "b", "c", "d", "e"]);

        const dataLoader1 = new DataLoader<string, string[]>(dataset, { batchSize: 2 });
        expect(dataLoader1.length).toBe(3); // ceil(5/2)

        const dataLoader2 = new DataLoader<string, string[]>(dataset, {
          batchSize: 2,
          dropLast: true,
        });
        expect(dataLoader2.length).toBe(2); // floor(5/2)
      });

      it("should warn about length mismatch in iterable datasets", () => {
        const consoleSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

        // Dataset reports length 5 but only has 3 items
        const dataset = new MockIterableDataset(["a", "b"], 1);
        const dataLoader = new DataLoader<string, string>(dataset, { batchSize: 1 });

        // Consume all data
        const iterator = dataLoader[Symbol.iterator]();
        const done = [iterator.next().done, iterator.next().done, iterator.next().done];
        // Call length to trigger the warning
        expect(dataLoader.length).toBe(1);
        // Consume it again
        toArray(dataLoader);

        expect(done).toEqual([false, false, true]);
        expect(consoleSpy)
          .toHaveBeenCalledWith(expect.stringContaining("Length of IterableDataset"))
          .toHaveBeenCalledOnce();

        consoleSpy.mockRestore();
      });

      it("should handle infinite iterable datasets", () => {
        const dataset = new InfiniteIterableDataset(3);
        const dataLoader = new DataLoader<number, Tensor>(dataset, { batchSize: 2 });

        const iterator = dataLoader[Symbol.iterator]();

        // Take first few batches
        const batch1 = iterator.next().value;
        const batch2 = iterator.next().value;
        const batch3 = iterator.next().value;

        expect(batch1).toBeInstanceOf(Tensor);
        expect(batch2).toBeInstanceOf(Tensor);
        expect(batch3).toBeInstanceOf(Tensor);
        expect(batch1._data).toEqual([0, 1]);
        expect(batch2._data).toEqual([2, 0]);
        expect(batch3._data).toEqual([1, 2]);
      });

      it("should work without batching for iterable datasets", () => {
        const dataset = new MockIterableDataset(["a", "b", "c"]);
        const dataLoader = new DataLoader<string, string>(dataset, { batchSize: undefined });

        const items = toArray(dataLoader);
        expect(items).toEqual(["a", "b", "c"]);
      });
    });
  });

  describe("custom samplers", () => {
    it("should work with custom sampler", () => {
      const dataset = new MockDataset([10, 20, 30, 40, 50]);
      const sampler = new CustomSampler([4, 2, 0, 3, 1]);
      const dataLoader = new DataLoader<number, Tensor>(dataset, {
        sampler,
        batchSize: 2,
      });

      const batches = toArray(dataLoader);
      expect(batches).toHaveLength(3);
      expect(batches[0]).toBeInstanceOf(Tensor);
      expect(batches[1]).toBeInstanceOf(Tensor);
      expect(batches[2]).toBeInstanceOf(Tensor);
      expect(batches[0]._data).toEqual([50, 30]);
      expect(batches[1]._data).toEqual([10, 40]);
      expect(batches[2]._data).toEqual([20]);
    });

    it("should work with custom batch sampler", () => {
      const dataset = new MockDataset([10, 20, 30, 40, 50]);
      const batchSampler = new CustomBatchSampler([[1, 3], [0, 4], [2]]);
      const dataLoader = new DataLoader<number, Tensor>(dataset, { batchSampler });

      const batches = toArray(dataLoader);
      expect(batches).toHaveLength(3);
      expect(batches[0]).toBeInstanceOf(Tensor);
      expect(batches[1]).toBeInstanceOf(Tensor);
      expect(batches[2]).toBeInstanceOf(Tensor);
      expect(batches[0]._data).toEqual([20, 40]);
      expect(batches[1]._data).toEqual([10, 50]);
      expect(batches[2]._data).toEqual([30]);
    });

    it("should work with RandomSampler", () => {
      const dataset = new MockDataset([1, 2, 3, 4, 5]);
      const generator = new Random(MersenneTwister19937.seed(456));
      const sampler = new RandomSampler(dataset, { generator });
      const dataLoader = new DataLoader<number, Tensor>(dataset, { sampler, batchSize: 2 });

      const batches = toArray(dataLoader);
      expect(batches).toHaveLength(3);

      const allValues = batches.flatMap((batch) => batch._data).sort();
      expect(allValues).toEqual([1, 2, 3, 4, 5]);
    });

    it("should work with SequentialSampler explicitly", () => {
      const dataset = new MockDataset([1, 2, 3, 4]);
      const sampler = new SequentialSampler(dataset);
      const dataLoader = new DataLoader<number, Tensor>(dataset, { sampler, batchSize: 2 });

      const batches = toArray(dataLoader);
      expect(batches).toHaveLength(2);
      expect(batches[0]._data).toEqual([1, 2]);
      expect(batches[1]._data).toEqual([3, 4]);
    });
  });

  describe("collation", () => {
    it("should use default collate function", () => {
      const dataset = new MockDataset([1, 2, 3, 4]);
      const dataLoader = new DataLoader<number, Tensor>(dataset, { batchSize: 2 });

      const batches = toArray(dataLoader);
      expect(batches).toHaveLength(2);
      expect(batches[0]).toBeInstanceOf(Tensor);
      expect(batches[1]).toBeInstanceOf(Tensor);
      expect(batches[0]._data).toEqual([1, 2]);
      expect(batches[1]._data).toEqual([3, 4]);
    });

    it("should work with custom collate function", () => {
      const dataset = new MockDataset([1, 2, 3, 4]);
      const customCollate = (batch: number[]) => batch.reduce((sum, x) => sum + x, 0);
      const dataLoader = new DataLoader<number, number>(dataset, {
        batchSize: 2,
        collateFn: customCollate,
      });

      const batches = toArray(dataLoader);
      expect(batches).toEqual([3, 7]); // [1+2, 3+4]
    });

    it("should handle collation with mixed types", () => {
      const dataset = new (class extends Dataset<string> {
        private data = ["a", "b", "c", "d"];
        get length() {
          return this.data.length;
        }
        getItem(index: number) {
          return this.data[index];
        }
      })();
      const customCollate = (batch: string[]) => batch.join("-");
      const dataLoader = new DataLoader<string, string>(dataset, {
        batchSize: 2,
        collateFn: customCollate,
      });

      const batches = toArray(dataLoader);
      expect(batches).toEqual(["a-b", "c-d"]);
    });

    it("should use collate function with individual items when auto-collation is disabled", () => {
      const dataset = new MockDataset([1, 2, 3]);
      const customCollate = vi.fn((item: number) => item * 2);
      const dataLoader = new DataLoader<number, number>(dataset, {
        batchSize: undefined,
        collateFn: customCollate,
      });

      const items = toArray(dataLoader);
      expect(items).toEqual([2, 4, 6]); // Each item doubled by collate function
      expect(dataLoader.autoCollation).toBe(false);
      expect(customCollate).toHaveBeenCalledTimes(3);
      expect(customCollate).toHaveBeenNthCalledWith(1, 1);
      expect(customCollate).toHaveBeenNthCalledWith(2, 2);
      expect(customCollate).toHaveBeenNthCalledWith(3, 3);
    });
  });

  describe("iteration behavior", () => {
    it("should be iterable multiple times", () => {
      const dataset = new MockDataset([1, 2, 3, 4]);
      const dataLoader = new DataLoader<number, Tensor>(dataset, { batchSize: 2 });

      const batches1 = toArray(dataLoader);
      const batches2 = toArray(dataLoader);

      expect(batches1).toHaveLength(2);
      expect(batches2).toHaveLength(2);
      expect(batches1[0]._data).toEqual([1, 2]);
      expect(batches1[1]._data).toEqual([3, 4]);
      expect(batches2[0]._data).toEqual([1, 2]);
      expect(batches2[1]._data).toEqual([3, 4]);
    });

    it("should support partial iteration", () => {
      const dataset = new MockDataset([1, 2, 3, 4, 5, 6]);
      const dataLoader = new DataLoader<number, Tensor>(dataset, { batchSize: 2 });

      const iterator = dataLoader[Symbol.iterator]();

      const first = iterator.next();
      expect(first.value._data).toEqual([1, 2]);
      expect(first.done).toBe(false);

      const second = iterator.next();
      expect(second.value._data).toEqual([3, 4]);
      expect(second.done).toBe(false);

      const third = iterator.next();
      expect(third.value._data).toEqual([5, 6]);
      expect(third.done).toBe(false);

      const fourth = iterator.next();
      expect(fourth.done).toBe(true);
    });

    it("should work with for...of loops", () => {
      const dataset = new MockDataset([1, 2, 3, 4]);
      const dataLoader = new DataLoader<number, Tensor>(dataset, { batchSize: 2 });

      const batches: Tensor[] = [];
      for (const batch of dataLoader) {
        batches.push(batch);
      }

      expect(batches).toHaveLength(2);
      expect(batches[0]._data).toEqual([1, 2]);
      expect(batches[1]._data).toEqual([3, 4]);
    });

    it("should handle early termination gracefully", () => {
      const dataset = new MockDataset([1, 2, 3, 4, 5, 6]);
      const dataLoader = new DataLoader<number, Tensor>(dataset, { batchSize: 2 });

      const batches: Tensor[] = [];
      for (const batch of dataLoader) {
        batches.push(batch);
        if (batches.length === 2) break;
      }

      expect(batches).toHaveLength(2);
      expect(batches[0]._data).toEqual([1, 2]);
      expect(batches[1]._data).toEqual([3, 4]);
    });
  });

  describe("error handling", () => {
    it("should throw error for mutually exclusive sampler and shuffle", () => {
      const dataset = new MockDataset([1, 2, 3]);
      const sampler = new SequentialSampler(dataset);

      expect(() => new DataLoader(dataset, { sampler, shuffle: true })).toThrow(
        "sampler option is mutually exclusive with shuffle",
      );
    });

    it("should throw error for mutually exclusive batchSampler and other options", () => {
      const dataset = new MockDataset([1, 2, 3]);
      const batchSampler = new BatchSampler(new SequentialSampler(dataset), { batchSize: 2 });

      expect(
        () =>
          new DataLoader(dataset, {
            batchSampler,
            batchSize: 3,
          }),
      ).toThrow("batchSampler option is mutually exclusive with");

      expect(
        () =>
          new DataLoader(dataset, {
            batchSampler,
            shuffle: true,
          }),
      ).toThrow("batchSampler option is mutually exclusive with");

      expect(
        () =>
          new DataLoader(dataset, {
            batchSampler,
            dropLast: true,
          }),
      ).toThrow("batchSampler option is mutually exclusive with");

      expect(
        () =>
          new DataLoader(dataset, {
            batchSampler,
            sampler: new SequentialSampler(dataset),
          }),
      ).toThrow("batchSampler option is mutually exclusive with");
    });

    it("should throw error for dropLast with undefined batchSize", () => {
      const dataset = new MockDataset([1, 2, 3]);

      expect(
        () =>
          new DataLoader(dataset, {
            batchSize: undefined,
            dropLast: true,
          }),
      ).toThrow(
        "batchSize=undefined disables auto-batching and is mutually exclusive with dropLast",
      );
    });

    it("should throw error for sampler with IterableDataset", () => {
      const dataset = new MockIterableDataset(["a", "b", "c"]);
      const sampler = new SequentialSampler({ length: 3 });

      expect(() => new DataLoader(dataset, { sampler })).toThrow(
        "DataLoader with IterableDataset: expected unspecified sampler option",
      );
    });

    it("should throw error for batchSampler with IterableDataset", () => {
      const dataset = new MockIterableDataset(["a", "b", "c"]);
      const batchSampler = new CustomBatchSampler([[0, 1], [2]]);

      expect(() => new DataLoader(dataset, { batchSampler })).toThrow(
        "DataLoader with IterableDataset: expected unspecified batch_sampler option",
      );
    });

    it("should warn about shuffle with IterableDataset", () => {
      const consoleSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

      const dataset = new MockIterableDataset(["a", "b", "c"]);
      new DataLoader(dataset, { shuffle: true });

      expect(consoleSpy).toHaveBeenCalledWith(
        "shuffle=True with IterableDataset has no effect. IterableDataset does not support shuffling",
      );

      consoleSpy.mockRestore();
    });
  });

  describe("edge cases", () => {
    it("should handle empty datasets", () => {
      const dataset = new MockDataset([]);
      const dataLoader = new DataLoader<number, Tensor>(dataset, { batchSize: 2 });

      const batches = toArray(dataLoader);
      expect(batches).toEqual([]);
      expect(dataLoader.length).toBe(0);
    });

    it("should handle single element datasets", () => {
      const dataset = new MockDataset([42]);
      const dataLoader = new DataLoader<number, Tensor>(dataset, { batchSize: 3 });

      const batches = toArray(dataLoader);
      expect(batches).toHaveLength(1);
      expect(batches[0]).toBeInstanceOf(Tensor);
      expect(batches[0]._data).toEqual([42]);
      expect(dataLoader.length).toBe(1);
    });

    it("should handle batch size larger than dataset", () => {
      const dataset = new MockDataset([1, 2]);
      const dataLoader = new DataLoader<number, Tensor>(dataset, { batchSize: 5 });

      const batches = toArray(dataLoader);
      expect(batches).toHaveLength(1);
      expect(batches[0]._data).toEqual([1, 2]);
      expect(dataLoader.length).toBe(1);
    });

    it("should handle batch size of 1", () => {
      const dataset = new MockDataset([1, 2, 3]);
      const dataLoader = new DataLoader<number, Tensor>(dataset, { batchSize: 1 });

      const batches = toArray(dataLoader);
      expect(batches).toHaveLength(3);
      expect(batches[0]._data).toEqual([1]);
      expect(batches[1]._data).toEqual([2]);
      expect(batches[2]._data).toEqual([3]);
      expect(dataLoader.length).toBe(3);
    });

    it("should handle empty iterable dataset", () => {
      const dataset = new MockIterableDataset([]);
      const dataLoader = new DataLoader<string, string[]>(dataset, { batchSize: 2 });

      const batches = toArray(dataLoader);
      expect(batches).toEqual([]);
      expect(dataLoader.length).toBe(0);
    });

    it("should handle single element iterable dataset", () => {
      const dataset = new MockIterableDataset(["only"]);
      const dataLoader = new DataLoader<string, string[]>(dataset, { batchSize: 3 });

      const batches = toArray(dataLoader);
      expect(batches).toEqual([["only"]]);
      expect(dataLoader.length).toBe(1);
    });
  });

  describe("internal properties and methods", () => {
    it("should correctly identify iterable datasets", () => {
      const mapDataset = new MockDataset([1, 2, 3]);
      const iterableDataset = new MockIterableDataset(["a", "b", "c"]);

      const dataLoader1 = new DataLoader<number, Tensor>(mapDataset);
      const dataLoader2 = new DataLoader<string, string>(iterableDataset);

      expect(dataLoader1.isIterableDataset).toBe(false);
      expect(dataLoader2.isIterableDataset).toBe(true);
    });

    it("should correctly determine auto-collation mode", () => {
      const dataset = new MockDataset([1, 2, 3]);

      const dataLoader1 = new DataLoader<number, Tensor>(dataset, { batchSize: 2 });
      expect(dataLoader1.autoCollation).toBe(true);

      const dataLoader2 = new DataLoader<number, number>(dataset, { batchSize: undefined });
      expect(dataLoader2.autoCollation).toBe(false);
    });

    it("should correctly identify index sampler", () => {
      const dataset = new MockDataset([1, 2, 3]);

      const dataLoader1 = new DataLoader<number, Tensor>(dataset, { batchSize: 2 });
      expect(dataLoader1.indexSampler).toBe(dataLoader1.batchSampler);

      const dataLoader2 = new DataLoader<number, number>(dataset, { batchSize: undefined });
      expect(dataLoader2.indexSampler).toBe(dataLoader2.sampler);
    });

    it("should track iterable dataset length calls", () => {
      const dataset = new MockIterableDataset(["a", "b", "c"]);
      const dataLoader = new DataLoader<string, string[]>(dataset, { batchSize: 2 });

      expect(dataLoader.iterableDatasetLenCalled).toBeUndefined();

      // Access length property
      const length = dataLoader.length;
      expect(length).toBe(2);
      expect(dataLoader.iterableDatasetLenCalled).toBe(3);
    });
  });

  describe("complex scenarios", () => {
    it("should handle complex object batching", () => {
      interface DataPoint {
        id: number;
        value: string;
        features: number[];
      }

      const dataset = new (class extends Dataset<DataPoint> {
        private data: DataPoint[] = [
          { id: 1, value: "a", features: [1.0, 2.0] },
          { id: 2, value: "b", features: [3.0, 4.0] },
          { id: 3, value: "c", features: [5.0, 6.0] },
          { id: 4, value: "d", features: [7.0, 8.0] },
        ];

        get length() {
          return this.data.length;
        }
        getItem(index: number) {
          return this.data[index];
        }
      })();

      const customCollate = (batch: DataPoint[]) => ({
        ids: batch.map((item) => item.id),
        values: batch.map((item) => item.value),
        features: batch.map((item) => item.features),
      });

      const dataLoader = new DataLoader(dataset, {
        batchSize: 2,
        collateFn: customCollate,
      });

      const batches = toArray(dataLoader);
      expect(batches).toEqual([
        {
          ids: [1, 2],
          values: ["a", "b"],
          features: [
            [1.0, 2.0],
            [3.0, 4.0],
          ],
        },
        {
          ids: [3, 4],
          values: ["c", "d"],
          features: [
            [5.0, 6.0],
            [7.0, 8.0],
          ],
        },
      ]);
    });

    it("should work with large datasets efficiently", () => {
      const largeDataset = new MockDataset(Array.from({ length: 10000 }, (_, i) => i));
      const dataLoader = new DataLoader<number, Tensor>(largeDataset, { batchSize: 100 });

      expect(dataLoader.length).toBe(100);

      // Test first and last batch
      const iterator = dataLoader[Symbol.iterator]();
      const firstBatch = iterator.next().value;
      expect(firstBatch._data).toHaveLength(100);
      expect(firstBatch._data[0]).toBe(0);
      expect(firstBatch._data[99]).toBe(99);

      // Skip to last batch
      let batch: Tensor | undefined;
      let batchCount = 1;
      while (true) {
        const next = iterator.next();
        if (next.done) break;
        batch = next.value;
        batchCount++;
      }

      expect(batchCount).toBe(100);
      expect(batch).toBeDefined();
      expect(batch!._data).toHaveLength(100);
      expect(batch!._data[0]).toBe(9900);
      expect(batch!._data[99]).toBe(9999);
    });

    it("should maintain deterministic behavior with seeded random sampling", () => {
      const dataset = new MockDataset([1, 2, 3, 4, 5, 6, 7, 8]);
      const seed = 12345;

      const createDataLoader = () =>
        new DataLoader<number, Tensor>(dataset, {
          batchSize: 3,
          shuffle: true,
          generator: new Random(MersenneTwister19937.seed(seed)),
        });

      const dataLoader1 = createDataLoader();
      const dataLoader2 = createDataLoader();

      const batches1 = toArray(dataLoader1);
      const batches2 = toArray(dataLoader2);

      // The underlying data should be the same
      const data1 = batches1.map((batch) => batch._data);
      const data2 = batches2.map((batch) => batch._data);

      expect(data1).toEqual(data2);
    });

    it("should handle nested iteration correctly", () => {
      const dataset = new MockDataset([1, 2, 3, 4, 5, 6]);
      const dataLoader = new DataLoader<number, Tensor>(dataset, { batchSize: 2 });

      const results: number[][] = [];

      for (const batch1 of dataLoader) {
        for (const batch2 of dataLoader) {
          results.push([...batch1._data, ...batch2._data]);
          break; // Only take first batch from inner loop
        }
      }

      expect(results).toEqual([
        [1, 2, 1, 2],
        [3, 4, 1, 2],
        [5, 6, 1, 2],
      ]);
    });
  });

  describe("memory and performance", () => {
    it("should not load entire dataset into memory at once", () => {
      let accessCount = 0;
      const dataset = new (class implements Dataset<number> {
        get length() {
          return 1000;
        }

        getItem(index: number): number {
          accessCount++;
          return index * 2;
        }
      })();

      const dataLoader = new DataLoader<number, Tensor>(dataset, { batchSize: 10 });
      const iterator = dataLoader[Symbol.iterator]();

      // Get first batch
      const firstBatch = iterator.next().value;
      expect(firstBatch._data).toHaveLength(10);
      expect(accessCount).toBe(10); // Should only access 10 items

      // Get second batch
      const secondBatch = iterator.next().value;
      expect(secondBatch._data).toHaveLength(10);
      expect(accessCount).toBe(20); // Should access 20 items total
    });

    it("should handle iterable dataset streaming correctly", () => {
      let yieldCount = 0;
      const dataset = new (class extends IterableDataset<number> {
        constructor() {
          super();
        }
        get length() {
          return 100;
        }

        *[Symbol.iterator](): Iterator<number> {
          for (let i = 0; i < 100; i++) {
            yieldCount++;
            yield i;
          }
        }
      })();

      const dataLoader = new DataLoader<number, Tensor>(dataset, { batchSize: 5 });
      const iterator = dataLoader[Symbol.iterator]();

      // Get first batch
      const firstBatch = iterator.next().value;
      expect(firstBatch._data).toHaveLength(5);
      expect(yieldCount).toBe(5); // Should only yield 5 items

      // Get second batch
      const secondBatch = iterator.next().value;
      expect(secondBatch._data).toHaveLength(5);
      expect(yieldCount).toBe(10); // Should yield 10 items total
    });
  });
});
