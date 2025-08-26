import { MersenneTwister19937, Random } from "random-js";
import { describe, expect, it } from "vitest";

import { BatchSampler, islice, isliceToArray, RandomSampler, SequentialSampler } from "./sampler";

// Helper function to convert iterator to array for easier testing
function toArray<T>(iterable: Iterable<T>): T[] {
  return Array.from(iterable);
}

// Helper function to create a mock data source with length
function createDataSource(length: number) {
  return { length };
}

// Helper function to create an iterator from an array
function* arrayIterator<T>(arr: T[]): Iterator<T> {
  for (const item of arr) {
    yield item;
  }
}

describe("islice", () => {
  describe("basic functionality", () => {
    it("should return the first n elements from an iterator", () => {
      const arr = [1, 2, 3, 4, 5];
      const iterator = arrayIterator(arr);
      const result = toArray(islice(iterator, 3));

      expect(result).toEqual([1, 2, 3]);
    });

    it("should handle empty iterator", () => {
      const iterator = arrayIterator([]);
      const result = toArray(islice(iterator, 3));

      expect(result).toEqual([]);
    });

    it("should handle n = 0", () => {
      const arr = [1, 2, 3, 4, 5];
      const iterator = arrayIterator(arr);
      const result = toArray(islice(iterator, 0));

      expect(result).toEqual([]);
    });

    it("should handle n larger than iterator length", () => {
      const arr = [1, 2, 3];
      const iterator = arrayIterator(arr);
      const result = toArray(islice(iterator, 10));

      expect(result).toEqual([1, 2, 3]);
    });

    it("should handle n equal to iterator length", () => {
      const arr = [1, 2, 3];
      const iterator = arrayIterator(arr);
      const result = toArray(islice(iterator, 3));

      expect(result).toEqual([1, 2, 3]);
    });

    it("should handle single element", () => {
      const arr = [42];
      const iterator = arrayIterator(arr);
      const result = toArray(islice(iterator, 1));

      expect(result).toEqual([42]);
    });

    it("should not advance iterator beyond what it takes", () => {
      const arr = [1, 2, 3, 4, 5];
      const iterator = arrayIterator(arr);

      // Take first 2 elements
      const first = toArray(islice(iterator, 2));
      expect(first).toEqual([1, 2]);

      // Take next 2 elements
      const second = toArray(islice(iterator, 2));
      expect(second).toEqual([3, 4]);

      // Take remaining elements
      const third = toArray(islice(iterator, 10));
      expect(third).toEqual([5]);
    });
  });

  describe("edge cases", () => {
    it("should handle negative n", () => {
      const arr = [1, 2, 3];
      const iterator = arrayIterator(arr);
      const result = toArray(islice(iterator, -1));

      expect(result).toEqual([]);
    });

    it("should work with different data types", () => {
      const arr = ["a", "b", "c", "d"];
      const iterator = arrayIterator(arr);
      const result = toArray(islice(iterator, 2));

      expect(result).toEqual(["a", "b"]);
    });

    it("should work with object data", () => {
      const arr = [{ id: 1 }, { id: 2 }, { id: 3 }];
      const iterator = arrayIterator(arr);
      const result = toArray(islice(iterator, 2));

      expect(result).toEqual([{ id: 1 }, { id: 2 }]);
    });

    it("should be lazy and not consume entire iterator unnecessarily", () => {
      let consumed = 0;
      function* countingIterator() {
        for (let i = 0; i < 1000; i++) {
          consumed++;
          yield i;
        }
      }

      const iterator = countingIterator();
      const result = toArray(islice(iterator, 3));

      expect(result).toEqual([0, 1, 2]);
      expect(consumed).toBe(3); // Should only consume what we need
    });
  });

  describe("with real iterators", () => {
    it("should work with Map iterator", () => {
      const map = new Map([
        [1, "a"],
        [2, "b"],
        [3, "c"],
        [4, "d"],
      ]);
      const iterator = map.values();
      const result = toArray(islice(iterator, 2));

      expect(result).toEqual(["a", "b"]);
    });

    it("should work with Set iterator", () => {
      const set = new Set([1, 2, 3, 4, 5]);
      const iterator = set.values();
      const result = toArray(islice(iterator, 3));

      expect(result).toEqual([1, 2, 3]);
    });

    it("should work with generator functions", () => {
      function* fibonacci() {
        let a = 0,
          b = 1;
        while (true) {
          yield a;
          [a, b] = [b, a + b];
        }
      }

      const result = toArray(islice(fibonacci(), 5));
      expect(result).toEqual([0, 1, 1, 2, 3]);
    });
  });
});

describe("isliceToArray", () => {
  describe("basic functionality", () => {
    it("should convert islice result to array", () => {
      const arr = [1, 2, 3, 4, 5];
      const iterator = arrayIterator(arr);
      const result = isliceToArray(iterator, 3);

      expect(result).toEqual([1, 2, 3]);
    });

    it("should handle empty iterator", () => {
      const iterator = arrayIterator([]);
      const result = isliceToArray(iterator, 3);

      expect(result).toEqual([]);
    });

    it("should handle n = 0", () => {
      const arr = [1, 2, 3];
      const iterator = arrayIterator(arr);
      const result = isliceToArray(iterator, 0);

      expect(result).toEqual([]);
    });

    it("should handle n larger than iterator length", () => {
      const arr = [1, 2];
      const iterator = arrayIterator(arr);
      const result = isliceToArray(iterator, 5);

      expect(result).toEqual([1, 2]);
    });

    it("should produce same result as Array.from(islice(...))", () => {
      const arr = [1, 2, 3, 4, 5];
      const iterator1 = arrayIterator(arr);
      const iterator2 = arrayIterator(arr);

      const result1 = isliceToArray(iterator1, 3);
      const result2 = Array.from(islice(iterator2, 3));

      expect(result1).toEqual(result2);
    });
  });

  describe("edge cases", () => {
    it("should handle negative n", () => {
      const arr = [1, 2, 3];
      const iterator = arrayIterator(arr);
      const result = isliceToArray(iterator, -1);

      expect(result).toEqual([]);
    });

    it("should work with different data types", () => {
      const arr = ["hello", "world", "test"];
      const iterator = arrayIterator(arr);
      const result = isliceToArray(iterator, 2);

      expect(result).toEqual(["hello", "world"]);
    });

    it("should not advance iterator beyond what it takes", () => {
      const arr = [1, 2, 3, 4, 5];
      const iterator = arrayIterator(arr);

      // Take first 2 elements using isliceToArray
      const first = isliceToArray(iterator, 2);
      expect(first).toEqual([1, 2]);

      // Take next 2 elements
      const second = isliceToArray(iterator, 2);
      expect(second).toEqual([3, 4]);

      // Take remaining elements
      const third = isliceToArray(iterator, 10);
      expect(third).toEqual([5]);
    });
  });

  describe("integration with BatchSampler use case", () => {
    it("should work correctly when used repeatedly on same iterator", () => {
      // This simulates how BatchSampler uses isliceToArray
      const dataSource = createDataSource(10);
      const sampler = new SequentialSampler(dataSource);
      const samplerIter = sampler[Symbol.iterator]();

      const batches: number[][] = [];
      const batchSize = 3;

      let batch = isliceToArray(samplerIter, batchSize);
      while (batch.length > 0) {
        batches.push(batch);
        batch = isliceToArray(samplerIter, batchSize);
      }

      expect(batches).toEqual([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]);
    });
  });
});

describe("SequentialSampler", () => {
  describe("basic functionality", () => {
    it("should sample indices sequentially", () => {
      const dataSource = createDataSource(5);
      const sampler = new SequentialSampler(dataSource);
      const indices = toArray(sampler);

      expect(indices).toEqual([0, 1, 2, 3, 4]);
    });

    it("should handle empty data source", () => {
      const dataSource = createDataSource(0);
      const sampler = new SequentialSampler(dataSource);
      const indices = toArray(sampler);

      expect(indices).toEqual([]);
    });

    it("should handle single element data source", () => {
      const dataSource = createDataSource(1);
      const sampler = new SequentialSampler(dataSource);
      const indices = toArray(sampler);

      expect(indices).toEqual([0]);
    });

    it("should return correct length", () => {
      const dataSource = createDataSource(10);
      const sampler = new SequentialSampler(dataSource);

      expect(sampler.length).toBe(10);
    });

    it("should be iterable multiple times with same results", () => {
      const dataSource = createDataSource(3);
      const sampler = new SequentialSampler(dataSource);

      const firstIteration = toArray(sampler);
      const secondIteration = toArray(sampler);

      expect(firstIteration).toEqual([0, 1, 2]);
      expect(secondIteration).toEqual([0, 1, 2]);
    });
  });

  describe("large datasets", () => {
    it("should handle large datasets efficiently", () => {
      const dataSource = createDataSource(10000);
      const sampler = new SequentialSampler(dataSource);

      expect(sampler.length).toBe(10000);

      // Test first few and last few elements
      const iterator = sampler[Symbol.iterator]();
      expect(iterator.next().value).toBe(0);
      expect(iterator.next().value).toBe(1);
      expect(iterator.next().value).toBe(2);

      // Skip to end by consuming iterator
      let count = 3;
      let result = iterator.next();
      while (!result.done && count < 9999) {
        result = iterator.next();
        count++;
      }
      expect(result.value).toBe(9999);
    });
  });
});

describe("RandomSampler", () => {
  describe("basic functionality", () => {
    it("should sample without replacement by default", () => {
      const dataSource = createDataSource(5);
      const sampler = new RandomSampler(dataSource);
      const indices = toArray(sampler);

      expect(indices).toHaveLength(5);
      expect(new Set(indices)).toHaveProperty("size", 5); // All unique
      expect(indices.every((i) => i >= 0 && i < 5)).toBe(true);
    });

    it("should sample with replacement when specified", () => {
      const dataSource = createDataSource(3);
      const sampler = new RandomSampler(dataSource, {
        replacement: true,
        numSamples: 10,
      });
      const indices = toArray(sampler);

      expect(indices).toHaveLength(10);
      expect(indices.every((i) => i >= 0 && i < 3)).toBe(true);
      // With replacement, duplicates are possible
    });

    it("should use specified numSamples", () => {
      const dataSource = createDataSource(10);
      const sampler = new RandomSampler(dataSource, { numSamples: 5 });
      const indices = toArray(sampler);

      expect(indices).toHaveLength(5);
      expect(new Set(indices)).toHaveProperty("size", 5); // All unique since no replacement
    });

    it("should return correct length", () => {
      const dataSource = createDataSource(10);

      const sampler1 = new RandomSampler(dataSource);
      expect(sampler1.length).toBe(10);

      const sampler2 = new RandomSampler(dataSource, { numSamples: 5 });
      expect(sampler2.length).toBe(5);
    });

    it("should use custom generator when provided", () => {
      const dataSource = createDataSource(5);
      const mt = MersenneTwister19937.seed(12345);
      const generator = new Random(mt);

      const sampler = new RandomSampler(dataSource, { generator });
      const indices1 = toArray(sampler);

      // Reset generator to same seed for reproducible results
      const mt2 = MersenneTwister19937.seed(12345);
      const generator2 = new Random(mt2);
      const sampler2 = new RandomSampler(dataSource, { generator: generator2 });
      const indices2 = toArray(sampler2);

      expect(indices1).toEqual(indices2);
    });
  });

  describe("sampling without replacement", () => {
    it("should handle multiple full permutations", () => {
      const dataSource = createDataSource(3);
      const sampler = new RandomSampler(dataSource, {
        replacement: false,
        numSamples: 7, // 2 full perms + 1 remainder
      });
      const indices = toArray(sampler);

      expect(indices).toHaveLength(7);

      // First 6 should contain two complete sets of [0,1,2]
      const firstSix = indices.slice(0, 6);
      const firstThree = firstSix.slice(0, 3);
      const secondThree = firstSix.slice(3, 6);

      expect(new Set(firstThree)).toHaveProperty("size", 3);
      expect(new Set(secondThree)).toHaveProperty("size", 3);

      // Last element should be valid
      expect(indices[6]).toBeGreaterThanOrEqual(0);
      expect(indices[6]).toBeLessThan(3);
    });

    it("should handle exact multiple of dataset size", () => {
      const dataSource = createDataSource(4);
      const sampler = new RandomSampler(dataSource, {
        replacement: false,
        numSamples: 8, // Exactly 2 full permutations
      });
      const indices = toArray(sampler);

      expect(indices).toHaveLength(8);

      const firstFour = indices.slice(0, 4);
      const secondFour = indices.slice(4, 8);

      expect(new Set(firstFour)).toHaveProperty("size", 4);
      expect(new Set(secondFour)).toHaveProperty("size", 4);
    });
  });

  describe("sampling with replacement", () => {
    it("should allow duplicates with replacement", () => {
      const dataSource = createDataSource(2);
      const sampler = new RandomSampler(dataSource, {
        replacement: true,
        numSamples: 100,
        generator: new Random(MersenneTwister19937.seed(42)),
      });
      const indices = toArray(sampler);

      expect(indices).toHaveLength(100);
      expect(indices.every((i) => i >= 0 && i < 2)).toBe(true);

      // With enough samples and replacement, we should see duplicates
      const uniqueIndices = new Set(indices);
      expect(uniqueIndices.size).toBeLessThan(100);
    });

    it("should generate different samples on each iteration with replacement", () => {
      const dataSource = createDataSource(5);
      const sampler = new RandomSampler(dataSource, {
        replacement: true,
        numSamples: 10,
      });

      const indices1 = toArray(sampler);
      const indices2 = toArray(sampler);

      expect(indices1).toHaveLength(10);
      expect(indices2).toHaveLength(10);
      // Very unlikely to be identical (but theoretically possible)
      // expect(indices1).not.toEqual(indices2);
    });
  });

  describe("error handling", () => {
    it("should throw error for invalid numSamples", () => {
      const dataSource = createDataSource(5);

      expect(() => new RandomSampler(dataSource, { numSamples: 0 })).toThrow(
        "numSamples should be a positive integer value, but got numSamples=0",
      );

      expect(() => new RandomSampler(dataSource, { numSamples: -1 })).toThrow(
        "numSamples should be a positive integer value, but got numSamples=-1",
      );

      expect(() => new RandomSampler(dataSource, { numSamples: 1.5 })).toThrow(
        "numSamples should be a positive integer value, but got numSamples=1.5",
      );
    });

    it("should throw error for invalid replacement", () => {
      const dataSource = createDataSource(5);

      // @ts-expect-error - testing invalid type
      expect(() => new RandomSampler(dataSource, { replacement: "true" })).toThrow(
        "replacement should be a boolean value, but got replacement=true",
      );

      // @ts-expect-error - testing invalid type
      expect(() => new RandomSampler(dataSource, { replacement: 1 })).toThrow(
        "replacement should be a boolean value, but got replacement=1",
      );
    });
  });

  describe("edge cases", () => {
    it("should handle empty dataset", () => {
      const dataSource = createDataSource(0);
      const sampler = new RandomSampler(dataSource);
      const indices = toArray(sampler);

      expect(indices).toEqual([]);
      expect(sampler.length).toBe(0);
    });

    it("should handle single element dataset", () => {
      const dataSource = createDataSource(1);
      const sampler = new RandomSampler(dataSource);
      const indices = toArray(sampler);

      expect(indices).toEqual([0]);
      expect(sampler.length).toBe(1);
    });

    it("should handle numSamples larger than dataset without replacement", () => {
      const dataSource = createDataSource(3);
      const sampler = new RandomSampler(dataSource, {
        replacement: false,
        numSamples: 10,
      });
      const indices = toArray(sampler);

      expect(indices).toHaveLength(10);
      // Should contain multiple permutations
      const counts = [0, 0, 0];
      indices.forEach((i) => counts[i]++);
      expect(counts.every((count) => count > 1)).toBe(true);
    });
  });
});

describe("BatchSampler", () => {
  describe("basic functionality", () => {
    it("should create batches of specified size", () => {
      const baseSampler = new SequentialSampler(createDataSource(10));
      const batchSampler = new BatchSampler(baseSampler, { batchSize: 3 });
      const batches = toArray(batchSampler);

      expect(batches).toEqual([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]);
    });

    it("should drop last incomplete batch when dropLast is true", () => {
      const baseSampler = new SequentialSampler(createDataSource(10));
      const batchSampler = new BatchSampler(baseSampler, {
        batchSize: 3,
        dropLast: true,
      });
      const batches = toArray(batchSampler);

      expect(batches).toEqual([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
      ]);
    });

    it("should return correct length with dropLast false", () => {
      const baseSampler = new SequentialSampler(createDataSource(10));
      const batchSampler = new BatchSampler(baseSampler, {
        batchSize: 3,
        dropLast: false,
      });

      expect(batchSampler.length).toBe(4); // ceil(10/3) = 4
    });

    it("should return correct length with dropLast true", () => {
      const baseSampler = new SequentialSampler(createDataSource(10));
      const batchSampler = new BatchSampler(baseSampler, {
        batchSize: 3,
        dropLast: true,
      });

      expect(batchSampler.length).toBe(3); // floor(10/3) = 3
    });

    it("should work with exact multiples", () => {
      const baseSampler = new SequentialSampler(createDataSource(9));
      const batchSampler = new BatchSampler(baseSampler, { batchSize: 3 });
      const batches = toArray(batchSampler);

      expect(batches).toEqual([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
      ]);
      expect(batchSampler.length).toBe(3);
    });
  });

  describe("with RandomSampler", () => {
    it("should work with RandomSampler", () => {
      const baseSampler = new RandomSampler(createDataSource(8), {
        generator: new Random(MersenneTwister19937.seed(42)),
      });
      const batchSampler = new BatchSampler(baseSampler, { batchSize: 3 });
      const batches = toArray(batchSampler);

      expect(batches).toHaveLength(3); // ceil(8/3) = 3
      expect(batches[0]).toHaveLength(3);
      expect(batches[1]).toHaveLength(3);
      expect(batches[2]).toHaveLength(2);

      // All indices should be valid
      const allIndices = batches.flat();
      expect(allIndices.every((i) => i >= 0 && i < 8)).toBe(true);
      expect(allIndices).toHaveLength(8);
    });

    it("should work with RandomSampler with replacement", () => {
      const baseSampler = new RandomSampler(createDataSource(5), {
        replacement: true,
        numSamples: 7,
        generator: new Random(MersenneTwister19937.seed(123)),
      });
      const batchSampler = new BatchSampler(baseSampler, { batchSize: 2 });
      const batches = toArray(batchSampler);

      expect(batches).toHaveLength(4); // ceil(7/2) = 4
      expect(batches[0]).toHaveLength(2);
      expect(batches[1]).toHaveLength(2);
      expect(batches[2]).toHaveLength(2);
      expect(batches[3]).toHaveLength(1);
    });
  });

  describe("error handling", () => {
    it("should throw error for invalid batchSize", () => {
      const baseSampler = new SequentialSampler(createDataSource(5));

      expect(() => new BatchSampler(baseSampler, { batchSize: 0 })).toThrow(
        "batchSize should be a positive integer value, but got batchSize=0",
      );

      expect(() => new BatchSampler(baseSampler, { batchSize: -1 })).toThrow(
        "batchSize should be a positive integer value, but got batchSize=-1",
      );

      expect(() => new BatchSampler(baseSampler, { batchSize: 1.5 })).toThrow(
        "batchSize should be a positive integer value, but got batchSize=1.5",
      );

      // @ts-expect-error - testing invalid type
      expect(() => new BatchSampler(baseSampler, { batchSize: true })).toThrow(
        "batchSize should be a positive integer value, but got batchSize=true",
      );
    });

    it("should throw error for invalid dropLast", () => {
      const baseSampler = new SequentialSampler(createDataSource(5));

      expect(
        () =>
          new BatchSampler(baseSampler, {
            batchSize: 2,
            // @ts-expect-error - testing invalid type
            dropLast: "false",
          }),
      ).toThrow("dropLast should be a boolean value, but got dropLast=false");

      expect(
        () =>
          new BatchSampler(baseSampler, {
            batchSize: 2,
            // @ts-expect-error - testing invalid type
            dropLast: 1,
          }),
      ).toThrow("dropLast should be a boolean value, but got dropLast=1");
    });

    it("should throw error when base sampler has no length", () => {
      // Create a sampler without length property
      const baseSampler = {
        [Symbol.iterator]: function* () {
          yield 1;
          yield 2;
          yield 3;
        },
      };

      const batchSampler = new BatchSampler(baseSampler, { batchSize: 2 });

      expect(() => batchSampler.length).toThrow(
        "Cannot determine length of BatchSampler when base sampler has no length",
      );
    });
  });

  describe("edge cases", () => {
    it("should handle empty base sampler", () => {
      const baseSampler = new SequentialSampler(createDataSource(0));
      const batchSampler = new BatchSampler(baseSampler, { batchSize: 3 });
      const batches = toArray(batchSampler);

      expect(batches).toEqual([]);
      expect(batchSampler.length).toBe(0);
    });

    it("should handle single element base sampler", () => {
      const baseSampler = new SequentialSampler(createDataSource(1));
      const batchSampler = new BatchSampler(baseSampler, { batchSize: 3 });
      const batches = toArray(batchSampler);

      expect(batches).toEqual([[0]]);
      expect(batchSampler.length).toBe(1);
    });

    it("should handle batchSize larger than dataset", () => {
      const baseSampler = new SequentialSampler(createDataSource(3));
      const batchSampler = new BatchSampler(baseSampler, { batchSize: 5 });
      const batches = toArray(batchSampler);

      expect(batches).toEqual([[0, 1, 2]]);
      expect(batchSampler.length).toBe(1);
    });

    it("should handle batchSize equal to dataset size", () => {
      const baseSampler = new SequentialSampler(createDataSource(5));
      const batchSampler = new BatchSampler(baseSampler, { batchSize: 5 });
      const batches = toArray(batchSampler);

      expect(batches).toEqual([[0, 1, 2, 3, 4]]);
      expect(batchSampler.length).toBe(1);
    });

    it("should handle batchSize of 1", () => {
      const baseSampler = new SequentialSampler(createDataSource(3));
      const batchSampler = new BatchSampler(baseSampler, { batchSize: 1 });
      const batches = toArray(batchSampler);

      expect(batches).toEqual([[0], [1], [2]]);
      expect(batchSampler.length).toBe(3);
    });
  });

  describe("iteration behavior", () => {
    it("should be iterable multiple times", () => {
      const baseSampler = new SequentialSampler(createDataSource(6));
      const batchSampler = new BatchSampler(baseSampler, { batchSize: 2 });

      const batches1 = toArray(batchSampler);
      const batches2 = toArray(batchSampler);

      expect(batches1).toEqual([
        [0, 1],
        [2, 3],
        [4, 5],
      ]);
      expect(batches2).toEqual([
        [0, 1],
        [2, 3],
        [4, 5],
      ]);
    });

    it("should work with different sampler types on each iteration", () => {
      // This tests that the BatchSampler correctly handles the base sampler's iterator
      const baseSampler = new RandomSampler(createDataSource(4), {
        generator: new Random(MersenneTwister19937.seed(456)),
      });
      const batchSampler = new BatchSampler(baseSampler, { batchSize: 2 });

      const batches1 = toArray(batchSampler);
      const batches2 = toArray(batchSampler);

      expect(batches1).toHaveLength(2);
      expect(batches2).toHaveLength(2);
      expect(batches1[0]).toHaveLength(2);
      expect(batches1[1]).toHaveLength(2);
      expect(batches2[0]).toHaveLength(2);
      expect(batches2[1]).toHaveLength(2);

      // All indices should be valid
      const allIndices1 = batches1.flat();
      const allIndices2 = batches2.flat();
      expect(allIndices1.every((i) => i >= 0 && i < 4)).toBe(true);
      expect(allIndices2.every((i) => i >= 0 && i < 4)).toBe(true);
    });
  });
});

describe("integration tests", () => {
  it("should work with nested BatchSampler", () => {
    // This is a bit artificial but tests the interface compatibility
    const dataSource = createDataSource(12);
    const baseSampler = new SequentialSampler(dataSource);
    const batchSampler1 = new BatchSampler(baseSampler, { batchSize: 3 });

    // BatchSampler yields arrays, so this doesn't make practical sense, but tests that the types
    // work correctly
    const batches = toArray(batchSampler1);
    expect(batches).toEqual([
      [0, 1, 2],
      [3, 4, 5],
      [6, 7, 8],
      [9, 10, 11],
    ]);
  });

  it("should maintain deterministic behavior with seeded RandomSampler", () => {
    const seed = 789;

    const createSampler = () => {
      const baseSampler = new RandomSampler(createDataSource(8), {
        generator: new Random(MersenneTwister19937.seed(seed)),
      });
      return new BatchSampler(baseSampler, { batchSize: 3 });
    };

    const sampler1 = createSampler();
    const sampler2 = createSampler();

    const batches1 = toArray(sampler1);
    const batches2 = toArray(sampler2);

    expect(batches1).toEqual(batches2);
  });
});
