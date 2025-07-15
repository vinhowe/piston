import { Tensor } from "@/__mocks__/tensor";
import { tensor } from "@/globals";
import { describe, expect, it, vi } from "vitest";

import { collate, type CollateFnMap, type CollateOptions, defaultCollate } from "./collate";

// Mock collate functions for testing
const mockStringCollate = (batch: unknown[]) => `collated_${(batch as string[]).join("_")}`;
const mockNumberCollate = (batch: unknown[]) => (batch as number[]).reduce((a, b) => a + b, 0);
const mockBooleanCollate = (batch: unknown[]) => (batch as boolean[]).every(Boolean);
const mockArrayCollate = (batch: unknown[], options?: CollateOptions) => {
  const arrays = batch as unknown[][];
  return arrays[0].map((_, i) =>
    collate(
      arrays.map((arr) => arr[i]),
      options,
    ),
  );
};

// Custom class for testing instanceof behavior
class MockClass {
  constructor(public value: string) {}
}
const mockClassCollate = (batch: unknown[]) =>
  `mock_${(batch as MockClass[]).map((item) => item.value).join("_")}`;

describe("collate", () => {
  describe("basic functionality", () => {
    it("should handle empty batch", () => {
      expect(() => collate([])).toThrow();
    });

    it("should throw error for unsupported types when no collate map provided", () => {
      expect(() => collate([Symbol("test")])).toThrow(
        /batch must contain tensors, numbers, dicts or lists/,
      );
    });

    it("should use custom collate function from map", () => {
      const collateFnMap: CollateFnMap = new Map();
      collateFnMap.set("string", mockStringCollate);
      const result = collate(["a", "b", "c"], { collateFnMap });
      expect(result).toBe("collated_a_b_c");
    });

    it("should handle instanceof matching for custom classes", () => {
      const collateFnMap: CollateFnMap = new Map();
      collateFnMap.set(MockClass, mockClassCollate);
      const batch = [new MockClass("test1"), new MockClass("test2")];
      const result = collate(batch, { collateFnMap });
      expect(result).toBe("mock_test1_test2");
    });

    it("should prioritize exact type match over instanceof", () => {
      const exactMatch = () => "exact";
      const instanceMatch = () => "instance";

      const collateFnMap: CollateFnMap = new Map();
      collateFnMap.set(String, instanceMatch);
      collateFnMap.set("string", exactMatch);

      const result = collate(["test"], { collateFnMap });
      expect(result).toBe("exact");
    });
  });

  describe("plain object handling", () => {
    it("should collate plain objects recursively", () => {
      const collateFnMap: CollateFnMap = new Map();
      collateFnMap.set("string", mockStringCollate);
      const batch = [
        { name: "alice", city: "boston" },
        { name: "bob", city: "chicago" },
      ];

      const result = collate(batch, { collateFnMap }) as Record<string, unknown>;
      expect(result.name).toBe("collated_alice_bob");
      expect(result.city).toBe("collated_boston_chicago");
    });

    it("should handle nested objects", () => {
      const collateFnMap: CollateFnMap = new Map();
      collateFnMap.set("string", mockStringCollate);
      const batch = [
        { user: { name: "alice", role: "admin" } },
        { user: { name: "bob", role: "user" } },
      ];

      const result = collate(batch, { collateFnMap }) as Record<string, unknown>;
      const user = result.user as Record<string, unknown>;
      expect(user.name).toBe("collated_alice_bob");
      expect(user.role).toBe("collated_admin_user");
    });

    it("should handle mixed types in object values", () => {
      const collateFnMap: CollateFnMap = new Map();
      collateFnMap.set("string", mockStringCollate);
      collateFnMap.set("number", mockNumberCollate);
      collateFnMap.set("boolean", mockBooleanCollate);
      const batch = [
        { name: "alice", age: 25, active: true },
        { name: "bob", age: 30, active: false },
      ];

      const result = collate(batch, { collateFnMap }) as Record<string, unknown>;
      expect(result.name).toBe("collated_alice_bob");
      expect(result.age).toBe(55);
      expect(result.active).toBe(false); // every(Boolean) of [true, false] = false
    });

    it("should not treat arrays as plain objects", () => {
      const collateFnMap: CollateFnMap = new Map();
      collateFnMap.set("string", mockStringCollate);
      const batch = [
        ["a", "b"],
        ["c", "d"],
      ];

      // Should be handled as arrays, not objects
      const result = collate(batch, { collateFnMap }) as string[];
      expect(result).toEqual(["collated_a_c", "collated_b_d"]);
    });

    it("should handle arrays as object values", () => {
      const collateFnMap: CollateFnMap = new Map();
      collateFnMap.set("string", mockStringCollate);
      const batch = [{ tags: ["frontend", "react"] }, { tags: ["backend", "node"] }];

      // Mock array collate will transpose arrays
      const result = collate(batch, { collateFnMap }) as Record<string, unknown>;
      const tags = result.tags as string[];
      expect(tags).toEqual(["collated_frontend_backend", "collated_react_node"]);
    });

    it("should not treat instances of custom classes as plain objects", () => {
      const collateFnMap: CollateFnMap = new Map();
      collateFnMap.set(MockClass, mockClassCollate);
      const batch = [new MockClass("test1"), new MockClass("test2")];

      const result = collate(batch, { collateFnMap });
      expect(result).toBe("mock_test1_test2");
    });
  });

  describe("array handling", () => {
    it("should transpose and collate arrays", () => {
      const collateFnMap: CollateFnMap = new Map();
      collateFnMap.set("string", mockStringCollate);
      const batch = [
        ["a", "b"],
        ["c", "d"],
      ];

      const result = collate(batch, { collateFnMap }) as string[];
      expect(result).toEqual(["collated_a_c", "collated_b_d"]);
    });

    it("should handle nested array collation", () => {
      const collateFnMap: CollateFnMap = new Map();
      collateFnMap.set("string", mockStringCollate);
      const batch = [
        [["a"], ["b"]],
        [["c"], ["d"]],
      ];

      const result = collate(batch, { collateFnMap }) as string[][];
      expect(result).toEqual([["collated_a_c"], ["collated_b_d"]]);
    });

    it("should throw error for arrays of inconsistent length", () => {
      const collateFnMap: CollateFnMap = new Map();
      collateFnMap.set("string", mockStringCollate);
      const batch = [["a", "b"], ["c"]]; // Different lengths

      expect(() => collate(batch, { collateFnMap })).toThrow(
        "each element in list of batch should be of equal size",
      );
    });

    it("should handle empty arrays", () => {
      const collateFnMap: CollateFnMap = new Map();
      collateFnMap.set("string", mockStringCollate);
      const batch = [[], []];

      const result = collate(batch, { collateFnMap }) as unknown[];
      expect(result).toEqual([]);
    });

    it("should handle mixed types within arrays", () => {
      const collateFnMap: CollateFnMap = new Map();
      collateFnMap.set("string", mockStringCollate);
      collateFnMap.set("number", mockNumberCollate);
      const batch = [
        ["a", 1],
        ["b", 2],
      ];

      const result = collate(batch, { collateFnMap }) as [string, number];
      expect(result).toEqual(["collated_a_b", 3]);
    });

    it("should use custom array collate function when provided", () => {
      const collateFnMap: CollateFnMap = new Map();
      collateFnMap.set("string", mockStringCollate);
      collateFnMap.set(Array, mockArrayCollate);
      const batch = [
        ["first", "second"],
        ["third", "fourth"],
      ];

      const result = collate(batch, { collateFnMap }) as string[];
      expect(result).toEqual(["collated_first_third", "collated_second_fourth"]);
    });
  });

  describe("recursive behavior", () => {
    it("should handle arrays of objects", () => {
      const collateFnMap: CollateFnMap = new Map();
      collateFnMap.set("string", mockStringCollate);
      const batch = [
        [{ name: "alice" }, { name: "bob" }],
        [{ name: "charlie" }, { name: "diana" }],
      ];

      const result = collate(batch, { collateFnMap }) as Record<string, unknown>[];
      expect(result[0].name).toBe("collated_alice_charlie");
      expect(result[1].name).toBe("collated_bob_diana");
    });

    it("should handle objects with array values", () => {
      const collateFnMap: CollateFnMap = new Map();
      collateFnMap.set("string", mockStringCollate);
      const batch = [{ names: ["alice", "bob"] }, { names: ["charlie", "diana"] }];

      const result = collate(batch, { collateFnMap }) as Record<string, unknown>;
      const names = result.names as string[];
      expect(names).toEqual(["collated_alice_charlie", "collated_bob_diana"]);
    });

    it("should pass collate options through recursive calls", () => {
      let callCount = 0;
      const trackingCollate = (_batch: unknown[], options?: CollateOptions) => {
        callCount++;
        expect(options?.collateFnMap).toBeDefined();
        return `call_${callCount}`;
      };

      const collateFnMap: CollateFnMap = new Map();
      collateFnMap.set("string", trackingCollate);
      const batch = [{ nested: { value: "test" } }, { nested: { value: "test2" } }];

      collate(batch, { collateFnMap });
      expect(callCount).toBe(1); // Should be called once for the nested values
    });
  });

  describe("edge cases", () => {
    it("should handle null values in objects", () => {
      const collateFnMap: CollateFnMap = new Map();
      collateFnMap.set("string", mockStringCollate);
      const batch = [
        { name: "alice", value: null },
        { name: "bob", value: null },
      ];

      expect(() => collate(batch, { collateFnMap })).toThrow();
    });

    it("should handle undefined values", () => {
      const batch = [undefined, undefined];
      expect(() => collate(batch)).toThrow();
    });

    it("should work with single item batches", () => {
      const collateFnMap: CollateFnMap = new Map();
      collateFnMap.set("string", mockStringCollate);
      const result = collate(["test"], { collateFnMap });
      expect(result).toBe("collated_test");
    });

    it("should handle deeply nested structures", () => {
      const collateFnMap: CollateFnMap = new Map();
      collateFnMap.set("string", mockStringCollate);
      const batch = [
        { level1: { level2: { level3: ["deep"] } } },
        { level1: { level2: { level3: ["value"] } } },
      ];

      const result = collate(batch, { collateFnMap }) as {
        level1: { level2: { level3: string[] } };
      };

      expect(result.level1.level2.level3).toEqual(["collated_deep_value"]);
    });
  });

  describe("collate function map behavior", () => {
    it("should work without collate function map", () => {
      const batch = [{ a: 1 }, { a: 2 }];
      expect(() => collate(batch)).toThrow(); // No collate function for numbers
    });

    it("should iterate through map in insertion order", () => {
      const calls: string[] = [];
      const firstMatch = () => {
        calls.push("first");
        return "first";
      };
      const secondMatch = () => {
        calls.push("second");
        return "second";
      };

      const collateFnMap: CollateFnMap = new Map();
      collateFnMap.set(String, firstMatch);
      collateFnMap.set(Object, secondMatch);

      collate(["test"], { collateFnMap });
      expect(calls).toEqual(["first"]);
    });

    it("should handle custom types properly", () => {
      class CustomType {
        constructor(public value: number) {}
      }

      const customCollate = (batch: unknown[]) =>
        (batch as CustomType[]).reduce((sum, item) => sum + item.value, 0);

      const collateFnMap: CollateFnMap = new Map();
      collateFnMap.set(CustomType, customCollate);
      const batch = [new CustomType(1), new CustomType(2), new CustomType(3)];

      const result = collate(batch, { collateFnMap });
      expect(result).toBe(6);
    });
  });
});

describe("defaultCollate", () => {
  describe("primitive types", () => {
    it("should collate numbers into a tensor", () => {
      const batch = [1, 2, 3, 4];
      const result = defaultCollate(batch);

      // Should return a tensor (mocked)
      expect(result).toBeInstanceOf(Tensor);
    });

    it("should collate booleans into a tensor", () => {
      const batch = [true, false, true, false];
      const result = defaultCollate(batch);

      // Should return a tensor (mocked)
      expect(result).toBeInstanceOf(Tensor);
    });

    it("should collate bigints into a tensor", () => {
      const batch = [1n, 2n, 3n, 4n];
      const result = defaultCollate(batch);

      // Should return a tensor (mocked)
      expect(result).toBeInstanceOf(Tensor);
    });

    it("should collate strings into a string array", () => {
      const batch = ["hello", "world", "test"];
      const result = defaultCollate(batch);

      // Should return the same array for strings
      expect(result).toEqual(["hello", "world", "test"]);
    });
  });

  describe("tensor types", () => {
    it("should collate Tensor objects", () => {
      const tensor1 = tensor([1, 2]);
      const tensor2 = tensor([3, 4]);
      const batch = [tensor1, tensor2];

      // Create a custom collate map that includes our mock Tensor
      const collateFnMap: CollateFnMap = new Map();
      collateFnMap.set(Tensor, (_batch: unknown[]) => {
        // Mock tensor collation by stacking the tensors
        return tensor([0, 1]); // Return a mock stacked tensor
      });

      const result = collate(batch, { collateFnMap });

      // Should return a stacked tensor (mocked)
      expect(result).toBeInstanceOf(Tensor);
    });
  });

  describe("typed arrays", () => {
    it("should collate Float32Array into a tensor", () => {
      const array1 = new Float32Array([1.0, 2.0]);
      const array2 = new Float32Array([3.0, 4.0]);
      const batch = [array1, array2];

      const result = defaultCollate(batch);

      // Should return a tensor (mocked)
      expect(result).toBeInstanceOf(Tensor);
    });

    it("should collate Int32Array into a tensor", () => {
      const array1 = new Int32Array([1, 2]);
      const array2 = new Int32Array([3, 4]);
      const batch = [array1, array2];

      const result = defaultCollate(batch);

      // Should return a tensor (mocked)
      expect(result).toBeInstanceOf(Tensor);
    });

    it("should collate Uint32Array into a tensor", () => {
      const array1 = new Uint32Array([1, 2]);
      const array2 = new Uint32Array([3, 4]);
      const batch = [array1, array2];

      const result = defaultCollate(batch);

      // Should return a tensor (mocked)
      expect(result).toBeInstanceOf(Tensor);
    });

    it("should collate Uint8Array into a tensor", () => {
      const array1 = new Uint8Array([1, 2]);
      const array2 = new Uint8Array([3, 4]);
      const batch = [array1, array2];

      const result = defaultCollate(batch);

      // Should return a tensor (mocked)
      expect(result).toBeInstanceOf(Tensor);
    });

    it("should collate Float64Array into a tensor", () => {
      const array1 = new Float64Array([1.0, 2.0]);
      const array2 = new Float64Array([3.0, 4.0]);
      const batch = [array1, array2];

      const result = defaultCollate(batch);

      // Should return a tensor (mocked)
      expect(result).toBeInstanceOf(Tensor);
    });
  });

  describe("complex structures", () => {
    it("should collate objects recursively", () => {
      const batch = [
        { name: "alice", age: 25, active: true },
        { name: "bob", age: 30, active: false },
      ];

      const result = defaultCollate(batch) as Record<string, unknown>;

      // Strings should remain as arrays
      expect(result.name).toEqual(["alice", "bob"]);
      // Numbers should become tensors
      expect(result.age).toBeInstanceOf(Tensor);
      // Booleans should become tensors
      expect(result.active).toBeInstanceOf(Tensor);
    });

    it("should collate arrays of numbers", () => {
      const batch = [
        [1, 2],
        [3, 4],
      ];

      const result = defaultCollate(batch) as Tensor[];

      // Should return array of tensors
      expect(Array.isArray(result)).toBe(true);
      expect(result[0]).toBeInstanceOf(Tensor);
      expect(result[1]).toBeInstanceOf(Tensor);
    });

    it("should collate arrays of strings", () => {
      const batch = [
        ["a", "b"],
        ["c", "d"],
      ];

      const result = defaultCollate(batch) as string[];

      // Should return array of string arrays
      expect(Array.isArray(result)).toBe(true);
      expect(result[0]).toEqual(["a", "c"]);
      expect(result[1]).toEqual(["b", "d"]);
    });

    it("should handle mixed type objects", () => {
      const batch = [
        {
          name: "alice",
          scores: [90, 85],
          metadata: { country: "US", verified: true },
        },
        {
          name: "bob",
          scores: [88, 92],
          metadata: { country: "UK", verified: false },
        },
      ];

      const result = defaultCollate(batch) as Record<string, unknown>;

      // Names should be string arrays
      expect(result.name).toEqual(["alice", "bob"]);

      // Scores should be arrays of tensors
      const scores = result.scores as Tensor[];
      expect(Array.isArray(scores)).toBe(true);
      expect(scores[0]).toBeInstanceOf(Tensor);
      expect(scores[1]).toBeInstanceOf(Tensor);

      // Nested objects should be collated recursively
      const metadata = result.metadata as Record<string, unknown>;
      expect(metadata.country).toEqual(["US", "UK"]);
      expect(metadata.verified).toBeInstanceOf(Tensor);
    });
  });

  describe("edge cases", () => {
    it("should handle empty arrays", () => {
      const batch = [[], []];
      const result = defaultCollate(batch);
      expect(result).toEqual([]);
    });

    it("should handle single element batches", () => {
      const batch = [42];
      const result = defaultCollate(batch);
      expect(result).toBeInstanceOf(Tensor);
    });

    it("should handle bigint precision loss warning", () => {
      const consoleSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

      const largeBigInt = BigInt(Number.MAX_SAFE_INTEGER) + 1n;
      const batch = [largeBigInt, 2n];

      const result = defaultCollate(batch);

      expect(result).toBeInstanceOf(Tensor);
      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining(`BigInt ${largeBigInt} exceeds safe integer range`),
      );

      consoleSpy.mockRestore();
    });

    it("should throw error for unsupported types", () => {
      const batch = [Symbol("test"), Symbol("test2")];

      expect(() => defaultCollate(batch)).toThrow();
    });
  });
});
