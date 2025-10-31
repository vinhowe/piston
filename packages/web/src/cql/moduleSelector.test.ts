import { beforeEach, describe, expect, it, vi } from "vitest";

import { ModelNode } from "@/types";

import { createIndexedTestModel, createSimpleTestModel } from "./__mocks__/mockModels";
import { ModuleSelector, selectModules } from "./moduleSelector";
import { parse } from "./parser";

describe("ModuleSelector", () => {
  let simpleModel: ReturnType<typeof createSimpleTestModel>;
  let indexedModel: ReturnType<typeof createIndexedTestModel>;

  beforeEach(() => {
    simpleModel = createSimpleTestModel();
    indexedModel = createIndexedTestModel();

    // Setup mock implementation for findDescendants
    // simpleModel.adapter.findDescendants = vi
    const findDescendantsSpy = vi
      .fn()
      .mockImplementation(
        (node: ModelNode, predicate: (node: ModelNode) => boolean, results: ModelNode[]) => {
          // Use a recursive approach to properly traverse the tree
          const traverse = (currentNode: ModelNode) => {
            const children = currentNode.getChildren();
            for (const child of children) {
              if (predicate(child)) {
                results.push(child);
              }
              // Recursively traverse the child's children
              traverse(child);
            }
          };

          // Start the traversal from the initial node
          traverse(node);
        },
      );
    simpleModel.adapter.findDescendants = findDescendantsSpy;
    indexedModel.adapter.findDescendants = findDescendantsSpy;
  });

  describe("ModuleSelector class", () => {
    it("should create a selector with default options", () => {
      const selector = new ModuleSelector(simpleModel.adapter);
      expect(selector).toBeDefined();
    });

    it("should create a selector with custom options", () => {
      const selector = new ModuleSelector(simpleModel.adapter, {
        includeDescendants: true,
        maxDepth: 3,
      });
      expect(selector).toBeDefined();
    });

    it("should select modules based on name", () => {
      const selector = new ModuleSelector(simpleModel.adapter);
      const parsedQuery = parse("linear1")[0];

      const result = selector.selectModules(parsedQuery, simpleModel.root);

      expect(result.matchedModules).toHaveLength(1);
      expect(result.matchedModules[0].name).toBe("linear1");
    });

    it("should include descendants when that option is set", () => {
      const selector = new ModuleSelector(simpleModel.adapter, {
        includeDescendants: true,
      });
      const parsedQuery = parse("model")[0];

      const result = selector.selectModules(parsedQuery, simpleModel.root);

      // Should include all descendants of the model
      expect(result.matchedModules.length).toBeGreaterThan(1);
    });

    it("should respect maxDepth option", () => {
      // Create a spyable version of findDescendants
      const findDescendantsSpy = vi.fn().mockImplementation(simpleModel.adapter.findDescendants);
      simpleModel.adapter.findDescendants = findDescendantsSpy;

      const selector = new ModuleSelector(simpleModel.adapter, {
        includeDescendants: true,
        maxDepth: 2,
      });
      const parsedQuery = parse("model")[0];

      selector.selectModules(parsedQuery, simpleModel.root);

      // Verify maxDepth was passed to findDescendants
      expect(findDescendantsSpy).toHaveBeenCalledWith(
        expect.anything(),
        expect.anything(),
        expect.anything(),
        2,
      );
    });
  });

  describe("selectModules function", () => {
    it("should find modules by name", () => {
      const parsedQuery = parse("linear1")[0];

      const result = selectModules(parsedQuery, simpleModel.root, simpleModel.adapter);

      expect(result.matchedModules).toHaveLength(1);
      expect(result.matchedModules[0].name).toBe("linear1");
    });

    it("should find modules by type", () => {
      const parsedQuery = parse(".Linear")[0];

      const result = selectModules(parsedQuery, simpleModel.root, simpleModel.adapter);

      expect(result.matchedModules).toHaveLength(3);
      expect(result.matchedModules.every((m) => m.typeName === "Linear")).toBe(true);
    });

    it("should find modules by child combinator", () => {
      const parsedQuery = parse("model > linear1")[0];

      const result = selectModules(parsedQuery, simpleModel.root, simpleModel.adapter);

      expect(result.matchedModules).toHaveLength(1);
      expect(result.matchedModules[0].name).toBe("linear1");
    });

    it("should find modules by descendant combinator", () => {
      const parsedQuery = parse("model nested linear")[0];

      const result = selectModules(parsedQuery, simpleModel.root, simpleModel.adapter);

      expect(result.matchedModules).toHaveLength(1);
      expect(result.matchedModules[0].name).toBe("linear");
    });

    it("should find modules by index", () => {
      const parsedQuery = parse("model[1]")[0];

      const result = selectModules(parsedQuery, indexedModel.root, indexedModel.adapter);

      expect(result.matchedModules).toHaveLength(1);
      // The parsed model[1] should match a child at index 1
      expect(result.matchedModules[0]).toBeDefined();
    });

    it("should handle errors properly and return empty result", () => {
      const parsedQuery = parse("nonexistent")[0];

      const result = selectModules(parsedQuery, simpleModel.root, simpleModel.adapter);

      expect(result.matchedModules).toHaveLength(0);
      expect(result.context).toBeDefined();
    });

    it("should find modules by next-sibling combinator", () => {
      const parsedQuery = parse("linear1 + relu")[0];

      const result = selectModules(parsedQuery, simpleModel.root, simpleModel.adapter);

      expect(result.matchedModules).toHaveLength(1);
      expect(result.matchedModules[0].name).toBe("relu");
    });

    it("should find modules by subsequent-sibling combinator", () => {
      const parsedQuery = parse("linear1 ~ .Linear")[0];

      const result = selectModules(parsedQuery, simpleModel.root, simpleModel.adapter);

      // Should find linear2 (which is a Linear type that comes after linear1)
      expect(result.matchedModules).toHaveLength(1);
      expect(result.matchedModules[0].name).toBe("linear2");
      expect(result.matchedModules[0].typeName).toBe("Linear");
    });

    it("should demonstrate difference between next-sibling and subsequent-sibling", () => {
      // Test next-sibling: should only find the immediate next sibling
      const nextSiblingQuery = parse("linear1 + *")[0];
      const nextSiblingResult = selectModules(
        nextSiblingQuery,
        simpleModel.root,
        simpleModel.adapter,
      );

      // Should only find relu (the immediate next sibling)
      expect(nextSiblingResult.matchedModules).toHaveLength(1);
      expect(nextSiblingResult.matchedModules[0].name).toBe("relu");

      // Test subsequent-sibling: should find all subsequent siblings
      const subsequentSiblingQuery = parse("linear1 ~ *")[0];
      const subsequentSiblingResult = selectModules(
        subsequentSiblingQuery,
        simpleModel.root,
        simpleModel.adapter,
      );

      // Should find relu, linear2, and nested (all subsequent siblings)
      expect(subsequentSiblingResult.matchedModules.length).toBeGreaterThan(1);
      const siblingNames = subsequentSiblingResult.matchedModules.map((m) => m.name);
      expect(siblingNames).toContain("relu");
      expect(siblingNames).toContain("linear2");
      expect(siblingNames).toContain("nested");
    });

    it("should find modules by wildcard with child combinator", () => {
      const parsedQuery = parse("model > *")[0];

      const result = selectModules(parsedQuery, simpleModel.root, simpleModel.adapter);

      // Should match all direct children of model
      expect(result.matchedModules.length).toBeGreaterThan(0);
      expect(result.matchedModules.every((m) => m.parent && m.parent.name === "model")).toBe(true);
    });

    it("should find modules by wildcard with descendant combinator", () => {
      const parsedQuery = parse("model *")[0];

      const result = selectModules(parsedQuery, simpleModel.root, simpleModel.adapter);

      // Should match all nodes under model (but not model itself since it's selected first)
      expect(result.matchedModules.length).toBeGreaterThan(0);
      // Should not include model itself (it was the context, not a match)
      expect(result.matchedModules.some((m) => m.name === "model")).toBe(false);
      // All matched modules should be descendants of model (have model as an ancestor)
      const isDescendantOfModel = (node: ModelNode): boolean => {
        let current = node.parent;
        while (current) {
          if (current.name === "model") return true;
          current = current.parent;
        }
        return false;
      };
      expect(result.matchedModules.every(isDescendantOfModel)).toBe(true);
    });

    it("should find modules by wildcard", () => {
      const parsedQuery = parse("*")[0];

      const result = selectModules(parsedQuery, simpleModel.root, simpleModel.adapter);

      // Should match the root (model) and all its descendants
      expect(result.matchedModules.length).toBeGreaterThan(0);
      // Should include model itself
      expect(result.matchedModules.some((m) => m.name === "model")).toBe(true);
      // Should also include descendants
      const descendants = result.matchedModules.filter((m) => m.name !== "model");
      expect(descendants.length).toBeGreaterThan(0);
      // All descendants should have model as an ancestor
      const isDescendantOfModel = (node: ModelNode): boolean => {
        let current = node.parent;
        while (current) {
          if (current.name === "model") return true;
          current = current.parent;
        }
        return false;
      };
      expect(descendants.every(isDescendantOfModel)).toBe(true);
    });

    it("should find modules by next-sibling combinator with type selector", () => {
      const parsedQuery = parse(".Linear + *")[0];

      const result = selectModules(parsedQuery, simpleModel.root, simpleModel.adapter);

      // Should find immediate next siblings of Linear modules
      // linear1 (Linear) -> relu, linear2 (Linear) -> nested
      expect(result.matchedModules.length).toBeGreaterThan(0);
      const siblingNames = result.matchedModules.map((m) => m.name);
      expect(siblingNames).toContain("relu"); // next sibling of linear1
      expect(siblingNames).toContain("nested"); // next sibling of linear2
    });

    it("should find modules by subsequent-sibling combinator with type selector", () => {
      const parsedQuery = parse(".Linear ~ *")[0];

      const result = selectModules(parsedQuery, simpleModel.root, simpleModel.adapter);

      // Should find all subsequent siblings of Linear modules
      expect(result.matchedModules.length).toBeGreaterThan(0);
      const siblingNames = result.matchedModules.map((m) => m.name);
      // linear1 (Linear) should match: relu, linear2, nested
      // linear2 (Linear) should match: nested
      expect(siblingNames).toContain("relu");
      expect(siblingNames).toContain("linear2");
      expect(siblingNames).toContain("nested");
    });
  });

  describe("selectModulesFromParsedQuery function", () => {
    it("should select modules directly from parsed query", () => {
      const parsedQuery = parse("model > linear1")[0];

      const result = selectModules(parsedQuery, simpleModel.root, simpleModel.adapter);

      expect(result.matchedModules).toHaveLength(1);
      expect(result.matchedModules[0].name).toBe("linear1");
    });
  });
});
