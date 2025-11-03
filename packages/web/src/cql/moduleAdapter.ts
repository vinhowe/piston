import { Module } from "../nn/module";
import { Tensor } from "../tensor";
import { ModelAdapter, ModelNode } from "./types";

/**
 * An adapter that wraps a Module instance and implements the ModelNode interface.
 */
export class ModuleNodeAdapter implements ModelNode {
  constructor(
    private module: Module,
    public name: string,
    public parent: ModelNode | null = null,
  ) {}

  public getModule(): Module {
    return this.module;
  }

  get typeName(): string {
    return this.module.constructor.name;
  }

  /**
   * Get all children of this module.
   */
  getChildren(): ModelNode[] {
    const children: ModelNode[] = [];
    for (const [name, childModule] of this.module.namedChildren()) {
      if (childModule) {
        children.push(new ModuleNodeAdapter(childModule, name, this));
      }
    }
    return children;
  }

  /**
   * Get a specific child by name or index.
   */
  getChild(nameOrIndex: string | number): ModelNode | undefined {
    let childModule: Module | undefined;

    if (typeof nameOrIndex === "string") {
      childModule = (this.module as unknown as Record<string, unknown>)[nameOrIndex] as Module;
    } else if (typeof nameOrIndex === "number") {
      childModule = (this.module as unknown as Record<number, Module>)[nameOrIndex];
    }

    return childModule
      ? new ModuleNodeAdapter(childModule, nameOrIndex.toString(), this)
      : undefined;
  }

  /**
   * Get all parameters of this module.
   */
  getParameters(): Record<string, Tensor> {
    return Object.fromEntries(this.module.namedParameters("", false));
  }

  /**
   * Check if the module name matches a regex pattern.
   */
  matchesRegex(regex: RegExp): boolean {
    return regex.test(this.name);
  }

  path(): string[] {
    const parentPath = this.parent ? this.parent.path() : [];
    return [...parentPath, this.name];
  }
}

/**
 * A concrete implementation of the ModelAdapter interface.
 */
export class PistonModelAdapter implements ModelAdapter {
  private rootModule: Module;

  constructor(rootModule: Module) {
    this.rootModule = rootModule;
  }

  /**
   * Get the root modules.
   */
  getRootModules(): ModelNode[] {
    return [new ModuleNodeAdapter(this.rootModule, "root")];
  }

  /**
   * Get a parameter by name from a node.
   */
  getParameter(node: ModelNode, name: string): Tensor | undefined {
    const params = node.getParameters();
    return params[name];
  }

  /**
   * Get a child node by name.
   */
  getChild(node: ModelNode, name: string): ModelNode | undefined {
    return node.getChild(name);
  }

  /**
   * Find descendants of a node that match a predicate.
   */
  findDescendants(
    node: ModelNode,
    predicate: (node: ModelNode) => boolean,
    results: ModelNode[],
    maxDepth?: number,
  ): void {
    if (maxDepth !== undefined && maxDepth <= 0) {
      return;
    }

    // Check direct children
    const children = node.getChildren();
    for (const child of children) {
      if (predicate(child)) {
        results.push(child);
      }
      // Recursively check descendants
      this.findDescendants(
        child,
        predicate,
        results,
        maxDepth !== undefined ? maxDepth - 1 : undefined,
      );
    }
  }
}

/**
 * Factory function to create a module adapter and model adapter.
 */
export function createModuleAdapter(module: Module): {
  nodeAdapter: ModelNode;
  modelAdapter: ModelAdapter;
} {
  const nodeAdapter = new ModuleNodeAdapter(module, "root");
  const modelAdapter = new PistonModelAdapter(module);

  return {
    nodeAdapter,
    modelAdapter,
  };
}
