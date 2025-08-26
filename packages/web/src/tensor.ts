/* eslint-disable @typescript-eslint/no-unsafe-declaration-merging */
// Proxy-based Tensor wrapper that forwards to Tensor_wasm but exposes the same
// surface area via unsafe declaration merging for TypeScript typing.
import { Tensor_wasm } from "./wasm";

export type OpDescription = {
  name: string;
  // TODO(vinhowe): Fix this to be like the actual type
  fields: Record<string, unknown>;
};

// Track the inner wasm tensor for both the target and its proxy
const innerMap: WeakMap<object, Tensor_wasm> = new WeakMap();

function wrapIfTensor(value: unknown): unknown {
  return value instanceof Tensor_wasm ? Tensor._wrap(value) : value;
}

function wrapReturned(value: unknown): unknown {
  if (value instanceof Promise) {
    return value.then((v) => wrapIfTensor(v));
  }
  if (Array.isArray(value)) {
    return value.map((v) => wrapIfTensor(v));
  }
  return wrapIfTensor(value);
}

const proxyHandler: ProxyHandler<Tensor> = {
  get(target, prop, receiver) {
    const own = Reflect.get(target, prop, receiver) as unknown;
    if (own !== undefined) return own;

    const inner = innerMap.get(target);
    if (!inner) return undefined;

    const value = Reflect.get(inner, prop) as unknown;
    if (typeof value === "function") {
      return (...args: unknown[]) => {
        const unwrappedArgs = args.map((a) => (a instanceof Tensor ? Tensor._unwrap(a) : a));
        const fn = value as (...fnArgs: unknown[]) => unknown;
        const result = Reflect.apply(fn, inner, unwrappedArgs);
        return wrapReturned(result);
      };
    }
    return wrapReturned(value);
  },
  set(target, prop, value, receiver) {
    if (Object.prototype.hasOwnProperty.call(target, prop)) {
      return Reflect.set(target, prop, value, receiver);
    }
    const inner = innerMap.get(target);
    if (!inner) return false;
    const unwrapped = value instanceof Tensor ? Tensor._unwrap(value) : value;
    return Reflect.set(inner, prop, unwrapped);
  },
  has(target, prop) {
    const inner = innerMap.get(target);
    return Reflect.has(target, prop) || (inner ? Reflect.has(inner, prop) : false);
  },
  ownKeys(target) {
    const inner = innerMap.get(target);
    const own = Reflect.ownKeys(target);
    const innerKeys = inner ? Reflect.ownKeys(inner) : [];
    return [...new Set([...own, ...innerKeys])];
  },
  getOwnPropertyDescriptor(target, prop) {
    const own = Reflect.getOwnPropertyDescriptor(target, prop);
    if (own) return own;
    const inner = innerMap.get(target);
    if (!inner) return undefined;
    const desc = Reflect.getOwnPropertyDescriptor(inner, prop);
    return desc ? { ...desc, configurable: true } : undefined;
  },
};

export class Tensor {
  constructor(inner: Tensor_wasm) {
    innerMap.set(this, inner);
    return Tensor.createProxy(this);
  }

  private static createProxy(target: Tensor): Tensor {
    const proxy = new Proxy(target, proxyHandler) as Tensor;
    const inner = innerMap.get(target);
    if (inner) innerMap.set(proxy, inner);
    return proxy;
  }

  static _wrap(tensor: Tensor_wasm): Tensor {
    return new Tensor(tensor);
  }

  static _unwrap(tensor: Tensor): Tensor_wasm {
    const inner = innerMap.get(tensor);
    if (!inner) throw new Error("Tensor is missing its inner value");
    return inner._cloneWeak();
  }
}

// Unsafe declaration merging: give Tensor all Tensor_wasm members in type space
export interface Tensor extends Tensor_wasm {
  // Prevent empty-object-type lint while keeping structural merge
  readonly _typeBrand__tensor?: never;
}

// Allow static property/method access to fall through to wasm class via prototype chain
// We do this so we can access __wbg_piston_tensor for downcasting into wasm bindgen
Object.setPrototypeOf(Tensor, Tensor_wasm as unknown as object);
