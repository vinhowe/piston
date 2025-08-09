/* eslint-disable @typescript-eslint/no-unsafe-declaration-merging */
import { DimsType, ShapeType, TensorOrScalar } from "./types";
import { DType, Tensor_wasm } from "./wasm";

export type OpDescription = {
  name: string;
  // TODO(vinhowe): Fix this to be like the actual type
  fields: Record<string, unknown>;
};

/**
 * Promotes two data types to a common type according to a type hierarchy.
 * The promotion hierarchy is: uint32 < int32 < float16 < float32
 *
 * @param dtype1 - First data type
 * @param dtype2 - Second data type
 * @returns The promoted data type that can represent both inputs
 */
export function promoteTypes(dtype1: DType, dtype2: DType): DType {
  // If types are the same, no promotion needed
  if (dtype1.name === dtype2.name) {
    return dtype1;
  }

  if (dtype1.name === "U32" || dtype2.name === "U32") {
    if (dtype1.isFloatingPoint) {
      return dtype1;
    }
    if (dtype2.isFloatingPoint) {
      return dtype2;
    }
    throw new Error(
      `Promotion for uint16, uint32, uint64 types is not supported, attempted to promote ${dtype1.name} and ${dtype2.name}`,
    );
  }

  // Define type hierarchy using direct DType comparison
  // Each entry is [dtype, precedence] where higher precedence wins
  const getTypePrecedence = (dtype: DType): number => {
    if (dtype.name === "U32") return 0;
    if (dtype.name === "I32") return 1;
    if (dtype.name === "F16") return 2;
    if (dtype.name === "F32") return 3;
    throw new Error(`Cannot promote unknown type`);
  };

  const precedence1 = getTypePrecedence(dtype1);
  const precedence2 = getTypePrecedence(dtype2);

  // Return the type with higher precedence
  if (precedence1 >= precedence2) {
    return dtype1;
  } else {
    return dtype2;
  }
}

// Define operation name groups once so we can use them for both runtime wiring
// and compile-time typing via mapped types.
const binaryOps = [
  "add",
  "add_",
  "sub",
  "sub_",
  "mul",
  "mul_",
  "div",
  "div_",
  "pow",
  "pow_",
] as const satisfies readonly (keyof Tensor_wasm)[];

const binaryTensorOnlyOps = [
  "minimum",
  "maximum",
  "minimum_",
  "maximum_",
] as const satisfies readonly (keyof Tensor_wasm)[];

const ternaryOps = [
  "addcdiv",
  "addcdiv_",
  "addcmul",
  "addcmul_",
] as const satisfies readonly (keyof Tensor_wasm)[];

const cmpOps = [
  "eq",
  "eq_",
  "ne",
  "ne_",
  "gt",
  "gt_",
  "ge",
  "ge_",
  "lt",
  "lt_",
  "le",
  "le_",
] as const satisfies readonly (keyof Tensor_wasm)[];

const unaryOps = [
  "gelu",
  "gelu_",
  "tanh",
  "tanh_",
  "exp",
  "exp_",
  "log",
  "log_",
  "sin",
  "sin_",
  "cos",
  "cos_",
  "abs",
  "abs_",
  "sqrt",
  "sqrt_",
  "relu",
  "relu_",
  "relu2",
  "relu2_",
  "floor",
  "floor_",
  "ceil",
  "ceil_",
  "neg",
  "neg_",
  "sigmoid",
  "sigmoid_",
  "swiglu",
  "swiglu_",
  "silu",
  "silu_",
  "square",
  "square_",
  "recip",
  "recip_",
  "float",
  "half",
] as const satisfies readonly (keyof Tensor_wasm)[];

/**
 * Casts two tensors to a common promoted type if needed.
 * Returns the original tensors if no casting is required.
 *
 * @param tensor1 - First tensor
 * @param tensor2 - Second tensor
 * @returns Tuple of [tensor1, tensor2] with appropriate casting applied
 */
function promotedCast(tensor1: Tensor, tensor2: TensorOrScalar): [Tensor, TensorOrScalar] {
  if (!(tensor2 instanceof Tensor)) {
    return [tensor1, tensor2];
  }

  const dtype1 = tensor1.dtype;
  const dtype2 = tensor2.dtype;

  const promotedType = promoteTypes(dtype1, dtype2);

  // Cast each tensor only if its type is different from the promoted type
  const newTensor1 = dtype1 === promotedType ? tensor1 : tensor1.cast(promotedType);
  const newTensor2 = dtype2 === promotedType ? tensor2 : tensor2.cast(promotedType);

  return [newTensor1, newTensor2];
}

export class Tensor {
  constructor(private readonly innerTensor: Tensor_wasm) {}

  // Helper method to access inner tensor for internal use
  static _unwrap(tensor: Tensor): Tensor_wasm {
    return tensor?.innerTensor?._clone();
  }

  // Creates a new Tensor from a Tensor_wasm
  static _wrap(tensor: Tensor_wasm): Tensor {
    return new Tensor(tensor);
  }

  private wrappedOp(op: (selfClone: Tensor_wasm) => Tensor_wasm): Tensor {
    return Tensor._wrap(op(this._cloneInner()));
  }

  // Static factories for creating prototype methods
  private static makeBinaryOp(name: keyof Tensor_wasm) {
    return function (this: Tensor, other: TensorOrScalar): Tensor {
      const [promotedThis, promotedOther] = promotedCast(this, other);
      return promotedThis.wrappedOp((a: Tensor_wasm) =>
        (a[name] as (b: Tensor_wasm | number) => Tensor_wasm)(
          promotedOther instanceof Tensor ? promotedOther._cloneInner() : promotedOther,
        ),
      );
    };
  }

  private static makeBinaryOpTensorOnly(name: keyof Tensor_wasm) {
    return function (this: Tensor, other: Tensor): Tensor {
      const [promotedThis, promotedOther] = promotedCast(this, other);
      return promotedThis.wrappedOp((a: Tensor_wasm) =>
        (a[name] as (b: Tensor_wasm | number) => Tensor_wasm)(
          (promotedOther as Tensor)._cloneInner(),
        ),
      );
    };
  }

  private static makeTernaryOp(name: keyof Tensor_wasm) {
    return function (this: Tensor, tensor1: Tensor, tensor2: Tensor, value: number): Tensor {
      const [promotedTensor1, promotedTensor2] = promotedCast(tensor1, tensor2);
      const [promotedThis, _] = promotedCast(this, promotedTensor1);
      return promotedThis.wrappedOp((a: Tensor_wasm) =>
        (a[name] as (b: Tensor_wasm, c: Tensor_wasm, value: number) => Tensor_wasm)(
          promotedTensor1._cloneInner(),
          (promotedTensor2 as Tensor)._cloneInner(),
          value,
        ),
      );
    };
  }

  private static makeCmpOp(name: keyof Tensor_wasm) {
    return function (this: Tensor, other: TensorOrScalar): Tensor {
      const [promotedThis, promotedOther] = promotedCast(this, other);
      return promotedThis.wrappedOp((a: Tensor_wasm) =>
        (a[name] as (b: Tensor_wasm | number) => Tensor_wasm)(
          promotedOther instanceof Tensor ? promotedOther._cloneInner() : promotedOther,
        ),
      );
    };
  }

  private static makeUnaryOp(name: keyof Tensor_wasm) {
    return function (this: Tensor): Tensor {
      return this.wrappedOp((a: Tensor_wasm) => (a[name] as () => Tensor_wasm)());
    };
  }

  cast(dstDtype?: DType): Tensor {
    return this.wrappedOp((a: Tensor_wasm) => a.cast(dstDtype?._clone()));
  }

  groupNorm(numGroups: number, weight?: Tensor, bias?: Tensor, eps: number = 1e-5): Tensor {
    return this.wrappedOp((a: Tensor_wasm) =>
      a.groupNorm(numGroups, weight?._cloneInner(), bias ? bias._cloneInner() : null, eps),
    );
  }

  layerNorm(weight?: Tensor, bias?: Tensor, eps: number = 1e-5): Tensor {
    return this.wrappedOp((a: Tensor_wasm) =>
      a.layerNorm(weight?._cloneInner(), bias ? bias._cloneInner() : null, eps),
    );
  }

  rmsNorm(weight?: Tensor, eps: number = 1e-5): Tensor {
    return this.wrappedOp((a: Tensor_wasm) => a.rmsNorm(weight?._cloneInner(), eps));
  }

  conv1d(weight: Tensor, bias?: Tensor, stride: number = 1, padding: number = 0): Tensor {
    return this.wrappedOp((a: Tensor_wasm) =>
      a.conv1d(weight?._cloneInner(), bias ? bias._cloneInner() : null, stride, padding),
    );
  }

  softmax(dim: number): Tensor {
    return this.wrappedOp((a: Tensor_wasm) => a.softmax(dim));
  }

  rope(dim: number, base: number, offset: number): Tensor {
    return this.wrappedOp((a: Tensor_wasm) => a.rope_(dim, base, offset));
  }

  alibi(maxBias: number): Tensor {
    return this.wrappedOp((a: Tensor_wasm) => a.alibi(maxBias));
  }

  matmul(rhs: Tensor, transLhs: boolean = false, transRhs: boolean = false): Tensor {
    return this.wrappedOp((a: Tensor_wasm) => a.matmul(rhs._cloneInner(), transLhs, transRhs));
  }

  gemm(
    rhs: Tensor,
    bias?: Tensor,
    transLhs: boolean = false,
    transRhs: boolean = false,
    transOut: boolean = false,
  ): Tensor {
    return this.wrappedOp((a: Tensor_wasm) =>
      a.gemm(rhs._cloneInner(), bias ? bias._cloneInner() : null, transLhs, transRhs, transOut),
    );
  }

  affine(mul: number, add: number): Tensor {
    return this.wrappedOp((a: Tensor_wasm) => a.affine(mul, add));
  }

  sum(dim?: DimsType, keepdim?: boolean): Tensor {
    return this.wrappedOp((a: Tensor_wasm) => a.sum(dim as Int32Array, keepdim ?? false));
  }

  mean(dim?: DimsType, keepdim?: boolean): Tensor {
    return this.wrappedOp((a: Tensor_wasm) => a.mean(dim as Int32Array, keepdim ?? false));
  }

  var(dim?: DimsType, keepdim?: boolean): Tensor {
    return this.wrappedOp((a: Tensor_wasm) => a.var(dim, keepdim ?? false));
  }

  max(dim: number, keepdim?: boolean): Tensor {
    return this.wrappedOp((a: Tensor_wasm) => a.max(dim, keepdim ?? false));
  }

  min(dim: number, keepdim?: boolean): Tensor {
    return this.wrappedOp((a: Tensor_wasm) => a.min(dim, keepdim ?? false));
  }

  argmax(dim: number, keepdim?: boolean): Tensor {
    return this.wrappedOp((a: Tensor_wasm) => a.argmax(dim, keepdim ?? false));
  }

  argmin(dim: number, keepdim?: boolean): Tensor {
    return this.wrappedOp((a: Tensor_wasm) => a.argmin(dim, keepdim ?? false));
  }

  norm(ord?: string | number | null, dim?: DimsType, keepdim?: boolean): Tensor {
    return this.wrappedOp((a: Tensor_wasm) => a.norm(ord, dim, keepdim ?? false));
  }

  flatten(startDim?: number, endDim?: number): Tensor {
    return this.wrappedOp((a: Tensor_wasm) => a.flatten(startDim, endDim));
  }

  // TODO: Replace with more expressive indexer .i
  slice(ranges: number[][]): Tensor {
    return this.wrappedOp((a: Tensor_wasm) => a.slice(ranges));
  }

  view(...shape: number[] | [number[]]): Tensor {
    if (shape.length === 1 && Array.isArray(shape[0])) {
      return this.wrappedOp((a: Tensor_wasm) => a.view(shape[0] as number[]));
    } else {
      return this.wrappedOp((a: Tensor_wasm) => a.view(shape as number[]));
    }
  }

  unsqueeze(dim: number): Tensor {
    return this.wrappedOp((a: Tensor_wasm) => a.unsqueeze(dim));
  }

  squeeze(dims?: DimsType): Tensor {
    return this.wrappedOp((a: Tensor_wasm) => a.squeeze(dims));
  }

  permute(dims: DimsType): Tensor {
    return this.wrappedOp((a: Tensor_wasm) => a.permute(dims));
  }

  transpose(dim0: number, dim1: number): Tensor {
    return this.wrappedOp((a: Tensor_wasm) => a.transpose(dim0, dim1));
  }

  t(): Tensor {
    return this.wrappedOp((a: Tensor_wasm) => a.t());
  }

  get T(): Tensor {
    return this.wrappedOp((a: Tensor_wasm) => a.T);
  }

  get mT(): Tensor {
    return this.wrappedOp((a: Tensor_wasm) => a.mT);
  }

  cache(source: Tensor, dim: number, offset: number): Tensor {
    return this.wrappedOp((a: Tensor_wasm) => a.cache(source._cloneInner(), dim, offset));
  }

  broadcastLeft(leftShape: ShapeType): Tensor {
    return this.wrappedOp((a: Tensor_wasm) => a.broadcastLeft(leftShape));
  }

  broadcastTo(shape: ShapeType): Tensor {
    return this.wrappedOp((a: Tensor_wasm) => a.broadcastTo(shape));
  }

  indexSelect(indices: Tensor, dim: number): Tensor {
    return this.wrappedOp((a: Tensor_wasm) => a.indexSelect(indices._cloneInner(), dim));
  }

  indexWrite(src: Tensor, writeStart: DimsType): Tensor {
    return this.wrappedOp((a: Tensor_wasm) => a.indexWrite(src._cloneInner(), writeStart));
  }

  where(condition: Tensor, onFalse: TensorOrScalar): Tensor {
    return this.wrappedOp((a: Tensor_wasm) =>
      a.where(condition._cloneInner(), onFalse instanceof Tensor ? onFalse._cloneInner() : onFalse),
    );
  }

  scatterAdd(indices: Tensor, source: Tensor, dim: number): Tensor {
    return this.wrappedOp((a: Tensor_wasm) =>
      a.scatterAdd(indices._cloneInner(), source._cloneInner(), dim),
    );
  }

  indexAdd_(indices: Tensor, source: Tensor, dim: number): Tensor {
    return this.wrappedOp((a: Tensor_wasm) =>
      a.indexAdd_(indices._cloneInner(), source._cloneInner(), dim),
    );
  }

  gather(indices: Tensor, dim: number): Tensor {
    return this.wrappedOp((a: Tensor_wasm) => a.gather(indices._cloneInner(), dim));
  }

  triu(k?: number): Tensor {
    return this.wrappedOp((a: Tensor_wasm) => a.triu(k));
  }

  triu_(k?: number): Tensor {
    return this.wrappedOp((a: Tensor_wasm) => a.triu_(k));
  }

  tril(k?: number): Tensor {
    return this.wrappedOp((a: Tensor_wasm) => a.tril(k));
  }

  tril_(k?: number): Tensor {
    return this.wrappedOp((a: Tensor_wasm) => a.tril_(k));
  }

  lerp(end: Tensor, weight: TensorOrScalar): Tensor {
    return this.wrappedOp((a: Tensor_wasm) =>
      a.lerp(end._cloneInner(), weight instanceof Tensor ? weight._cloneInner() : weight),
    );
  }

  lerp_(end: Tensor, weight: TensorOrScalar): Tensor {
    return this.wrappedOp((a: Tensor_wasm) =>
      a.lerp_(end._cloneInner(), weight instanceof Tensor ? weight._cloneInner() : weight),
    );
  }

  bernoulli(): Tensor {
    return this.wrappedOp((a: Tensor_wasm) => a.bernoulli());
  }

  bernoulli_(): Tensor {
    return this.wrappedOp((a: Tensor_wasm) => a.bernoulli_());
  }

  zero_(): Tensor {
    return this.wrappedOp((a: Tensor_wasm) => a.zero_());
  }

  // We skip onesLike and zerosLike because they're not defined as members in
  // the Tensor class in PyTorch

  // Tensor properties and operations
  isContiguous(): boolean {
    return this._cloneInner().isContiguous();
  }

  contiguous(): Tensor {
    return this.wrappedOp((a: Tensor_wasm) => a.contiguous());
  }

  detach(): Tensor {
    return this.wrappedOp((a: Tensor_wasm) => a.detach());
  }

  detach_(): Tensor {
    return this.wrappedOp((a: Tensor_wasm) => a.detach_());
  }

  requiresGrad_(requiresGrad: boolean = true): Tensor {
    return this.wrappedOp((a: Tensor_wasm) => a.requiresGrad_(requiresGrad));
  }

  async item(dtype?: DType): Promise<number> {
    return this._cloneInner().item(dtype);
  }

  async toVec(dtype?: DType): Promise<Float32Array | Int32Array | Uint32Array> {
    return this._cloneInner().toVec(dtype);
  }

  async to(device: string): Promise<Tensor> {
    return Tensor._wrap(await this._cloneInner().to(device));
  }

  hasNaN(dtype?: DType): boolean {
    return this._cloneInner().hasNaN(dtype);
  }

  // Getter methods for tensor properties
  get id(): number {
    return this._cloneInner().id;
  }

  dim(): number {
    return this._cloneInner().dim();
  }

  get ndim(): number {
    return this._cloneInner().ndim;
  }

  get dtype(): DType {
    return this._cloneInner().dtype;
  }

  size(): number[];
  size(dim: number): number;
  size(dim?: number): number | number[] {
    return this._cloneInner().size(dim);
  }

  stride(): number[];
  stride(dim: number): number;
  stride(dim?: number): number | number[] {
    return this._cloneInner().stride(dim);
  }

  get shape(): number[] {
    return Array.from(this._cloneInner().shape());
  }

  get device(): string {
    return this._cloneInner().device();
  }

  resolved(): boolean {
    return this._cloneInner().resolved();
  }

  op(): OpDescription {
    return this._cloneInner().op();
  }

  srcIds(): number[] {
    return Array.from(this._cloneInner().srcIds());
  }

  storageId(): number | undefined {
    return this._cloneInner().storageId();
  }

  isScalar(): boolean {
    return this._cloneInner().isScalar();
  }

  get requiresGrad(): boolean {
    return this._cloneInner().requiresGrad;
  }

  debugTensor(): Tensor {
    return Tensor._wrap(this._cloneInner().debugTensor());
  }

  get grad(): Tensor | undefined {
    const grad = this._cloneInner().grad;
    return grad ? Tensor._wrap(grad) : undefined;
  }

  set grad(grad: Tensor | undefined | null) {
    this._cloneInner().grad = grad ? grad._cloneInner() : undefined;
  }

  backward(): void {
    this._cloneInner().backward();
  }

  get nbytes(): number {
    return this._cloneInner().nbytes;
  }

  private _cloneInner(): Tensor_wasm {
    return this.innerTensor._clone();
  }

  invalidate(): void {
    this.innerTensor.invalidate();
  }

  static {
    for (const op of binaryOps) {
      (Tensor.prototype as unknown as Record<string, unknown>)[op] = Tensor.makeBinaryOp(op);
    }

    for (const op of binaryTensorOnlyOps) {
      (Tensor.prototype as unknown as Record<string, unknown>)[op] =
        Tensor.makeBinaryOpTensorOnly(op);
    }

    for (const op of ternaryOps) {
      (Tensor.prototype as unknown as Record<string, unknown>)[op] = Tensor.makeTernaryOp(op);
    }

    for (const op of cmpOps) {
      (Tensor.prototype as unknown as Record<string, unknown>)[op] = Tensor.makeCmpOp(op);
    }

    for (const op of unaryOps) {
      (Tensor.prototype as unknown as Record<string, unknown>)[op] = Tensor.makeUnaryOp(op);
    }
  }
}

// Make dynamically-attached ops visible to TypeScript via interface merging.
type BinaryMethods = { [K in (typeof binaryOps)[number]]: (other: TensorOrScalar) => Tensor };
type BinaryTensorOnlyMethods = {
  [K in (typeof binaryTensorOnlyOps)[number]]: (other: Tensor) => Tensor;
};
type UnaryMethods = { [K in (typeof unaryOps)[number]]: () => Tensor };
type CmpMethods = { [K in (typeof cmpOps)[number]]: (other: TensorOrScalar) => Tensor };
type TernaryMethods = {
  [K in (typeof ternaryOps)[number]]: (tensor1: Tensor, tensor2: Tensor, value: number) => Tensor;
};

export interface Tensor
  extends BinaryMethods,
    BinaryTensorOnlyMethods,
    UnaryMethods,
    CmpMethods,
    TernaryMethods {}
