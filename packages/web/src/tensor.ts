import { Module } from "./nn/module";
import { ScopeItem, tensorScopeStack } from "./nn/tracking";
import { BinaryOpInput, DimsType, ShapeType } from "./types";
import { DType, Tensor_wasm } from "./wasm";

export type OpDescription = {
  name: string;
  // TODO(vinhowe): Fix this to be like the actual type
  fields: Record<string, unknown>;
};

export type TensorHookOptions = {
  dependency: boolean;
};

export type TensorHook = (tensor: Tensor, options: TensorHookOptions) => Tensor | undefined;

export class Tensor {
  scope: ScopeItem[] | undefined;
  nameOnParent: string | undefined;
  constructor(private readonly innerTensor: Tensor_wasm) {
    this.scope = [...tensorScopeStack];
  }

  // @internal
  get parentModule(): Module<unknown, unknown> | undefined {
    return this.scope?.findLast((item) => item.type === "module")?.module;
  }

  private static runHooks<T extends Tensor | Tensor[]>(
    tensorOrTensors: T,
    options?: TensorHookOptions,
  ): T {
    options = options ?? { dependency: false };
    const tensors = Array.isArray(tensorOrTensors) ? tensorOrTensors : [tensorOrTensors];
    for (let i = 0; i < tensors.length; i++) {
      const tensor = tensors[i];
      const hooks = tensor.parentModule?._tensorHooks;
      if (hooks) {
        for (const hook of hooks.values()) {
          const result = hook(tensor, options);
          if (result) {
            tensors[i] = result;
          }
        }
      }
    }
    return tensors.length === 1 ? tensors[0] : (tensors as T);
  }

  // Helper method to access inner tensor for internal use
  static _unwrap(tensor: Tensor): Tensor_wasm {
    return tensor?.innerTensor?._clone();
  }

  // Creates a new Tensor from a Tensor_wasm
  static _wrap(tensor: Tensor_wasm): Tensor {
    // TODO: This might be more cloning than necessary
    return new Tensor(tensor._clone());
  }

  // Create a scalar tensor (empty shape) with the given value
  static scalar(value: number, dtype?: DType): Tensor {
    return Tensor._wrap(Tensor_wasm.full([], value, dtype || null, null, false));
  }

  private wrappedOp(op: (a: Tensor_wasm) => Tensor_wasm): Tensor {
    Tensor.runHooks(this, { dependency: true });
    const result = Tensor._wrap(op(this._cloneInner()));
    return Tensor.runHooks(result);
  }

  // Define binary operations using a constructor
  private makeBinaryOp(name: keyof Tensor_wasm): (other: BinaryOpInput) => Tensor {
    return (other: BinaryOpInput): Tensor => {
      const otherValue = other instanceof Tensor ? other._cloneInner() : other;
      if (otherValue instanceof Tensor) {
        Tensor.runHooks(otherValue, { dependency: true });
      }
      return this.wrappedOp((a) => (a[name] as (b: Tensor_wasm) => Tensor_wasm)(otherValue));
    };
  }

  add = this.makeBinaryOp("add");
  add_ = this.makeBinaryOp("add_");
  sub = this.makeBinaryOp("sub");
  sub_ = this.makeBinaryOp("sub_");
  mul = this.makeBinaryOp("mul");
  mul_ = this.makeBinaryOp("mul_");
  div = this.makeBinaryOp("div");
  div_ = this.makeBinaryOp("div_");
  minimum = this.makeBinaryOp("minimum");
  minimum_ = this.makeBinaryOp("minimum_");
  maximum = this.makeBinaryOp("maximum");
  maximum_ = this.makeBinaryOp("maximum_");

  private makeTernaryOp(
    name: keyof Tensor_wasm,
  ): (tensor1: Tensor, tensor2: Tensor, value: number) => Tensor {
    return (tensor1: Tensor, tensor2: Tensor, value: number): Tensor => {
      Tensor.runHooks([tensor1, tensor2], { dependency: true });
      return this.wrappedOp((a) =>
        (a[name] as (b: Tensor_wasm, c: Tensor_wasm, d: number) => Tensor_wasm)(
          tensor1._cloneInner(),
          tensor2._cloneInner(),
          value,
        ),
      );
    };
  }

  addcdiv = this.makeTernaryOp("addcdiv");
  addcdiv_ = this.makeTernaryOp("addcdiv_");
  addcmul = this.makeTernaryOp("addcmul");
  addcmul_ = this.makeTernaryOp("addcmul_");

  private makeCmpOp(name: keyof Tensor_wasm): (other: Tensor) => Tensor {
    return (other: Tensor): Tensor => {
      Tensor.runHooks(other, { dependency: true });
      return this.wrappedOp((a) =>
        (a[name] as (b: Tensor_wasm) => Tensor_wasm)(other._cloneInner()),
      );
    };
  }

  eq = this.makeCmpOp("eq");
  eq_ = this.makeCmpOp("eq_");
  ne = this.makeCmpOp("ne");
  ne_ = this.makeCmpOp("ne_");
  gt = this.makeCmpOp("gt");
  gt_ = this.makeCmpOp("gt_");
  ge = this.makeCmpOp("ge");
  ge_ = this.makeCmpOp("ge_");
  lt = this.makeCmpOp("lt");
  lt_ = this.makeCmpOp("lt_");
  le = this.makeCmpOp("le");
  le_ = this.makeCmpOp("le_");

  private makeUnaryOp(name: keyof Tensor_wasm): () => Tensor {
    return (): Tensor => this.wrappedOp((a) => (a[name] as () => Tensor_wasm)());
  }

  gelu = this.makeUnaryOp("gelu");
  gelu_ = this.makeUnaryOp("gelu_");
  tanh = this.makeUnaryOp("tanh");
  tanh_ = this.makeUnaryOp("tanh_");
  exp = this.makeUnaryOp("exp");
  exp_ = this.makeUnaryOp("exp_");
  log = this.makeUnaryOp("log");
  log_ = this.makeUnaryOp("log_");
  sin = this.makeUnaryOp("sin");
  sin_ = this.makeUnaryOp("sin_");
  cos = this.makeUnaryOp("cos");
  cos_ = this.makeUnaryOp("cos_");
  abs = this.makeUnaryOp("abs");
  abs_ = this.makeUnaryOp("abs_");
  sqrt = this.makeUnaryOp("sqrt");
  sqrt_ = this.makeUnaryOp("sqrt_");
  relu = this.makeUnaryOp("relu");
  relu_ = this.makeUnaryOp("relu_");
  relu2 = this.makeUnaryOp("relu2");
  relu2_ = this.makeUnaryOp("relu2_");
  floor = this.makeUnaryOp("floor");
  floor_ = this.makeUnaryOp("floor_");
  ceil = this.makeUnaryOp("ceil");
  ceil_ = this.makeUnaryOp("ceil_");
  neg = this.makeUnaryOp("neg");
  neg_ = this.makeUnaryOp("neg_");
  sigmoid = this.makeUnaryOp("sigmoid");
  sigmoid_ = this.makeUnaryOp("sigmoid_");
  swiglu = this.makeUnaryOp("swiglu");
  swiglu_ = this.makeUnaryOp("swiglu_");
  silu = this.makeUnaryOp("silu");
  silu_ = this.makeUnaryOp("silu_");
  square = this.makeUnaryOp("square");
  square_ = this.makeUnaryOp("square_");
  recip = this.makeUnaryOp("recip");
  recip_ = this.makeUnaryOp("recip_");

  float = this.makeUnaryOp("float");
  half = this.makeUnaryOp("half");

  cast(dstDtype?: DType): Tensor {
    return this.wrappedOp((a) => a.cast(dstDtype?._clone()));
  }

  groupNorm(numGroups: number, weight: Tensor, bias?: Tensor, eps: number = 1e-5): Tensor {
    Tensor.runHooks(weight, { dependency: true });
    if (bias) {
      Tensor.runHooks(bias, { dependency: true });
    }
    return this.wrappedOp((a) =>
      a.groupNorm(numGroups, weight?._cloneInner(), bias ? bias._cloneInner() : null, eps),
    );
  }

  layerNorm(weight: Tensor, bias?: Tensor, eps: number = 1e-5): Tensor {
    Tensor.runHooks(weight, { dependency: true });
    if (bias) {
      Tensor.runHooks(bias, { dependency: true });
    }
    return this.wrappedOp((a) =>
      a.layerNorm(weight?._cloneInner(), bias ? bias._cloneInner() : null, eps),
    );
  }

  rmsNorm(weight: Tensor, eps: number = 1e-5): Tensor {
    Tensor.runHooks(weight, { dependency: true });
    return this.wrappedOp((a) => a.rmsNorm(weight?._cloneInner(), eps));
  }

  conv1d(weight: Tensor, bias?: Tensor, stride: number = 1, padding: number = 0): Tensor {
    Tensor.runHooks(weight, { dependency: true });
    if (bias) {
      Tensor.runHooks(bias, { dependency: true });
    }
    return this.wrappedOp((a) =>
      a.conv1d(weight?._cloneInner(), bias ? bias._cloneInner() : null, stride, padding),
    );
  }

  softmax(dim: number): Tensor {
    return this.wrappedOp((a) => a.softmax(dim));
  }

  rope(dim: number, base: number, offset: number): Tensor {
    return this.wrappedOp((a) => a.rope(dim, base, offset));
  }

  alibi(maxBias: number): Tensor {
    return this.wrappedOp((a) => a.alibi(maxBias));
  }

  matmul(rhs: Tensor, transLhs: boolean = false, transRhs: boolean = false): Tensor {
    Tensor.runHooks(rhs, { dependency: true });
    return this.wrappedOp((a) => a.matmul(rhs._cloneInner(), transLhs, transRhs));
  }

  gemm(
    rhs: Tensor,
    bias?: Tensor,
    transLhs: boolean = false,
    transRhs: boolean = false,
    transOut: boolean = false,
  ): Tensor {
    Tensor.runHooks(rhs, { dependency: true });
    if (bias) {
      Tensor.runHooks(bias, { dependency: true });
    }
    return this.wrappedOp((a) =>
      a.gemm(rhs._cloneInner(), bias ? bias._cloneInner() : null, transLhs, transRhs, transOut),
    );
  }

  affine(mul: number, add: number): Tensor {
    return this.wrappedOp((a) => a.affine(mul, add));
  }

  pow(e: number): Tensor {
    return this.wrappedOp((a) => a.pow(e));
  }

  pow_(e: number): Tensor {
    return this.wrappedOp((a) => a.pow_(e));
  }

  sum(dim?: DimsType, keepdim?: boolean): Tensor {
    return this.wrappedOp((a) => a.sum(dim as Int32Array, keepdim ?? false));
  }

  mean(dim?: DimsType, keepdim?: boolean): Tensor {
    return this.wrappedOp((a) => a.mean(dim as Int32Array, keepdim ?? false));
  }

  var(dim?: DimsType, keepdim?: boolean): Tensor {
    return this.wrappedOp((a) => a.var(dim, keepdim ?? false));
  }

  max(dim: number, keepdim?: boolean): Tensor {
    return this.wrappedOp((a) => a.max(dim, keepdim ?? false));
  }

  min(dim: number, keepdim?: boolean): Tensor {
    return this.wrappedOp((a) => a.min(dim, keepdim ?? false));
  }

  argmax(dim: number, keepdim?: boolean): Tensor {
    return this.wrappedOp((a) => a.argmax(dim, keepdim ?? false));
  }

  argmin(dim: number, keepdim?: boolean): Tensor {
    return this.wrappedOp((a) => a.argmin(dim, keepdim ?? false));
  }

  norm(): Tensor {
    return this.wrappedOp((a) => a.norm());
  }

  flatten(startDim?: number, endDim?: number): Tensor {
    return this.wrappedOp((a) => a.flatten(startDim, endDim));
  }

  // TODO: Replace with more expressive indexer .i
  slice(ranges: number[][]): Tensor {
    return this.wrappedOp((a) => a.slice(ranges));
  }

  view(...shape: number[] | [number[]]): Tensor {
    if (shape.length === 1 && Array.isArray(shape[0])) {
      return this.wrappedOp((a) => a.view(shape[0] as number[]));
    } else {
      return this.wrappedOp((a) => a.view(shape as number[]));
    }
  }

  unsqueeze(dim: number): Tensor {
    return this.wrappedOp((a) => a.unsqueeze(dim));
  }

  squeeze(dims?: DimsType): Tensor {
    return this.wrappedOp((a) => a.squeeze(dims));
  }

  permute(dims: DimsType): Tensor {
    return this.wrappedOp((a) => a.permute(dims));
  }

  transpose(dim0: number, dim1: number): Tensor {
    return this.wrappedOp((a) => a.transpose(dim0, dim1));
  }

  t(): Tensor {
    return this.wrappedOp((a) => a.t());
  }

  cache(source: Tensor, dim: number, offset: number): Tensor {
    Tensor.runHooks(source, { dependency: true });
    return this.wrappedOp((a) => a.cache(source._cloneInner(), dim, offset));
  }

  broadcastLeft(leftShape: ShapeType): Tensor {
    return this.wrappedOp((a) => a.broadcastLeft(leftShape));
  }

  broadcastTo(shape: ShapeType): Tensor {
    return this.wrappedOp((a) => a.broadcastTo(shape));
  }

  indexSelect(indices: Tensor, dim: number): Tensor {
    Tensor.runHooks(indices, { dependency: true });
    return this.wrappedOp((a) => a.indexSelect(indices._cloneInner(), dim));
  }

  indexWrite(src: Tensor, writeStart: DimsType): Tensor {
    Tensor.runHooks(src, { dependency: true });
    return this.wrappedOp((a) => a.indexWrite(src._cloneInner(), writeStart));
  }

  where(condition: Tensor, onFalse: Tensor): Tensor {
    Tensor.runHooks(condition, { dependency: true });
    Tensor.runHooks(onFalse, { dependency: true });
    return this.wrappedOp((a) => a.whereCond(condition._cloneInner(), onFalse._cloneInner()));
  }

  scatterAdd(indices: Tensor, source: Tensor, dim: number): Tensor {
    Tensor.runHooks(indices, { dependency: true });
    Tensor.runHooks(source, { dependency: true });
    return this.wrappedOp((a) => a.scatterAdd(indices._cloneInner(), source._cloneInner(), dim));
  }

  indexAdd_(indices: Tensor, source: Tensor, dim: number): Tensor {
    Tensor.runHooks(indices, { dependency: true });
    Tensor.runHooks(source, { dependency: true });
    return this.wrappedOp((a) => a.indexAdd_(indices._cloneInner(), source._cloneInner(), dim));
  }

  gather(indices: Tensor, dim: number): Tensor {
    Tensor.runHooks(indices, { dependency: true });
    return this.wrappedOp((a) => a.gather(indices._cloneInner(), dim));
  }

  triu(k?: number): Tensor {
    return this.wrappedOp((a) => a.triu(k));
  }

  triu_(k?: number): Tensor {
    return this.wrappedOp((a) => a.triu_(k));
  }

  tril(k?: number): Tensor {
    return this.wrappedOp((a) => a.tril(k));
  }

  tril_(k?: number): Tensor {
    return this.wrappedOp((a) => a.tril_(k));
  }

  bernoulli(): Tensor {
    return this.wrappedOp((a) => a.bernoulli());
  }

  bernoulli_(): Tensor {
    return this.wrappedOp((a) => a.bernoulli_());
  }

  zero_(): Tensor {
    return this.wrappedOp((a) => a.zero_());
  }

  // We skip onesLike and zerosLike because they're not defined as members in
  // the Tensor class in PyTorch

  // Tensor properties and operations
  isContiguous(): boolean {
    return this._cloneInner().isContiguous();
  }

  contiguous(): Tensor {
    return this.wrappedOp((a) => a.contiguous());
  }

  detach(): Tensor {
    return this.wrappedOp((a) => a.detach());
  }

  detach_(): Tensor {
    return this.wrappedOp((a) => a.detach_());
  }

  requiresGrad_(requiresGrad: boolean = true): Tensor {
    return this.wrappedOp((a) => a.requiresGrad_(requiresGrad));
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
}
