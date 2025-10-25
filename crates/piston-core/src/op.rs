#[cfg(feature = "debug")]
use crate::MIN_STORAGE_BUFFER_SIZE;
#[cfg(feature = "debug")]
use crate::gpu::BufferUsagesExt;
use crate::gpu::{
    BindGroupLayoutDescriptor, ComputePipelineDescriptor, CpuUniform, PipelineLayoutDescriptor,
    PoolError, WgpuDevice,
};
use crate::{
    CompiledOp, DType, HashMap, InvariantError, Kernel, KernelBuildError, KernelMetadata,
    KernelModuleDesc, OpTensor, RVec, Shape, StorageView, TensorId, TensorTypeOrScalarEnum,
    WgslFragment, WorkgroupSize, Workload, ops::*, rvec,
};
#[cfg(feature = "debug")]
use slotmap::Key;
#[cfg(feature = "debug")]
use smallvec::SmallVec;
use std::borrow::Cow;
#[cfg(feature = "debug")]
use std::cmp::max;
use std::collections::BTreeMap;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::ops::Range;
#[cfg(feature = "debug")]
use std::sync::Arc;

#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum LazyOp {
    Const,
    Matmul(Matmul),
    Conv(Conv),
    Binary(Binary),
    Unary(Unary),
    Reindex(Reindex),
    Concat(Concat),
    Norm(NormOp),
    Affine(Affine),
    Cmp(Cmp),
    Powf(Powf),
    Cast(Cast),
    WhereCond(WhereCond),
    Reduce(Reduce),
    Gather(Gather),
    Ternary(Ternary),
    Lerp(Lerp),
    // ---- Everything below this line shouldn't exist ----
    FillPointwise(FillPointwise),
    Bernoulli(Bernoulli),
    RoPE(RoPE),
    Alibi(Alibi),
    Softmax(Softmax),
    View(View),             //Should be general class, metadata modification
    Select(IndexSelect),    //Can probably be Reindex
    IndexWrite(IndexWrite), //Above 2 should be merged
    Cache(Cache),           //Should be a general class
    IndexAdd(IndexAdd),
    ScatterAdd(ScatterAdd),
    Trilu(TriluOp),
    Eye(Eye),
    Arange(Arange),
    Copy(TensorCopy),
    Detach(Box<LazyOp>), //Because the entire graph is lazy, you can't actually detach something without computing the graph in parts
}

impl LazyOp {
    pub fn name(&self) -> &str {
        match self {
            LazyOp::Binary(b) => b.name(),
            LazyOp::Cast(c) => c.name(),
            LazyOp::Matmul(m) => m.name(),
            LazyOp::Softmax(s) => s.name(),
            LazyOp::Unary(u) => u.name(),
            LazyOp::Reindex(r) => r.name(),
            LazyOp::Concat(c) => c.name(),
            LazyOp::Norm(n) => n.name(),
            LazyOp::Affine(a) => a.name(),
            LazyOp::Cmp(c) => c.name(),
            LazyOp::Powf(p) => p.name(),
            LazyOp::WhereCond(w) => w.name(),
            LazyOp::Reduce(s) => s.name(),
            LazyOp::Gather(g) => g.name(),
            LazyOp::Conv(c) => c.name(),
            LazyOp::Ternary(t) => t.name(),
            LazyOp::Lerp(l) => l.name(),
            LazyOp::Select(s) => s.name(),
            LazyOp::IndexWrite(iw) => iw.name(),
            LazyOp::IndexAdd(ia) => ia.name(),
            LazyOp::ScatterAdd(sa) => sa.name(),
            LazyOp::Trilu(t) => t.name(),
            LazyOp::Eye(e) => e.name(),
            LazyOp::FillPointwise(f) => f.name(),
            LazyOp::Bernoulli(b) => b.name(),
            LazyOp::Arange(a) => a.name(),
            LazyOp::RoPE(r) => r.name(),
            LazyOp::Alibi(a) => a.name(),
            LazyOp::Cache(c) => c.name(),
            LazyOp::View(v) => v.name(),
            LazyOp::Copy(c) => c.name(),
            LazyOp::Const => "Const",
            LazyOp::Detach(d) => Box::leak(format!("Detach<{}>", d.name()).into_boxed_str()),
        }
    }

    #[inline(always)]
    pub fn srcs(&self) -> RVec<&OpTensor> {
        match self {
            LazyOp::Binary(b) => b.srcs(),
            LazyOp::Cast(c) => c.srcs(),
            LazyOp::Matmul(m) => m.srcs(),
            LazyOp::RoPE(r) => r.srcs(),
            LazyOp::Alibi(a) => a.srcs(),
            LazyOp::Softmax(s) => s.srcs(),
            LazyOp::Unary(u) => u.srcs(),
            LazyOp::Reindex(r) => r.srcs(),
            LazyOp::Concat(c) => c.srcs(),
            LazyOp::Norm(n) => n.srcs(),
            LazyOp::Affine(a) => a.srcs(),
            LazyOp::Cmp(c) => c.srcs(),
            LazyOp::Powf(p) => p.srcs(),
            LazyOp::WhereCond(w) => w.srcs(),
            LazyOp::Reduce(s) => s.srcs(),
            LazyOp::Gather(g) => g.srcs(),
            LazyOp::Conv(c) => c.srcs(),
            LazyOp::Ternary(t) => t.srcs(),
            LazyOp::Lerp(l) => l.srcs(),
            LazyOp::Select(s) => s.srcs(),
            LazyOp::IndexWrite(iw) => iw.srcs(),
            LazyOp::IndexAdd(ia) => ia.srcs(),
            LazyOp::ScatterAdd(sa) => sa.srcs(),
            LazyOp::Trilu(t) => t.srcs(),
            LazyOp::Eye(e) => e.srcs(),
            LazyOp::Cache(c) => c.srcs(),
            LazyOp::Detach(d) => d.srcs(),
            LazyOp::View(v) => v.srcs(),
            LazyOp::Copy(c) => c.srcs(),
            LazyOp::Bernoulli(b) => b.srcs(),
            LazyOp::FillPointwise(_) | LazyOp::Arange(_) | LazyOp::Const => {
                rvec![]
            } //end of the line kid
        }
    }

    pub fn supports_inplace(&self) -> bool {
        match self {
            LazyOp::Binary(b) => b.supports_inplace(),
            LazyOp::Cast(c) => c.supports_inplace(),
            LazyOp::Matmul(m) => m.supports_inplace(),
            LazyOp::RoPE(r) => r.supports_inplace(),
            LazyOp::Alibi(a) => a.supports_inplace(),
            LazyOp::Softmax(s) => s.supports_inplace(),
            LazyOp::Unary(u) => u.supports_inplace(),
            LazyOp::Reindex(r) => r.supports_inplace(),
            LazyOp::Concat(c) => c.supports_inplace(),
            LazyOp::Norm(n) => n.supports_inplace(),
            LazyOp::Affine(a) => a.supports_inplace(),
            LazyOp::Cmp(c) => c.supports_inplace(),
            LazyOp::Powf(p) => p.supports_inplace(),
            LazyOp::WhereCond(w) => w.supports_inplace(),
            LazyOp::Reduce(s) => s.supports_inplace(),
            LazyOp::Gather(g) => g.supports_inplace(),
            LazyOp::Conv(c) => c.supports_inplace(),
            LazyOp::Ternary(t) => t.supports_inplace(),
            LazyOp::Lerp(l) => l.supports_inplace(),
            LazyOp::Select(s) => s.supports_inplace(),
            LazyOp::IndexWrite(iw) => iw.supports_inplace(),
            LazyOp::IndexAdd(ia) => ia.supports_inplace(),
            LazyOp::ScatterAdd(sa) => sa.supports_inplace(),
            LazyOp::Trilu(t) => t.supports_inplace(),
            LazyOp::Eye(e) => e.supports_inplace(),
            LazyOp::FillPointwise(f) => f.supports_inplace(),
            LazyOp::Bernoulli(b) => b.supports_inplace(),
            LazyOp::Arange(a) => a.supports_inplace(),
            LazyOp::Cache(c) => c.supports_inplace(),
            LazyOp::View(_v) => true,
            LazyOp::Const => false,
            LazyOp::Detach(d) => d.supports_inplace(),
            LazyOp::Copy(c) => c.supports_inplace(),
        }
    }

    pub fn is_const(&self) -> bool {
        matches!(self, LazyOp::Const)
    }

    // We have to keep this distinct from is_const, because we use is_const to determine if
    // something will be excluded from the computation graph. No good for these
    // shader-generated parameters.
    pub fn can_be_parameter(&self) -> bool {
        matches!(
            self,
            LazyOp::Const | LazyOp::FillPointwise(_) | LazyOp::Eye(_) | LazyOp::Arange(_)
        )
    }

    #[track_caller]
    pub fn check_invariants(&self) {
        match self {
            LazyOp::Binary(b) => b.check_invariants(),
            LazyOp::Cast(c) => c.check_invariants(),
            LazyOp::Matmul(m) => m.check_invariants(),
            LazyOp::RoPE(r) => r.check_invariants(),
            LazyOp::Alibi(a) => a.check_invariants(),
            LazyOp::Softmax(s) => s.check_invariants(),
            LazyOp::Unary(u) => u.check_invariants(),
            LazyOp::Reindex(r) => match r {
                Reindex::Permute(p) => p.check_invariants(),
                Reindex::Slice(s) => s.check_invariants(),
                Reindex::Broadcast(b) => b.check_invariants(),
                Reindex::Flip(f) => f.check_invariants(),
            },
            LazyOp::Concat(c) => c.check_invariants(),
            LazyOp::Norm(n) => n.check_invariants(),
            LazyOp::Affine(a) => a.check_invariants(),
            LazyOp::Cmp(c) => c.check_invariants(),
            LazyOp::Powf(p) => p.check_invariants(),
            LazyOp::WhereCond(w) => w.check_invariants(),
            LazyOp::Reduce(s) => s.check_invariants(),
            LazyOp::Gather(g) => g.check_invariants(),
            LazyOp::Conv(c) => c.check_invariants(),
            LazyOp::Ternary(t) => t.check_invariants(),
            LazyOp::Lerp(l) => l.check_invariants(),
            LazyOp::Select(s) => s.check_invariants(),
            LazyOp::IndexWrite(iw) => iw.check_invariants(),
            LazyOp::IndexAdd(ia) => ia.check_invariants(),
            LazyOp::ScatterAdd(sa) => sa.check_invariants(),
            LazyOp::Trilu(t) => t.check_invariants(),
            LazyOp::Eye(e) => e.check_invariants(),
            LazyOp::FillPointwise(f) => f.check_invariants(),
            LazyOp::Bernoulli(b) => b.check_invariants(),
            LazyOp::Arange(a) => a.check_invariants(),
            LazyOp::Cache(c) => c.check_invariants(),
            LazyOp::View(v) => v.check_invariants(),
            LazyOp::Const => {}
            LazyOp::Detach(d) => d.check_invariants(),
            LazyOp::Copy(c) => c.check_invariants(),
        }
    }

    pub fn ir(&self) -> Ir {
        match self {
            LazyOp::Binary(b) => b.ir(),
            LazyOp::Cast(c) => c.ir(),
            LazyOp::Matmul(m) => m.ir(),
            LazyOp::RoPE(r) => r.ir(),
            LazyOp::Alibi(a) => a.ir(),
            LazyOp::Softmax(s) => s.ir(),
            LazyOp::Unary(u) => u.ir(),
            LazyOp::Reindex(r) => r.ir(),
            LazyOp::Concat(c) => c.ir(),
            LazyOp::Norm(n) => n.ir(),
            LazyOp::Affine(a) => a.ir(),
            LazyOp::Cmp(c) => c.ir(),
            LazyOp::Powf(p) => p.ir(),
            LazyOp::WhereCond(w) => w.ir(),
            LazyOp::Reduce(s) => s.ir(),
            LazyOp::Gather(g) => g.ir(),
            LazyOp::Conv(c) => c.ir(),
            LazyOp::Ternary(t) => t.ir(),
            LazyOp::Lerp(l) => l.ir(),
            LazyOp::Select(s) => s.ir(),
            LazyOp::IndexWrite(iw) => iw.ir(),
            LazyOp::IndexAdd(ia) => ia.ir(),
            LazyOp::ScatterAdd(sa) => sa.ir(),
            LazyOp::Trilu(t) => t.ir(),
            LazyOp::Eye(e) => e.ir(),
            LazyOp::FillPointwise(f) => f.ir(),
            LazyOp::Bernoulli(b) => b.ir(),
            LazyOp::Arange(a) => a.ir(),
            LazyOp::Cache(c) => c.ir(),
            LazyOp::View(v) => v.ir(),
            LazyOp::Detach(d) => {
                let detached = d.ir();
                let mut ir = Ir::new("Detach");
                ir.with_field("op", detached);
                ir
            }
            LazyOp::Const => Ir::new("Const"),
            LazyOp::Copy(c) => c.ir(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum OperationError {
    #[error("Failed to compile operation: {0}")]
    CompileError(String),
    #[error("Failed to get storage layout: {0}")]
    StorageLayoutError(#[from] PoolError),
    #[error(transparent)]
    InvariantError(#[from] InvariantError),
    #[error(transparent)]
    KernelBuildError(#[from] KernelBuildError),
    #[error(transparent)]
    UniformError(#[from] encase::internal::Error),
    #[error(transparent)]
    UnknownError(#[from] anyhow::Error),
    #[error("Cannot inplace operation: {0}")]
    InplaceError(String),
    #[error(transparent)]
    DeviceError(#[from] crate::DeviceError),
}

/// Unique string representing a kernel.
/// If the key is registered in the compute pipeline pool, the pipeline is reused.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KernelKey(String);

impl KernelKey {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn new(
        stem: &str,
        inputs: &[&OpTensor],
        output: &OpTensor,
        workgroup_size: &WorkgroupSize,
        inplace: bool,
        kernel_element: &KernelElement,
        additional: Option<&str>,
    ) -> Self {
        let input_dtypes = inputs.iter().map(|t| t.dtype().as_str());
        let inplace_str = if inplace { "ip" } else { "oop" };

        let key_parts: Vec<Cow<'_, str>> = vec![
            Cow::Borrowed(stem),
            Cow::Owned(input_dtypes.collect::<Vec<_>>().join("_")),
            Cow::Owned(output.dtype().to_string()),
            Cow::Owned(workgroup_size.as_key()),
            Cow::Borrowed(inplace_str),
            Cow::Borrowed(additional.unwrap_or("")),
            Cow::Borrowed(kernel_element.as_str()),
        ];

        Self(key_parts.into_iter().collect::<Vec<_>>().join("_"))
    }
}

impl std::fmt::Display for KernelKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug)]
pub struct KernelSource(pub Cow<'static, str>);

impl From<WgslFragment> for KernelSource {
    fn from(value: WgslFragment) -> Self {
        Self(Cow::Owned(value.0))
    }
}

impl From<KernelSource> for wgpu::ShaderSource<'static> {
    fn from(val: KernelSource) -> Self {
        wgpu::ShaderSource::Wgsl(val.0)
    }
}

impl std::fmt::Display for KernelSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// # Operation Guards - Runtime guards for operation correctness.
///
/// Guards should be implemented for all types that will be a node on the high-level CFG.
/// It is used to ensure that the operation is valid and that the resultant tensor is correctly
/// shaped.
///
/// The Rust type system is not sufficient to check all invariants at compile time (we need
/// dependent types). Therefore, we move the checks to runtime.
///
/// All of these methods panic, as they're unrecoverable errors.
pub trait OpGuards {
    #[track_caller]
    fn check_shapes(&self);

    #[track_caller]
    fn check_dtypes(&self);

    // Some operations may have custom invariants to be upheld.
    // e.g reduction dimension being within rank
    #[track_caller]
    fn check_custom(&self) {}
}

#[derive(Debug)]
pub enum IrScalarValue {
    F32(f32),
    I32(i32),
    U32(u32),
    Bool(bool),
    String(String),
    Vec4U32(glam::UVec4),
}

impl Hash for IrScalarValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            IrScalarValue::F32(f) => state.write_u32(f.to_bits()),
            IrScalarValue::I32(i) => i.hash(state),
            IrScalarValue::U32(u) => u.hash(state),
            IrScalarValue::Bool(b) => b.hash(state),
            IrScalarValue::String(s) => s.hash(state),
            IrScalarValue::Vec4U32(v) => v.hash(state),
        }
    }
}

#[derive(Debug)]
pub struct IrTensorValue {
    pub id: TensorId,
    pub shape: Shape,
    pub dtype: DType,
}

#[derive(Debug)]
pub enum IrValue {
    Scalar(IrScalarValue),
    Tensor(IrTensorValue),
    // This is used for Detach only
    Ir(Ir),
    // This should never actually exist on the IR; we use it to
    // pull in a bunch of fields from a struct
    Fields(Box<dyn IrFields>),
    Vec(RVec<Box<IrValue>>),
    // For optional fields
    None,
}

#[derive(Debug)]
pub struct Ir {
    name: String,
    fields: Option<HashMap<String, IrValue>>,
}

fn hash_key<H: Hasher>(key: &str, state: &mut H) {
    state.write(key.as_bytes());
}

fn hash_ir_value<H: Hasher>(
    value: &IrValue,
    state: &mut H,
    tensor_hashes: &BTreeMap<TensorId, u64>,
    compile_key: &Option<GpuCompileKey>,
) {
    match value {
        IrValue::Tensor(IrTensorValue { id, dtype, shape }) => {
            let hash = tensor_hashes.get(id).unwrap();
            hash.hash(state);
            dtype.hash(state);
            shape.hash(state);
        }
        IrValue::Ir(ir) => ir.shape_hash(state, tensor_hashes, compile_key),
        // If list size changes, the IR changes. Can't think of a case where we wouldn't want this
        // to be true...
        IrValue::Vec(vec) => {
            for (i, value) in vec.iter().enumerate() {
                hash_key(&format!("{i}"), state);
                hash_ir_value(value, state, tensor_hashes, compile_key);
            }
        }
        // We don't hash scalar values or None values, relying on the compile key to differentiate
        // them.
        IrValue::Scalar(_) | IrValue::None => state.write_u64(0),
        IrValue::Fields(_) => panic!("Fields should not be present in finalized op"),
    }
}

impl Ir {
    pub fn new(name: impl ToString) -> Self {
        Self {
            name: name.to_string(),
            fields: None,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn fields(&self) -> Option<&HashMap<String, IrValue>> {
        self.fields.as_ref()
    }

    pub fn push_prefix(&mut self, prefix: impl ToString) {
        self.name = format!("{}.{}", prefix.to_string(), self.name);
    }

    pub fn pop_prefix(&mut self) {
        self.name = self.name.split('.').skip(1).collect::<Vec<_>>().join(".");
    }

    pub fn with_field(&mut self, name: impl ToString, value: impl Into<IrValue>) -> &mut Self {
        let value = value.into();
        let fields = self.fields.get_or_insert_default();
        match value {
            IrValue::Fields(fields) => {
                self.push_prefix(name.to_string());
                fields.ir_fields(self);
                self.pop_prefix();
            }
            _ => {
                fields.insert(name.to_string(), value);
            }
        };
        self
    }

    pub fn shape_hash<H: Hasher>(
        &self,
        state: &mut H,
        tensor_hashes: &BTreeMap<TensorId, u64>,
        compile_key: &Option<GpuCompileKey>,
    ) {
        self.name.hash(state);
        // We hash the compile key as our main differentiating factor.
        // If it doesn't change the shader, the IR is the same.
        if let Some(inner) = compile_key.as_ref() {
            match inner {
                GpuCompileKey::Compute(compute_key) => compute_key.key.hash(state),
                GpuCompileKey::Copy(copy_key) => {
                    hash_key("src", state);
                    hash_ir_value(
                        &copy_key.src.clone().into(),
                        state,
                        tensor_hashes,
                        compile_key,
                    );
                    hash_key("dst", state);
                    hash_ir_value(
                        &copy_key.dst.clone().into(),
                        state,
                        tensor_hashes,
                        compile_key,
                    );
                }
            }
        } else {
            state.write_u64(0);
        }

        if let Some(fields) = &self.fields {
            let mut fields = fields.iter().collect::<Vec<_>>();
            fields.sort_by_key(|(k, _)| *k);

            for (key, value) in fields {
                hash_key(key, state);
                hash_ir_value(value, state, tensor_hashes, compile_key);
            }
        }
    }
}

/// # IR Fields
///
/// Allows describing fields of an operation for IR, which makes graph hashing possible.
pub trait IrFields: Debug {
    /// Add operation-specific fields to the IR description
    fn ir_fields(&self, ir: &mut Ir);
}

impl<T: Into<IrValue>> From<Option<T>> for IrValue {
    fn from(value: Option<T>) -> Self {
        match value {
            Some(tensor) => tensor.into(),
            None => IrValue::None,
        }
    }
}

impl From<OpTensor> for IrValue {
    fn from(value: OpTensor) -> Self {
        IrValue::Tensor(IrTensorValue {
            id: value.id(),
            shape: value.shape().clone(),
            dtype: value.dtype(),
        })
    }
}

impl From<f32> for IrValue {
    fn from(value: f32) -> Self {
        IrValue::Scalar(IrScalarValue::F32(value))
    }
}

impl From<i32> for IrValue {
    fn from(value: i32) -> Self {
        IrValue::Scalar(IrScalarValue::I32(value))
    }
}

impl From<u32> for IrValue {
    fn from(value: u32) -> Self {
        IrValue::Scalar(IrScalarValue::U32(value))
    }
}

impl From<usize> for IrValue {
    fn from(value: usize) -> Self {
        IrValue::Scalar(IrScalarValue::U32(value as u32))
    }
}

impl From<bool> for IrValue {
    fn from(value: bool) -> Self {
        IrValue::Scalar(IrScalarValue::Bool(value))
    }
}

impl From<String> for IrValue {
    fn from(value: String) -> Self {
        IrValue::Scalar(IrScalarValue::String(value))
    }
}

impl From<&str> for IrValue {
    fn from(value: &str) -> Self {
        IrValue::Scalar(IrScalarValue::String(value.to_string()))
    }
}

impl From<Range<usize>> for IrValue {
    fn from(value: Range<usize>) -> Self {
        IrValue::Vec(rvec![
            Box::new(IrValue::Scalar(IrScalarValue::U32(value.start as u32))),
            Box::new(IrValue::Scalar(IrScalarValue::U32(value.end as u32)))
        ])
    }
}

macro_rules! impl_irvalue_for_rvec {
    ($t:ty) => {
        impl From<$t> for IrValue {
            fn from(value: $t) -> Self {
                IrValue::Vec(value.into_iter().map(|v| Box::new(v.into())).collect())
            }
        }
    };
}

impl_irvalue_for_rvec!(RVec<Range<usize>>);
impl_irvalue_for_rvec!(RVec<OpTensor>);
impl_irvalue_for_rvec!(RVec<usize>);
impl_irvalue_for_rvec!(Shape);

impl From<DType> for IrValue {
    fn from(value: DType) -> Self {
        IrValue::Scalar(IrScalarValue::String(value.to_string()))
    }
}

impl From<Ir> for IrValue {
    fn from(value: Ir) -> Self {
        IrValue::Ir(value)
    }
}

impl<T: IrFields + 'static> From<T> for IrValue {
    fn from(value: T) -> Self {
        IrValue::Fields(Box::new(value))
    }
}

impl<T: Into<IrValue>> From<TensorTypeOrScalarEnum<T>> for IrValue {
    fn from(value: TensorTypeOrScalarEnum<T>) -> Self {
        match value {
            TensorTypeOrScalarEnum::Tensor(tensor) => tensor.into(),
            TensorTypeOrScalarEnum::Scalar(scalar) => scalar.into(),
        }
    }
}

pub struct ComputeCompileKey<'a> {
    pub dst: &'a OpTensor,
    pub workload: Workload,
    pub key: KernelKey,
    pub can_inplace: bool,
    pub offset: u64,
}

pub struct CopyCompileKey<'a> {
    pub src: &'a OpTensor,
    pub dst: &'a OpTensor,
}

pub enum GpuCompileKey<'a> {
    Compute(ComputeCompileKey<'a>),
    Copy(CopyCompileKey<'a>),
}

impl GpuCompileKey<'_> {
    pub fn can_inplace(&self) -> bool {
        match self {
            GpuCompileKey::Compute(compute_key) => compute_key.can_inplace,
            GpuCompileKey::Copy(_) => false,
        }
    }
}

/// # Operation
///
/// Operation should be implemented for all types that will be a node on the high-level CFG.
///
/// Hardware invariant functions.
pub trait Operation: OpGuards + IrFields + Debug + 'static {
    /// # Operation Name
    fn name(&self) -> &'static str;

    /// # Check Invariants
    ///
    /// All operations have some invariants that must be upheld to ensure correctness.
    fn check_invariants(&self) {
        self.check_shapes();
        self.check_dtypes();
        self.check_custom();
    }
    /// # Compute View
    ///
    /// Determine the type, shape & stride of the resultant tensor.
    fn compute_view(&self) -> Result<StorageView, OperationError>;

    /// # Source Tensors
    fn srcs(&self) -> RVec<&OpTensor>;

    /// # Supports Inplace
    ///
    /// Determine if the operation can be performed in-place.
    fn supports_inplace(&self) -> bool {
        false
    }

    /// Corresponding IR description of the operation.
    fn ir(&self) -> Ir {
        let mut ir = Ir::new(self.name());
        self.ir_fields(&mut ir);
        ir
    }
}

/// An (Web)-GPU implementation of an operation.
///
/// Has an associated kernel enum, which enumerates all possible kernels that can be used for this
/// operation.
/// Binary -> Standard (1:1 mapping)
/// Matmul ─┐          (1:N mapping)
///         ├ GEMM
///         ├ GEMV
///         ├ QGEMM
///         └ QGEMV
pub trait GPUOperation: Operation {
    /// # Kernel Selection
    /// Enumeration of all possible kernels that can be used for this operation.
    type KernelEnum: Kernel;

    fn select_kernel(&self) -> Self::KernelEnum;

    fn create_gpu_compile_key<'a>(
        &self,
        dst: &'a OpTensor,
        can_inplace: bool,
        uniform: &mut CpuUniform,
    ) -> Result<ComputeCompileKey<'a>, OperationError> {
        let kernel = self.select_kernel();

        let kernel_element = kernel.kernel_element(dst);

        let workload = kernel.calculate_dispatch(dst)?;

        let key = kernel.kernel_key(
            &workload.workgroup_size,
            can_inplace,
            &self.srcs(),
            dst,
            &kernel_element,
        );

        let metadata = kernel.metadata(dst, &kernel_element)?;
        let offset = metadata.write(uniform)?;

        log::debug!("Kernel key: {key}");
        log::debug!("Can inplace: {can_inplace}");

        Ok(ComputeCompileKey {
            dst,
            key,
            can_inplace,
            workload,
            offset,
        })
    }

    fn compile_gpu<'a>(
        &self,
        gpu_compile_key: &ComputeCompileKey<'a>,
        device: &'a WgpuDevice,
        // TODO(vinhowe): We should remove this
        debug: bool,
    ) -> Result<CompiledOp, OperationError> {
        let ComputeCompileKey {
            dst,
            key,
            can_inplace,
            workload,
            offset,
        } = gpu_compile_key;

        let kernel = self.select_kernel();

        let storage_bind_group_layout_desc = kernel.storage_bind_group_layout(*can_inplace)?;

        let storage_layout =
            device.get_or_create_bind_group_layout(&storage_bind_group_layout_desc)?;

        let uniform_layout =
            device.get_or_create_bind_group_layout(&BindGroupLayoutDescriptor::uniform())?;
        let pipeline_layout = device.get_or_create_pipeline_layout(&PipelineLayoutDescriptor {
            entries: rvec![storage_layout, uniform_layout],
        })?;

        let kernel_src_desc = KernelModuleDesc { key: key.clone() };

        let kernel_module = device.get_or_create_compute_module(
            &kernel_src_desc,
            &kernel,
            *can_inplace,
            dst,
            &workload.workgroup_size,
            dst.device().try_gpu().unwrap(),
        );

        let pipeline_descriptor = ComputePipelineDescriptor {
            pipeline_layout,
            kernel_key: kernel_src_desc.key.clone(),
            kernel_module,
        };
        let pipeline_handle = device.get_or_create_compute_pipeline(&pipeline_descriptor)?;

        //TODO: Not sure i like this call here
        let storage_bind_groups = CompiledOp::create_storage_bind_groups(
            &self.srcs(),
            dst,
            rvec![storage_layout],
            device,
            *can_inplace,
        )?;

        #[cfg(feature = "debug")]
        let storage_bind_group_layout_entries = storage_bind_group_layout_desc.entries;
        #[cfg(feature = "debug")]
        {
            let storage_bind_group = &storage_bind_groups[0];
            // Assert that no two or more of the storage bind group entries with the same handle have different read_only values
            for (i, (bind_group_entry, layout_entry)) in storage_bind_group
                .descriptor()
                .entries
                .iter()
                .zip(storage_bind_group_layout_entries.iter())
                .enumerate()
            {
                for (j, (other_bind_group_entry, other_layout_entry)) in storage_bind_group
                    .descriptor()
                    .entries
                    .iter()
                    .zip(storage_bind_group_layout_entries.iter())
                    .enumerate()
                {
                    if bind_group_entry.handle == other_bind_group_entry.handle {
                        assert_eq!(
                            layout_entry.read_only,
                            other_layout_entry.read_only,
                            "Found conflicting read_only values for the same buffer handle: {:?} at index {} has read_only={} but {:?} at index {} has read_only={} ({:?})",
                            bind_group_entry.handle,
                            i,
                            layout_entry.read_only,
                            other_bind_group_entry.handle,
                            j,
                            other_layout_entry.read_only,
                            storage_bind_group
                                .descriptor()
                                .entries
                                .iter()
                                .map(|e| e.handle.data())
                                .collect::<RVec<_>>()
                        );
                    }
                }
            }
        }

        #[cfg(feature = "debug")]
        let debug_buffer = if debug {
            Some((
                dst.id(),
                Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("debug buffer"),
                    size: dst.num_bytes() as _,
                    usage: wgpu::BufferUsages::standard(),
                    mapped_at_creation: false,
                })),
            ))
        } else {
            None
        };

        #[cfg(feature = "debug")]
        let debug_input_buffers = if debug {
            Some(
                self.srcs()
                    .iter()
                    .map(|s| {
                        (
                            s.id(),
                            Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some("debug input buffer"),
                                size: max(s.num_bytes(), MIN_STORAGE_BUFFER_SIZE) as _,
                                usage: wgpu::BufferUsages::standard(),
                                mapped_at_creation: false,
                            })),
                        )
                    })
                    .collect::<SmallVec<[(TensorId, Arc<wgpu::Buffer>); 4]>>(),
            )
        } else {
            None
        };

        Ok(CompiledOp::new(
            pipeline_handle,
            workload.workgroup_count.clone(),
            storage_bind_groups,
            *offset as _,
            kernel_src_desc.key,
            // TODO(vinhowe): Figure out how to handle when this should be None
            Some(dst.id()),
            None,
            #[cfg(feature = "debug")]
            debug_input_buffers,
            #[cfg(feature = "debug")]
            storage_bind_group_layout_entries,
        ))
    }
}
