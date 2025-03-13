use crate::gpu::{BindGroupEntry, CpuUniform, WgpuDevice};
use crate::{
    cpu, get_current_scope, ops::*, rvec, shape, BufferSegment, CPUBuffer, Compiled, CompiledOp,
    ComputeCompileKey, DType, Device, DeviceStorage, GPUOperation, GpuCompileKey, InvariantError,
    LazyOp, Operation, OperationError, RVec, RawCPUBuffer, ScopePusher, Shape, Storage, Stride,
    TensorDType, TensorId,
};
use anyhow::Result;
use bitvec::prelude::*;
use derive_new::new;
use maybe_async::maybe_async;
use npyz::WriterBuilder;
use num_traits::AsPrimitive;
use parking_lot::{RwLock, RwLockReadGuard};
use std::borrow::Cow;
use std::io::{BufRead, Seek};
use std::mem::ManuallyDrop;
use std::ops::Bound;
use std::path::Path;
use std::sync::Arc;

#[cfg(feature = "rand")]
use {
    rand::prelude::*,
    rand_distr::{Normal, Uniform},
};

#[cfg(feature = "testing")]
use ndarray::{ArrayD, ArrayViewD, Dimension};

#[cfg(all(not(target_arch = "wasm32"), feature = "pyo3"))]
use numpy::PyArrayDyn;

// thiserror error for Tensor
#[derive(thiserror::Error, Debug)]
pub enum TensorError {
    #[error("Tensor is not resolved")]
    NotResolved,
    #[error("Tensor {0:?} is missing storage")]
    NoStorage(TensorId),
    #[error(transparent)]
    DeviceError(#[from] crate::DeviceError),
    #[error("Failed to transfer data to host")]
    TransferError,
    #[error(transparent)]
    OperationError(#[from] OperationError),
}

/// A multi-dimensional array of data.
///
/// A tensor is a lazy representation of an operation. The nodes required to compute it's
/// value and it's own value will not be computed until `resolve` is called.
#[derive(Clone)]
pub struct Tensor {
    pub(crate) inner: Arc<Inner>,
}

unsafe impl Send for Tensor {}

macro_rules! ensure_resolved_sync {
    ($self:ident) => {
        #[cfg(not(target_arch = "wasm32"))]
        if !$self.resolved() {
            $self
                .apply_pending_graph()
                .expect("Failed to apply pending graph");
        }
    };
}

macro_rules! ensure_resolved {
    ($self:ident) => {
        if !$self.resolved() {
            #[cfg(target_arch = "wasm32")]
            {
                $self
                    .apply_pending_graph()
                    .await
                    .expect("Failed to apply pending graph");
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                $self
                    .apply_pending_graph()
                    .expect("Failed to apply pending graph");
            }
        }
    };
}

impl Tensor {
    fn register_with_device(&self) {
        if let Device::GPU(inner) = self.device() {
            log::trace!("Attempting to register tensor {:?}", self.id());
            inner.register_tensor(self);
        }
    }

    pub(crate) fn new_impl(
        op: LazyOp,
        meta: StorageView,
        storage: Option<Storage>,
        device: Device,
        requires_grad: bool,
    ) -> Self {
        let _scope_guard = ScopePusher::new(op.name());
        let value = Self {
            inner: Arc::new(Inner::new(
                op,
                Some(get_current_scope()),
                meta,
                storage,
                device.clone(),
                requires_grad,
            )),
        };
        value.register_with_device();
        value
    }

    pub fn new(op: LazyOp, meta: StorageView, storage: Option<Storage>, device: Device) -> Self {
        Self::new_impl(op, meta, storage, device, false)
    }

    pub(crate) fn full_impl<T: TensorDType + num_traits::AsPrimitive<f32>>(
        shape: &Shape,
        value: T,
        device: &Device,
        requires_grad: bool,
    ) -> Result<Self> {
        let meta = StorageView {
            shape: shape.clone(),
            dtype: T::dtype(),
            stride: Stride::from(&shape.clone()),
        };
        Ok(Self::new_impl(
            LazyOp::FillConstant(FillConstant {
                shape: shape.clone(),
                value: value.as_(),
            }),
            meta,
            None,
            device.clone(),
            requires_grad,
        ))
    }

    pub fn full<T: TensorDType + num_traits::AsPrimitive<f32>>(
        shape: &Shape,
        value: T,
        device: &Device,
    ) -> Result<Self> {
        Self::full_impl::<T>(shape, value, device, false)
    }

    #[track_caller]
    fn lazy(op: LazyOp, meta: StorageView, device: Device, requires_grad: bool) -> Self {
        op.check_invariants();
        Self::new_impl(op, meta, None, device, requires_grad)
    }

    pub fn shallow(
        op: LazyOp,
        meta: StorageView,
        storage: Arc<RwLock<Option<Storage>>>,
        device: Device,
        requires_grad: bool,
    ) -> Self {
        let _scope_guard = ScopePusher::new(op.name());
        let value = Self {
            inner: Arc::new(Inner::from_shallow(
                op,
                Some(get_current_scope()),
                meta,
                storage,
                device,
                requires_grad,
            )),
        };
        value.register_with_device();
        value
    }

    pub(crate) fn strong_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }

    pub(crate) fn update_storage(&self, storage: Storage) {
        *self.inner.storage.write() = Some(storage);
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.device() {
            Device::CPU => match self.dtype() {
                DType::F32 => self.to_ndarray_view::<f32>().fmt(f),
                _ => {
                    let storage_fmt = self.storage().as_ref().map(|s| s.dump(self.dtype(), false));
                    let (id, op) = (self.id(), self.op());
                    f.debug_struct("Tensor")
                        .field("id", &id)
                        .field("shape", &self.shape())
                        .field("dtype", &self.dtype())
                        .field("op", &op)
                        .field("storage", &storage_fmt)
                        .finish()
                }
            },
            Device::GPU(_) => {
                let storage_fmt = self.storage().as_ref().map(|s| s.dump(self.dtype(), false));
                let (id, op) = (self.id(), self.op());
                f.debug_struct("Tensor")
                    .field("id", &id)
                    .field("shape", &self.shape())
                    .field("dtype", &self.dtype())
                    .field("op", &op)
                    .field("storage", &storage_fmt)
                    .finish()
            }
        }
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.inner.id == other.inner.id
    }
}

impl std::ops::Deref for Tensor {
    type Target = Inner;

    fn deref(&self) -> &Self::Target {
        self.inner.as_ref()
    }
}

/// Tensors are just an view into their underlying byte storage.
#[derive(new, Debug, Clone)]
pub struct StorageView {
    shape: Shape,
    dtype: DType,
    stride: Stride,
}

impl StorageView {
    pub fn is_contiguous(&self) -> bool {
        self.shape.is_contiguous(&self.stride)
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        if let Device::GPU(inner) = &self.device {
            log::trace!("Attempting to unregister tensor {:?}", self.id);
            inner.unregister_tensor(self.id);
        }

        unsafe {
            ManuallyDrop::drop(&mut self.storage);
        }
    }
}

#[derive(Debug)]
pub struct Inner {
    id: TensorId,
    scope: Option<String>,
    op: LazyOp,
    device: Device,
    view: StorageView,
    requires_grad: bool,
    storage: ManuallyDrop<Arc<RwLock<Option<Storage>>>>,
}

impl AsRef<Inner> for Inner {
    fn as_ref(&self) -> &Inner {
        self
    }
}

impl Inner {
    fn new(
        op: LazyOp,
        scope: Option<String>,
        meta: StorageView,
        storage: Option<Storage>,
        device: Device,
        requires_grad: bool,
    ) -> Self {
        Self {
            id: TensorId::new(),
            scope,
            view: meta,
            op,
            device,
            storage: ManuallyDrop::new(Arc::new(RwLock::new(storage))),
            requires_grad,
        }
    }

    fn from_shallow(
        op: LazyOp,
        scope: Option<String>,
        meta: StorageView,
        storage: Arc<RwLock<Option<Storage>>>,
        device: Device,
        requires_grad: bool,
    ) -> Self {
        Self {
            id: TensorId::new(),
            scope,
            view: meta,
            op,
            device,
            storage: ManuallyDrop::new(storage),
            requires_grad,
        }
    }
}

impl Tensor {
    pub fn id(&self) -> TensorId {
        self.inner.id
    }

    pub fn storage_view(&self) -> &StorageView {
        &self.view
    }

    pub fn rank(&self) -> usize {
        self.view.shape.rank()
    }

    pub fn dtype(&self) -> DType {
        self.view.dtype
    }

    pub fn shape(&self) -> &Shape {
        &self.view.shape
    }

    pub fn stride(&self) -> &Stride {
        &self.view.stride
    }

    //WARNING: very wrong for quantized types!
    pub fn num_bytes(&self) -> usize {
        self.view.shape.numel() * self.view.dtype.size_of()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn storage(&self) -> RwLockReadGuard<Option<Storage>> {
        self.inner.storage.read()
    }

    pub fn resolved(&self) -> bool {
        self.storage().is_some()
    }

    pub fn op(&self) -> &LazyOp {
        &self.inner.op
    }

    pub fn scope(&self) -> &Option<String> {
        &self.inner.scope
    }

    pub fn is_scalar(&self) -> bool {
        self.shape().is_scalar()
    }

    pub fn requires_grad(&self) -> bool {
        self.inner.requires_grad
    }

    #[cfg(feature = "plotting")]
    pub fn plot_fmt(&self) -> String {
        let shape = self.shape();
        let dtype = self.dtype();
        let storage = self.storage();
        let storage_fmt = storage
            .as_ref()
            .map(|s| s.plot_fmt())
            .unwrap_or_else(|| "Unresolved".to_string());
        let references = self.strong_count();
        format!(
            "#{:?}-{:?}-{:?}{}\n{:#?}\n{}\n{:?} references",
            self.id(),
            dtype,
            shape,
            if self.requires_grad() { " (param)" } else { "" },
            self.op().ir().fields(),
            storage_fmt,
            references
        )
    }
}

macro_rules! impl_binary_op {
    ($method_name:ident, $op:expr) => {
        #[allow(clippy::should_implement_trait)]
        pub fn $method_name(self, other: Tensor) -> Result<Self> {
            let device = self.device.clone();
            //TODO: avoid broadcasting if either operand is scalar
            let (mut lhs, mut rhs) = (self, other);
            let shapes = &[lhs.shape(), rhs.shape()];
            let broadcasted = Shape::multi_broadcast(shapes);
            if broadcasted.is_none() {
                let failed = shapes.iter().map(|s| (*s).clone()).collect::<Vec<_>>();
                return Err(InvariantError::BroadcastingFailed(failed).into());
            }
            let broadcasted = broadcasted.unwrap();
            let left_required = shapes[0] != &broadcasted;
            let right_required = shapes[1] != &broadcasted;

            (lhs, rhs) = if left_required {
                (lhs.broadcast_to(broadcasted.clone())?, rhs.clone())
            } else if right_required {
                (lhs, rhs.broadcast_to(broadcasted.clone())?)
            } else {
                (lhs, rhs)
            };

            let binary = Binary::new(lhs, rhs, $op);
            let new_view = binary.compute_view()?;

            Ok(Tensor::lazy(
                LazyOp::Binary(binary),
                new_view,
                device,
                false,
            ))
        }
    };
}

macro_rules! impl_cmp_op {
    ($method_name:ident, $op:expr) => {
        #[allow(clippy::should_implement_trait)]
        pub fn $method_name(self, other: Tensor) -> Result<Self> {
            let device = self.device.clone();
            //TODO: avoid broadcasting if either operand is scalar
            let (mut lhs, mut rhs) = (self, other);
            let shapes = &[lhs.shape(), rhs.shape()];
            let broadcasted = Shape::multi_broadcast(shapes);
            if broadcasted.is_none() {
                let failed = shapes.iter().map(|s| (*s).clone()).collect::<Vec<_>>();
                return Err(InvariantError::BroadcastingFailed(failed).into());
            }
            let broadcasted = broadcasted.unwrap();
            let left_required = shapes[0] != &broadcasted;
            let right_required = shapes[1] != &broadcasted;

            (lhs, rhs) = if left_required {
                (lhs.broadcast_to(broadcasted.clone())?, rhs.clone())
            } else if right_required {
                (lhs, rhs.broadcast_to(broadcasted.clone())?)
            } else {
                (lhs, rhs)
            };

            let cmp = Cmp::new(lhs, rhs, $op);
            let new_view = cmp.compute_view()?;

            Ok(Tensor::lazy(LazyOp::Cmp(cmp), new_view, device, false))
        }
    };
}

macro_rules! impl_unary_op {
    ($method_name:ident, $op:expr) => {
        #[allow(clippy::should_implement_trait)]
        pub fn $method_name(self) -> Result<Self> {
            let device = self.device.clone();
            let unary = Unary::new(self.clone(), $op);
            let new_view = unary.compute_view()?;
            Ok(Tensor::lazy(LazyOp::Unary(unary), new_view, device, false))
        }
    };
}

impl Tensor {
    impl_binary_op!(add, BinaryOp::Add);
    impl_binary_op!(sub, BinaryOp::Sub);
    impl_binary_op!(mul, BinaryOp::Mul);
    impl_binary_op!(div, BinaryOp::Div);

    impl_cmp_op!(eq, CmpOp::Eq);
    impl_cmp_op!(ne, CmpOp::Ne);
    impl_cmp_op!(le, CmpOp::Le);
    impl_cmp_op!(ge, CmpOp::Ge);
    impl_cmp_op!(lt, CmpOp::Lt);
    impl_cmp_op!(gt, CmpOp::Gt);

    impl_unary_op!(gelu, UnaryOp::Gelu);
    impl_unary_op!(tanh, UnaryOp::Tanh);
    impl_unary_op!(exp, UnaryOp::Exp);
    impl_unary_op!(log, UnaryOp::Log);
    impl_unary_op!(sin, UnaryOp::Sin);
    impl_unary_op!(cos, UnaryOp::Cos);
    impl_unary_op!(abs, UnaryOp::Abs);
    impl_unary_op!(sqrt, UnaryOp::Sqrt);
    impl_unary_op!(relu, UnaryOp::Relu);
    impl_unary_op!(relu2, UnaryOp::Relu2);
    impl_unary_op!(floor, UnaryOp::Floor);
    impl_unary_op!(ceil, UnaryOp::Ceil);
    impl_unary_op!(neg, UnaryOp::Neg);
    impl_unary_op!(sigmoid, UnaryOp::Sigmoid);
    impl_unary_op!(swiglu, UnaryOp::Swiglu);
    impl_unary_op!(silu, UnaryOp::Silu);
    impl_unary_op!(square, UnaryOp::Square);
    impl_unary_op!(recip, UnaryOp::Reciprocal);

    pub fn cast(self, dst_dtype: DType) -> Result<Self> {
        if self.dtype() == dst_dtype {
            return Ok(self);
        }

        let device = self.device.clone();
        let cast = Cast::new(self, dst_dtype);
        let new_view = cast.compute_view()?;
        Ok(Tensor::lazy(LazyOp::Cast(cast), new_view, device, false))
    }

    /// Cast a tensor to full precision (IEEE 754 32-bit floating point).
    pub fn float(self) -> Result<Self> {
        self.cast(DType::F32)
    }

    /// Cast a tensor to half precision (IEEE 754 16-bit floating point).
    pub fn half(self) -> Result<Self> {
        self.cast(DType::F16)
    }

    pub fn group_norm(
        self,
        num_groups: usize,
        weight: Tensor,
        bias: Option<Tensor>,
        eps: f32,
    ) -> Result<Self> {
        let device = self.device.clone();
        let group_norm = GroupNorm::new(Norm::new(self, weight, bias, eps), num_groups);
        let norm_op = NormOp::GroupNorm(group_norm);
        let new_view = norm_op.compute_view()?;
        Ok(Tensor::lazy(LazyOp::Norm(norm_op), new_view, device, false))
    }

    pub fn layer_norm(self, weight: Tensor, bias: Option<Tensor>, eps: f32) -> Result<Self> {
        let device = self.device.clone();
        let layer_norm = Norm::new(self, weight, bias, eps);
        let op = NormOp::LayerNorm(layer_norm);
        let new_view = op.compute_view()?;
        Ok(Tensor::lazy(LazyOp::Norm(op), new_view, device, false))
    }

    pub fn rms_norm(self, weight: Tensor, eps: f32) -> Result<Self> {
        let device = self.device.clone();
        let rms = Norm::new(self, weight, None, eps);
        let op = NormOp::RMSNorm(rms);
        let new_view = op.compute_view()?;
        Ok(Tensor::lazy(LazyOp::Norm(op), new_view, device, false))
    }

    pub fn conv1d(
        self,
        weight: Tensor,
        bias: Option<Tensor>,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        let device = self.device.clone();
        let conv = Conv::new(self, weight, bias, stride, padding);
        let new_view = conv.compute_view()?;
        Ok(Tensor::lazy(LazyOp::Conv(conv), new_view, device, false))
    }

    //TODO: switch dim to isize and allow negative indexing
    pub fn softmax(self, dim: usize) -> Result<Self> {
        let device = self.device.clone();
        let softmax = Softmax::new(self, dim);
        let new_view = softmax.compute_view()?;
        Ok(Tensor::lazy(
            LazyOp::Softmax(softmax),
            new_view,
            device,
            false,
        ))
    }

    fn rope_impl(self, dim: usize, base: f32, offset: usize, is_backward: bool) -> Result<Self> {
        let device = self.device.clone();
        let rope = RoPE::new(self, dim, base, offset, is_backward);
        let new_view = rope.compute_view()?;
        Ok(Tensor::lazy(LazyOp::RoPE(rope), new_view, device, false))
    }

    pub fn rope(self, dim: usize, base: f32, offset: usize) -> Result<Self> {
        self.rope_impl(dim, base, offset, false)
    }

    pub(crate) fn rope_backward(self, dim: usize, base: f32, offset: usize) -> Result<Self> {
        self.rope_impl(dim, base, offset, true)
    }

    pub fn alibi(self, max_bias: f32) -> Result<Self> {
        let device = self.device.clone();
        let alibi = Alibi::new(self, max_bias);
        let new_view = alibi.compute_view()?;
        Ok(Tensor::lazy(LazyOp::Alibi(alibi), new_view, device, false))
    }

    //TODO: horrific interface
    pub fn matmul(self, rhs: Tensor, trans_lhs: bool, trans_rhs: bool) -> Result<Self> {
        let device = self.device.clone();
        let matmul = Matmul::new(self, rhs, None, trans_lhs, trans_rhs, false);
        let new_view = matmul.compute_view()?;
        Ok(Tensor::lazy(
            LazyOp::Matmul(matmul),
            new_view,
            device,
            false,
        ))
    }

    pub fn gemm(
        self,
        rhs: Tensor,
        bias: Option<Tensor>,
        trans_lhs: bool,
        trans_rhs: bool,
        trans_out: bool,
    ) -> Result<Self> {
        let device = self.device.clone();
        let gemm = Matmul::new(self, rhs, bias, trans_lhs, trans_rhs, trans_out);
        let new_view = gemm.compute_view()?;
        Ok(Tensor::lazy(LazyOp::Matmul(gemm), new_view, device, false))
    }

    pub fn affine(self, mul: f32, add: f32) -> Result<Self> {
        let device = self.device.clone();
        let affine = Affine::new(self, mul, add);
        let new_view = affine.compute_view()?;
        Ok(Tensor::lazy(
            LazyOp::Affine(affine),
            new_view,
            device,
            false,
        ))
    }

    pub fn powf(self, e: f32) -> Result<Self> {
        let device = self.device.clone();
        let powf = Powf::new(self, e);
        let new_view = powf.compute_view()?;
        Ok(Tensor::lazy(LazyOp::Powf(powf), new_view, device, false))
    }

    fn reduce_impl(self, dim: usize, keepdim: bool, op: ReduceOp) -> Result<Self> {
        let device = self.device.clone();
        let reduce = Reduce::new(self, op, rvec![dim], keepdim);
        let new_view = reduce.compute_view()?;
        Ok(Tensor::lazy(
            LazyOp::Reduce(reduce),
            new_view,
            device,
            false,
        ))
    }

    fn sum_impl(self, sum_dims: &[usize], keepdim: bool) -> Result<Self> {
        let device = self.device.clone();
        let sum = Reduce::new(self, ReduceOp::Sum, sum_dims.into(), keepdim);
        let new_view = sum.compute_view()?;
        Ok(Tensor::lazy(LazyOp::Reduce(sum), new_view, device, false))
    }

    pub fn sum_keepdim(self, sum_dims: &[usize]) -> Result<Self> {
        self.sum_impl(sum_dims, true)
    }

    pub fn sum(self, sum_dims: &[usize]) -> Result<Self> {
        self.sum_impl(sum_dims, false)
    }

    pub fn sum_all(self) -> Result<Self> {
        let dims: Vec<_> = (0..self.rank()).collect();
        self.sum(&dims)
    }

    pub fn max_keepdim(self, dim: usize) -> Result<Self> {
        self.reduce_impl(dim, true, ReduceOp::Max)
    }

    pub fn max(self, dim: usize) -> Result<Self> {
        self.reduce_impl(dim, false, ReduceOp::Max)
    }

    pub fn min_keepdim(self, dim: usize) -> Result<Self> {
        self.reduce_impl(dim, true, ReduceOp::Min)
    }

    pub fn min(self, dim: usize) -> Result<Self> {
        self.reduce_impl(dim, false, ReduceOp::Min)
    }

    pub fn argmax_keepdim(self, dim: usize) -> Result<Self> {
        self.reduce_impl(dim, true, ReduceOp::ArgMax)
    }

    pub fn argmax(self, dim: usize) -> Result<Self> {
        self.reduce_impl(dim, false, ReduceOp::ArgMax)
    }

    pub fn argmin_keepdim(self, dim: usize) -> Result<Self> {
        self.reduce_impl(dim, true, ReduceOp::ArgMin)
    }

    /// Similar to `argmin_keepdim` but the target dimension is squeezed.
    pub fn argmin(self, dim: usize) -> Result<Self> {
        self.reduce_impl(dim, false, ReduceOp::ArgMin)
    }

    pub fn norm(self) -> Result<Self> {
        self.square()?.sum_all()?.sqrt()
    }

    fn flatten_impl(self, start_dim: Option<usize>, end_dim: Option<usize>) -> Result<Self> {
        if self.rank() == 0 {
            self.view(shape![1])
        } else {
            let start_dim = start_dim.unwrap_or(0);
            let end_dim = end_dim.unwrap_or(self.rank() - 1);
            if start_dim < end_dim {
                let dims = self.shape();
                let mut dst_dims = dims[..start_dim].to_vec();
                dst_dims.push(
                    dims.to_vec()[start_dim..end_dim + 1]
                        .iter()
                        .product::<usize>(),
                );
                if end_dim + 1 < dims.len() {
                    dst_dims.extend(&dims[end_dim + 1..]);
                }
                self.view(Shape::from(dst_dims))
            } else {
                Ok(self.clone())
            }
        }
    }

    pub fn flatten(self, start_dim: usize, end_dim: usize) -> Result<Self> {
        self.flatten_impl(Some(start_dim), Some(end_dim))
    }

    pub fn flatten_to(self, end_dim: usize) -> Result<Self> {
        self.flatten_impl(None::<usize>, Some(end_dim))
    }

    pub fn flatten_from(self, start_dim: usize) -> Result<Self> {
        self.flatten_impl(Some(start_dim), None::<usize>)
    }

    pub fn flatten_all(self) -> Result<Self> {
        self.flatten_impl(None::<usize>, None::<usize>)
    }

    /// # Slice
    ///
    /// Current slice implementation requires specification of all dimensions.
    /// Currently very user hostile, but will be improved.
    /// TODO: should allow mixed range types
    pub fn slice<D: std::ops::RangeBounds<usize>>(self, ranges: &[D]) -> Result<Self> {
        let device = self.device.clone();
        let mut resolved_ranges = rvec![];

        for (ridx, r) in ranges.iter().enumerate() {
            let start = match r.start_bound() {
                Bound::Included(&s) => s,
                Bound::Excluded(&s) => s + 1,
                Bound::Unbounded => 0,
            };
            let end = match r.end_bound() {
                Bound::Included(&e) => e + 1,
                Bound::Excluded(&e) => e,
                Bound::Unbounded => self.shape()[ridx],
            };
            resolved_ranges.push(start..end);
        }

        let slice = Slice::new(self, resolved_ranges);
        let out_view = slice.compute_view()?;
        let op = LazyOp::Reindex(Reindex::Slice(slice));
        Ok(Tensor::lazy(op, out_view, device, false))
    }

    /// # View
    ///
    /// Creates a new tensor with the same data, but a different shape.
    /// The new shape must have the same number of elements as the original shape.
    pub fn view(self, shape: Shape) -> Result<Self> {
        if self.shape().numel() != shape.numel() {
            anyhow::bail!(
                "Cannot reshape tensor with {} elements to shape {:?} ({} elements)",
                self.shape().numel(),
                shape,
                shape.numel()
            );
        }
        let device = self.device.clone();
        let storage = Arc::clone(&self.storage);
        let op = View::new(self, shape);
        let out_view = op.compute_view()?;

        Ok(Tensor::shallow(
            LazyOp::View(op),
            out_view,
            storage,
            device,
            false,
        ))
    }

    // Use view to add a singleton dimension
    pub fn unsqueeze(self, dim: usize) -> Result<Self> {
        let mut new_shape = self.shape().clone();
        new_shape.unsqueeze(dim);
        self.view(new_shape)
    }

    pub fn squeeze(self) -> Result<Self> {
        let mut new_shape = self.shape().clone();
        new_shape.squeeze();
        self.view(new_shape)
    }

    pub fn cat(tensors: RVec<Tensor>, dim: usize) -> Result<Self> {
        let device = tensors[0].device.clone();
        assert!(tensors.iter().all(|t| t.device == device), "Mixed devices");

        let cat = Concat::new(tensors, dim);
        let new_view = cat.compute_view()?;
        Ok(Tensor::lazy(LazyOp::Concat(cat), new_view, device, false))
    }

    fn stack_impl(tensors: RVec<Tensor>, dim: usize, root: bool) -> Result<Self> {
        match tensors.len() {
            0 => anyhow::bail!("Cannot stack empty list of tensors"),
            1 => {
                if root {
                    Ok(tensors[0].clone().unsqueeze(dim)?)
                } else {
                    Ok(tensors[0].clone())
                }
            }
            len => {
                let tensors = if root {
                    tensors
                        .iter()
                        .map(|t| t.clone().unsqueeze(dim))
                        .collect::<anyhow::Result<RVec<Tensor>>>()?
                } else {
                    tensors
                };

                let device = tensors[0].device.clone();
                assert!(tensors.iter().all(|t| t.device == device), "Mixed devices");

                if len <= 4 {
                    return Self::cat(tensors, dim);
                }

                // Process tensors in chunks of 4 recursively
                let mut current_level = tensors;

                while current_level.len() > 1 {
                    let mut next_level = RVec::with_capacity((current_level.len() + 3) / 4);

                    for chunk in current_level.chunks(4) {
                        let chunk_vec = chunk.iter().cloned().collect();
                        let reduced = Self::stack_impl(chunk_vec, dim, false)?;
                        next_level.push(reduced);
                    }

                    current_level = next_level;
                }

                Ok(current_level.into_iter().next().unwrap())
            }
        }
    }

    pub fn stack(tensors: RVec<Tensor>, dim: usize) -> Result<Self> {
        Self::stack_impl(tensors, dim, true)
    }

    pub fn permute(self, dims: &[usize]) -> Result<Self> {
        let device = self.device.clone();
        let permute = Permute::new(self, dims.into());
        let out_view = permute.compute_view()?;

        let op = LazyOp::Reindex(Reindex::Permute(permute));
        Ok(Tensor::lazy(op, out_view, device, false))
    }

    pub fn cache(self, source: Tensor, dim: usize, offset: usize) -> Result<Self> {
        let device = self.device.clone();
        let cache = Cache::new(self, source, dim, offset);
        let new_view = cache.compute_view()?;
        Ok(Tensor::lazy(LazyOp::Cache(cache), new_view, device, false))
    }

    /// Returns a new tensor duplicating data from the original tensor. New dimensions are inserted
    /// on the left.
    pub fn broadcast_left(self, left_shape: Shape) -> Result<Self> {
        let mut dims = left_shape.to_vec();
        dims.extend(self.shape().to_vec());
        self.broadcast_to(Shape::from(dims))
    }

    pub fn broadcast_to(self, shape: Shape) -> Result<Self> {
        let device = self.device.clone();
        let broadcast = Broadcast::new(self, shape);
        let new_view = broadcast.compute_view()?;

        let op = LazyOp::Reindex(Reindex::Broadcast(broadcast));
        Ok(Tensor::lazy(op, new_view, device, false))
    }

    pub fn index_select(self, indices: Tensor, dim: usize) -> Result<Self> {
        let device = self.device.clone();
        let index_select = IndexSelect::new(self, indices, dim);
        let new_view = index_select.compute_view()?;
        Ok(Tensor::lazy(
            LazyOp::Select(index_select),
            new_view,
            device,
            false,
        ))
    }

    pub fn index_write(self, src: Tensor, write_start: RVec<usize>) -> Result<Self> {
        let device = self.device.clone();
        let index_write = IndexWrite::new(self, src, write_start);
        let new_view = index_write.compute_view()?;
        let op = LazyOp::IndexWrite(index_write);
        Ok(Tensor::lazy(op, new_view, device, false))
    }

    pub fn where_cond(self, on_true: Tensor, on_false: Tensor) -> Result<Self> {
        let device = self.device.clone();
        let where_cond = WhereCond::new(self, on_true, on_false);
        let new_view = where_cond.compute_view()?;
        Ok(Tensor::lazy(
            LazyOp::WhereCond(where_cond),
            new_view,
            device,
            false,
        ))
    }

    pub fn scatter_add(self, indices: Tensor, source: Tensor, dim: usize) -> Result<Self> {
        let source_dims = source.shape().to_vec();
        let self_dims = self.shape().to_vec();
        let mismatch = if source_dims.len() != self_dims.len() {
            true
        } else {
            let mut mismatch = false;
            for (i, (&d1, &d2)) in self_dims.iter().zip(source_dims.iter()).enumerate() {
                if i != dim && d1 != d2 {
                    mismatch = true;
                    break;
                }
            }
            mismatch
        };
        if mismatch {
            Err(InvariantError::ShapeMismatchBinaryOp {
                op: "scatter-add (self, src)",
                lhs: self.shape().clone(),
                rhs: source.shape().clone(),
            })?
        }
        if indices.shape() != source.shape() {
            Err(InvariantError::ShapeMismatchBinaryOp {
                op: "scatter-add (indexes, src)",
                lhs: indices.shape().clone(),
                rhs: source.shape().clone(),
            })?
        }
        let device = self.device.clone();
        let scatter_add = ScatterAdd::new(self, source, indices, dim);
        let new_view = scatter_add.compute_view()?;
        Ok(Tensor::lazy(
            LazyOp::ScatterAdd(scatter_add),
            new_view,
            device,
            false,
        ))
    }

    pub fn index_add(self, indices: Tensor, source: Tensor, dim: usize) -> Result<Self> {
        let source_dims = source.shape().to_vec();
        let self_dims = self.shape().to_vec();
        let mismatch = if source_dims.len() != self_dims.len() {
            true
        } else {
            let mut mismatch = false;
            for (i, (&d1, &d2)) in self_dims.iter().zip(source_dims.iter()).enumerate() {
                if i != dim && d1 != d2 {
                    mismatch = true;
                    break;
                }
            }
            mismatch
        };
        if mismatch {
            Err(InvariantError::ShapeMismatchBinaryOp {
                op: "index_add",
                lhs: self.shape().clone(),
                rhs: source.shape().clone(),
            })?
        }
        if indices.rank() != 1 {
            Err(InvariantError::RankMismatch {
                accepted: 1..=1,
                actual: indices.rank(),
            })?
        }
        let indices_len = indices.shape()[0];
        if source_dims[dim] != indices_len {
            Err(InvariantError::ShapeMismatchBinaryOp {
                op: "index_add",
                lhs: indices.shape().clone(),
                rhs: source.shape().clone(),
            })?
        }
        let device = self.device.clone();
        let index_add = IndexAdd::new(self, source, indices, dim);
        let new_view = index_add.compute_view()?;
        Ok(Tensor::lazy(
            LazyOp::IndexAdd(index_add),
            new_view,
            device,
            false,
        ))
    }

    pub fn gather(self, indices: Tensor, dim: usize) -> Result<Self> {
        let self_dims = self.shape().to_vec();
        let indices_dims = indices.shape().to_vec();
        let mismatch = if indices_dims.len() != self_dims.len() {
            true
        } else {
            let mut mismatch = false;
            for (i, (&d1, &d2)) in self_dims.iter().zip(indices_dims.iter()).enumerate() {
                if i != dim && d1 != d2 {
                    mismatch = true;
                    break;
                }
            }
            mismatch
        };
        if mismatch {
            Err(InvariantError::ShapeMismatchBinaryOp {
                op: "gather",
                lhs: self.shape().clone(),
                rhs: indices.shape().clone(),
            })?
        }
        let device = self.device.clone();
        let gather = Gather::new(self, indices, dim);
        let new_view = gather.compute_view()?;
        Ok(Tensor::lazy(
            LazyOp::Gather(gather),
            new_view,
            device,
            false,
        ))
    }

    pub fn arange<T: TensorDType + PartialOrd + AsPrimitive<f32>>(
        start: T,
        end: T,
        device: &Device,
    ) -> Result<Self> {
        Self::arange_step::<T>(start, end, T::one(), device)
    }

    /// Creates a new 1D tensor with values from the interval `[start, end)` taken with a common
    /// difference `step` from `start`.
    pub fn arange_step<T: TensorDType + PartialOrd + AsPrimitive<f32>>(
        start: T,
        end: T,
        step: T,
        device: &Device,
    ) -> Result<Self> {
        if step == T::zero() {
            anyhow::bail!("step cannot be zero")
        }

        if device.is_cpu() {
            let mut data = vec![];
            let mut current = start;
            if step >= T::zero() {
                while current < end {
                    data.push(current);
                    current = current + step;
                }
            } else {
                while current > end {
                    data.push(current);
                    current = current + step;
                }
            }
            let len = data.len();
            Ok(Tensor::from_data(data, shape![len], device.clone()))
        } else {
            let arange = Arange::new(start.as_(), end.as_(), step.as_());
            let numel = arange.numel();
            let op = LazyOp::Arange(arange);

            let meta = StorageView {
                shape: shape![numel],
                dtype: T::dtype(),
                stride: Stride::from(&shape![numel]),
            };

            Ok(Tensor::lazy(op, meta, device.clone(), false))
        }
    }

    #[cfg(feature = "rand")]
    pub(crate) fn randint_impl<T: TensorDType + rand_distr::uniform::SampleUniform + PartialOrd>(
        low: T,
        high: T,
        shape: Shape,
        device: Device,
        requires_grad: bool,
    ) -> Result<Self> {
        let rng = device.get_rng();
        let data = (0..shape.numel())
            .map(|_| {
                let sample: T = rng.write().gen_range(low..high);
                sample
            })
            .collect::<Vec<_>>();
        Ok(Tensor::from_data_impl(data, shape, device, requires_grad))
    }

    #[cfg(feature = "rand")]
    pub fn randint<T: TensorDType + rand_distr::uniform::SampleUniform + PartialOrd>(
        low: T,
        high: T,
        shape: Shape,
        device: Device,
    ) -> Result<Self> {
        Self::randint_impl(low, high, shape, device, false)
    }

    #[cfg(feature = "rand")]
    pub(crate) fn randn_impl<T: TensorDType + num_traits::Float>(
        mean: T,
        std: T,
        shape: Shape,
        device: Device,
        requires_grad: bool,
    ) -> Result<Self> {
        let rng = device.get_rng();
        if device.is_cpu() {
            let distr = Normal::new(mean.to_f64().unwrap(), std.to_f64().unwrap()).unwrap();
            let data = (0..shape.numel())
                .map(|_| {
                    let sample: f64 = distr.sample(&mut *rng.write());
                    T::from(sample as f32).expect("Failed to convert sample")
                })
                .collect::<Vec<_>>();
            let storage = Storage::from_slice(&data, &shape, &device);
            let stride = Stride::from(&shape);
            let meta = StorageView::new(shape, T::dtype(), stride);
            Ok(Self::new_impl(
                LazyOp::Const,
                meta,
                Some(storage),
                device,
                requires_grad,
            ))
        } else {
            let meta = StorageView {
                shape: shape.clone(),
                dtype: T::dtype(),
                stride: Stride::from(&shape.clone()),
            };
            Ok(Self::new_impl(
                LazyOp::FillRandn(FillRandn {
                    shape,
                    mean: mean.to_f32().unwrap(),
                    std: std.to_f32().unwrap(),
                    seed: Some(rng.write().next_u32()),
                }),
                meta,
                None,
                device,
                requires_grad,
            ))
        }
    }

    #[cfg(feature = "rand")]
    pub fn randn<T: TensorDType + num_traits::Float>(
        mean: T,
        std: T,
        shape: Shape,
        device: Device,
    ) -> Result<Self> {
        Self::randn_impl::<T>(mean, std, shape, device, false)
    }

    #[cfg(feature = "rand")]
    pub(crate) fn rand_impl<T: TensorDType + num_traits::Float>(
        lo: T,
        up: T,
        shape: Shape,
        device: Device,
        requires_grad: bool,
    ) -> Result<Self> {
        let rng = device.get_rng();
        let distr = Uniform::new(lo.to_f32().unwrap(), up.to_f32().unwrap());
        let data = (0..shape.numel())
            .map(|_| {
                let sample: f32 = distr.sample(&mut *rng.write());
                T::from(sample as f32).expect("Failed to convert sample")
            })
            .collect::<Vec<_>>();

        Ok(Self::from_data_impl(data, shape, device, requires_grad))
    }

    #[cfg(feature = "rand")]
    pub fn rand<T: TensorDType + num_traits::Float>(
        lo: T,
        up: T,
        shape: Shape,
        device: Device,
    ) -> Result<Self> {
        Self::rand_impl::<T>(lo, up, shape, device, false)
    }

    #[cfg(feature = "rand")]
    pub fn bernoulli(self) -> Result<Self> {
        let rng = self.device().get_rng();
        let seed = rng.write().next_u32();
        let shape = self.shape();
        let device = self.device().clone();

        let meta = StorageView {
            shape: shape.clone(),
            dtype: DType::F32,
            stride: Stride::from(shape),
        };

        Ok(Self::new_impl(
            LazyOp::Bernoulli(Bernoulli::new(self, Some(seed))),
            meta,
            None,
            device,
            false,
        ))
    }

    pub(crate) fn zeros_impl<T: TensorDType + num_traits::AsPrimitive<f32>>(
        shape: &Shape,
        device: &Device,
        requires_grad: bool,
    ) -> Result<Self> {
        if device.is_cpu() {
            let storage = Storage::zeros::<T>(shape, device);
            let stride = Stride::from(shape);
            let meta = StorageView::new(shape.clone(), T::dtype(), stride);
            Ok(Tensor::new_impl(
                LazyOp::Const,
                meta,
                Some(storage),
                device.clone(),
                requires_grad,
            ))
        } else {
            Self::full_impl(shape, T::zero(), device, requires_grad)
        }
    }

    pub fn zeros<T: TensorDType + num_traits::AsPrimitive<f32>>(
        shape: &Shape,
        device: &Device,
    ) -> Result<Self> {
        Self::zeros_impl::<T>(shape, device, false)
    }

    pub fn zeros_like<T: TensorDType + num_traits::AsPrimitive<f32>>(
        &self,
        device: Option<&Device>,
    ) -> Result<Self> {
        Self::zeros::<T>(self.shape(), device.unwrap_or(self.device()))
    }

    pub(crate) fn ones_impl<T: TensorDType + num_traits::AsPrimitive<f32>>(
        shape: &Shape,
        device: &Device,
        requires_grad: bool,
    ) -> Result<Self> {
        if device.is_cpu() {
            let storage = Storage::ones::<T>(shape, device);
            let stride = Stride::from(shape);
            let meta = StorageView::new(shape.clone(), T::dtype(), stride);
            Ok(Tensor::new_impl(
                LazyOp::Const,
                meta,
                Some(storage),
                device.clone(),
                requires_grad,
            ))
        } else {
            Self::full_impl(shape, T::one(), device, requires_grad)
        }
    }

    pub fn ones<T: TensorDType + num_traits::AsPrimitive<f32>>(
        shape: &Shape,
        device: &Device,
    ) -> Result<Self> {
        Self::ones_impl::<T>(shape, device, false)
    }

    pub fn ones_like<T: TensorDType + num_traits::AsPrimitive<f32>>(
        &self,
        device: Option<&Device>,
    ) -> Result<Self> {
        Self::ones::<T>(self.shape(), device.unwrap_or(self.device()))
    }

    fn trilu(self, upper: bool, k: Option<i32>) -> Result<Self> {
        let device = self.device.clone();
        let trilu = Trilu::new(self, upper, k);
        let new_view = trilu.compute_view()?;
        Ok(Tensor::lazy(LazyOp::Trilu(trilu), new_view, device, false))
    }

    pub fn triu(self, k: Option<i32>) -> Result<Self> {
        self.trilu(true, k)
    }

    pub fn tril(self, k: Option<i32>) -> Result<Self> {
        self.trilu(false, k)
    }

    /// Returns true if the data is stored in a C contiguous (aka row major) way.
    pub fn is_contiguous(&self) -> bool {
        self.view.is_contiguous()
    }

    /// Returns a tensor that is in row major order. This is the same as the original tensor if it
    /// was already contiguous, otherwise a copy is triggered.
    pub fn contiguous(self) -> Result<Self> {
        if self.is_contiguous() {
            Ok(self.clone())
        } else {
            let storage_guard = self.storage();
            let storage = storage_guard.as_ref().unwrap();
            let cloned_storage = storage.deep_clone(self.device()).unwrap();
            Ok(Tensor::new_impl(
                LazyOp::Const,
                self.view.clone(),
                Some(cloned_storage),
                self.device.clone(),
                false,
            ))
        }
    }

    pub fn has_nan<T: TensorDType + num_traits::Float>(&self) -> bool {
        assert!(self.device().is_cpu());
        let self_nd = self.to_ndarray_view::<T>();
        self_nd.iter().any(|&x| !x.is_finite())
    }

    /// Creates a new tensor from a chunk of data.
    ///
    /// The Tensor is instantly resolved.
    /// If a non-CPU device is specified, the data will be copied to the device.
    pub(crate) fn from_data_impl<T: TensorDType, U: AsRef<[T]>>(
        data: U,
        shape: Shape,
        device: Device,
        requires_grad: bool,
    ) -> Self {
        let storage = Storage::from_slice(data.as_ref(), &shape, &device);
        let stride = Stride::from(&shape);
        let meta = StorageView::new(shape, T::dtype(), stride);
        Tensor::new_impl(LazyOp::Const, meta, Some(storage), device, requires_grad)
    }

    pub fn from_data<T: TensorDType, U: AsRef<[T]>>(data: U, shape: Shape, device: Device) -> Self {
        Self::from_data_impl(data, shape, device, false)
    }

    pub fn from_bytes(data: &[u8], dtype: DType, shape: Shape, device: Device) -> Result<Self> {
        let storage = Storage::from_bytes(data, dtype.size_of(), &device);
        let stride = Stride::from(&shape);
        let meta = StorageView::new(shape, dtype, stride);
        Ok(Tensor::new_impl(
            LazyOp::Const,
            meta,
            Some(storage),
            device,
            false,
        ))
    }

    /// Create a parameter based on the values currently stored in a tensor. The storage is always
    /// copied.
    pub(crate) fn make_parameter(&self) -> Result<Self> {
        let storage_guard = self.storage();
        let storage = storage_guard.as_ref().unwrap();
        let cloned_storage = storage.deep_clone(self.device()).unwrap();
        Ok(Tensor::new_impl(
            LazyOp::Const,
            self.view.clone(),
            Some(cloned_storage),
            self.device.clone(),
            true,
        ))
    }

    /// Returns a new tensor detached from the current graph, gradient are not propagated through
    /// this new node. The storage of this tensor is shared with the initial tensor.
    ///
    /// If the tensor is already detached from the computation graph, the same tensor is returned.
    pub fn detach(&self) -> Self {
        match self.op {
            LazyOp::Const if !self.requires_grad => self.clone(),
            _ => {
                let storage_guard = self.storage();
                let storage = storage_guard.as_ref().map(|s| s.clone());
                Self::new(
                    LazyOp::Detach(Box::new(self.op().clone())),
                    self.view.clone(),
                    storage,
                    self.device.clone(),
                )
            }
        }
    }

    pub fn copy(&self, dst: &Self) -> Self {
        Tensor::new_impl(
            LazyOp::Copy(TensorCopy {
                src: self.clone(),
                dst: dst.clone(),
            }),
            self.view.clone(),
            None,
            self.device.clone(),
            false,
        )
    }

    pub(crate) fn same_storage(&self, rhs: &Self) -> bool {
        match (self.storage().as_ref(), rhs.storage().as_ref()) {
            (Some(lhs), Some(rhs)) => std::ptr::eq(lhs, rhs),
            _ => false,
        }
    }

    /// # Safety
    ///
    /// If the tensor has more than 1 reference, you die.
    /// If the tensor has no storage, you die.
    pub fn into_bytes(self) -> anyhow::Result<Vec<u8>> {
        let mut inner = Arc::try_unwrap(self.inner).map_err(|_| {
            anyhow::anyhow!("Cannot convert tensor into bytes with multiple references.")
        })?;
        let storage = unsafe { ManuallyDrop::take(&mut inner.storage) };
        let storage = Arc::try_unwrap(storage).unwrap().into_inner().unwrap();
        Ok(storage.into_bytes())
    }

    pub fn from_quantized<T: TensorDType, U: AsRef<[T]>>(
        data: U,
        dtype: DType,
        shape: Shape,
        device: Device,
    ) -> Self {
        let storage = unsafe { Storage::from_quantized(data.as_ref(), &device) };
        let stride = Stride::from(&shape);
        let meta = StorageView::new(shape, dtype, stride);
        Tensor::new_impl(LazyOp::Const, meta, Some(storage), device, false)
    }

    pub fn from_disk<T: TensorDType, R: BufRead + Seek>(
        reader: &mut R,
        shape: Shape,
        device: Device,
    ) -> Result<Self> {
        let storage = Storage::from_disk::<T, R>(reader, &shape, &device)?;
        let stride = Stride::from(&shape);
        let meta = StorageView::new(shape, T::dtype(), stride);
        Ok(Tensor::new_impl(
            LazyOp::Const,
            meta,
            Some(storage),
            device,
            false,
        ))
    }

    #[maybe_async]
    pub async fn item<T: TensorDType>(&self) -> T {
        assert!(self.is_scalar());
        ensure_resolved!(self);
        let storage_guard = self.storage();
        let buffer = storage_guard.as_ref().unwrap().try_cpu().unwrap();
        buffer.to_slice::<T>(self.shape())[0]
    }

    /// # Bindings
    ///
    /// Only applicable to GPU tensors.
    /// Generates the bind group entries required to bind the tensor to a kernel.
    /// Quantized tensors may use multiple bind groups.
    /// Unquantized tensors should only use a single bind group.
    pub(crate) fn bind_group_entries(&self) -> RVec<BindGroupEntry> {
        assert!(self.device().is_gpu());
        let storage_guard = self.storage();
        let storage = storage_guard
            .as_ref()
            .unwrap_or_else(|| panic!("Storage missing for {:?}", self.id()));
        let gpu_buf = storage.try_gpu().unwrap();
        let handle = gpu_buf.inner().handle;
        self.segments()
            .iter()
            .fold(rvec![], |mut entries, segment| {
                let (offset, size) = (segment.offset, segment.size);
                entries.push(BindGroupEntry {
                    handle,
                    offset,
                    size: Some(size),
                });
                entries
            })
    }

    /// # Segments
    ///
    /// In Piston, a tensor may be split into multiple segments.
    /// This is due to our quantization scheme allowing multiple quantized components to be packed
    /// and stored in a single tensor.
    pub(crate) fn segments(&self) -> RVec<BufferSegment> {
        self.dtype().segments(self.shape().numel())
    }

    /// Converts the tensor into a 1D vector.
    ///
    /// The 1D vector contains the data from the tensor, as it was laid out in memory.
    #[maybe_async]
    pub async fn to_vec<T: TensorDType>(&self) -> anyhow::Result<Vec<T>> {
        ensure_resolved!(self);
        let storage_guard = self.storage();
        let buffer = storage_guard.as_ref().unwrap().try_cpu()?;
        let slice = buffer.to_slice::<T>(self.shape());
        Ok(slice.to_vec())
    }

    pub(crate) fn execution_order(&self) -> Vec<&Tensor> {
        let mut done = BitVec::<u32>::repeat(false, self.id().0 + 1);
        let mut pending = BitVec::<u32>::repeat(false, self.id().0 + 1);
        let mut order = Vec::new();

        let mut stack: Vec<(&Tensor, usize)> = vec![(self, 0)];
        while let Some((cur_t, cur_src)) = stack.pop() {
            let all_deps_done = cur_src == cur_t.op().srcs().len();

            if all_deps_done {
                done.set(cur_t.id().0, true);
                pending.set(cur_t.id().0, false);
                order.push(cur_t);
                continue;
            }

            let (srcs_with_deps, srcs_without_deps): (Vec<_>, Vec<_>) = cur_t
                .op()
                .srcs()
                .iter()
                .partition(|s| s.op().srcs().is_empty());

            let all_srcs = srcs_with_deps
                .into_iter()
                .chain(srcs_without_deps)
                .collect::<RVec<_>>();

            let precursor: &Tensor = all_srcs[cur_src];
            let precursor_id = precursor.id().0;

            if done[precursor_id] {
                stack.push((cur_t, cur_src + 1));
            } else if pending[precursor_id] {
                panic!(
                    "Cycle detected whilst computing topological order: {:?}. Try plotting with feature `plotting`.",
                    precursor_id
                );
            } else {
                pending.set(precursor_id, true);
                stack.push((cur_t, cur_src));
                stack.push((precursor, 0));
            }
        }

        order
    }

    #[maybe_async]
    pub async fn cpu_apply(self, dst: Tensor) -> Option<Tensor> {
        cpu::apply_operation(self.op().clone(), dst).await.ok()
    }

    fn gpu_compile_key_for_op<'a>(
        &'a self,
        op: &'a LazyOp,
        can_inplace: bool,
        uniform: &mut CpuUniform,
    ) -> Option<ComputeCompileKey<'a>> {
        match op {
            LazyOp::Binary(b) => b.create_gpu_compile_key(self, can_inplace, uniform).ok(),
            LazyOp::Cast(c) => c.create_gpu_compile_key(self, can_inplace, uniform).ok(),
            LazyOp::Matmul(m) => m.create_gpu_compile_key(self, can_inplace, uniform).ok(),
            LazyOp::Softmax(s) => s.create_gpu_compile_key(self, can_inplace, uniform).ok(),
            LazyOp::RoPE(r) => r.create_gpu_compile_key(self, can_inplace, uniform).ok(),
            LazyOp::Alibi(a) => a.create_gpu_compile_key(self, can_inplace, uniform).ok(),
            LazyOp::Unary(u) => u.create_gpu_compile_key(self, can_inplace, uniform).ok(),
            LazyOp::Reindex(r) => r.create_gpu_compile_key(self, can_inplace, uniform).ok(),
            LazyOp::Concat(c) => c.create_gpu_compile_key(self, can_inplace, uniform).ok(),
            LazyOp::Norm(n) => n.create_gpu_compile_key(self, can_inplace, uniform).ok(),
            LazyOp::Affine(a) => a.create_gpu_compile_key(self, can_inplace, uniform).ok(),
            LazyOp::Cmp(c) => c.create_gpu_compile_key(self, can_inplace, uniform).ok(),
            LazyOp::Powf(p) => p.create_gpu_compile_key(self, can_inplace, uniform).ok(),
            LazyOp::WhereCond(w) => w.create_gpu_compile_key(self, can_inplace, uniform).ok(),
            LazyOp::Conv(c) => c.create_gpu_compile_key(self, can_inplace, uniform).ok(),
            LazyOp::Select(i) => i.create_gpu_compile_key(self, can_inplace, uniform).ok(),
            LazyOp::IndexWrite(i) => i.create_gpu_compile_key(self, can_inplace, uniform).ok(),
            LazyOp::IndexAdd(i) => i.create_gpu_compile_key(self, can_inplace, uniform).ok(),
            LazyOp::ScatterAdd(s) => s.create_gpu_compile_key(self, can_inplace, uniform).ok(),
            LazyOp::Trilu(t) => t.create_gpu_compile_key(self, can_inplace, uniform).ok(),
            LazyOp::Cache(c) => c.create_gpu_compile_key(self, can_inplace, uniform).ok(),
            LazyOp::Reduce(s) => s.create_gpu_compile_key(self, can_inplace, uniform).ok(),
            LazyOp::Detach(d) => self.gpu_compile_key_for_op(d, can_inplace, uniform),
            LazyOp::Gather(g) => g.create_gpu_compile_key(self, can_inplace, uniform).ok(),
            LazyOp::FillConstant(f) => f.create_gpu_compile_key(self, can_inplace, uniform).ok(),
            LazyOp::FillRandn(f) => f.create_gpu_compile_key(self, can_inplace, uniform).ok(),
            LazyOp::Bernoulli(b) => b.create_gpu_compile_key(self, can_inplace, uniform).ok(),
            LazyOp::Arange(a) => a.create_gpu_compile_key(self, can_inplace, uniform).ok(),
            LazyOp::Copy(_) | LazyOp::View(_) | LazyOp::Const => None,
        }
    }

    pub fn gpu_compile_key(
        &self,
        can_inplace: bool,
        uniform: &mut CpuUniform,
    ) -> Option<GpuCompileKey> {
        match self.op() {
            LazyOp::Copy(c) => Some(GpuCompileKey::Copy(c.create_gpu_compile_key())),
            _ => self
                .gpu_compile_key_for_op(self.op(), can_inplace, uniform)
                .map(GpuCompileKey::Compute),
        }
    }

    pub fn compile_gpu<'a>(
        &'a self,
        gpu_compile_key: &GpuCompileKey<'a>,
        gpu_device: &'a WgpuDevice,
        debug: bool,
    ) -> Option<Compiled> {
        match gpu_compile_key {
            GpuCompileKey::Copy(_) => {
                if let LazyOp::Copy(c) = self.op() {
                    c.compile_gpu().ok()
                } else {
                    None
                }
            }
            GpuCompileKey::Compute(compute_key) => {
                compile_gpu_for_op(self.op(), compute_key, gpu_device, debug).map(Compiled::Compute)
            }
        }
    }

    #[maybe_async]
    async fn resolve_cpu(self) -> Result<Tensor, TensorError> {
        let mut tensor = self.clone();
        let execution_order = self.execution_order();

        for t in execution_order.into_iter() {
            log::debug!("Running: {:?}", t.op().name());
            assert!(t.device().is_cpu());
            if t.resolved() {
                continue;
            }
            tensor = tensor.cpu_apply(t.clone()).await.unwrap();
        }

        Ok(tensor.clone())
    }

    /// Applies the pending graph to the tensor.
    #[maybe_async]
    async fn apply_pending_graph(&self) -> Result<Tensor, TensorError> {
        if self.resolved() {
            return Ok(self.clone());
        }

        match self.device() {
            Device::GPU(gpu_device) => {
                #[cfg(target_arch = "wasm32")]
                {
                    Box::pin(gpu_device.sync_tensors_graph(vec![&self])).await?;
                }
                #[cfg(not(target_arch = "wasm32"))]
                {
                    gpu_device.sync_tensors_graph(vec![&self])?;
                }
                Ok(self.clone())
            }
            Device::CPU => {
                #[cfg(target_arch = "wasm32")]
                {
                    Box::pin(self.clone().resolve_cpu()).await?;
                }
                #[cfg(not(target_arch = "wasm32"))]
                {
                    self.clone().resolve_cpu()?;
                }
                Ok(self.clone())
            }
        }
    }

    #[maybe_async]
    async fn to_gpu(&self, dst_device: &Device) -> Result<Tensor, TensorError> {
        ensure_resolved!(self);
        let storage_guard = self.storage();
        let cpu_buf = storage_guard
            .as_ref()
            .ok_or(TensorError::TransferError)?
            .try_cpu()?;
        let gpu_buf = cpu_buf.to_device(dst_device)?;

        let wgpu_device = dst_device.try_gpu()?;
        Ok(Tensor::new_impl(
            LazyOp::Const,
            self.view.clone(),
            Some(Storage::GPU(gpu_buf)),
            Device::GPU(wgpu_device.clone()),
            false,
        ))
    }

    #[maybe_async]
    pub async fn deep_clone(&self) -> Self {
        ensure_resolved!(self);
        let storage_guard = self.storage();
        let storage = storage_guard.as_ref().unwrap();
        let cloned_storage = storage.deep_clone(self.device()).unwrap();
        Tensor::new_impl(
            LazyOp::Const,
            self.view.clone(),
            Some(cloned_storage),
            self.device.clone(),
            false,
        )
    }

    #[maybe_async]
    async fn to_cpu(&self) -> Result<Tensor, TensorError> {
        ensure_resolved!(self);

        if self.device().is_cpu() {
            return Ok(self.clone());
        }
        let storage_guard = self.storage();
        let gpu_buf = storage_guard
            .as_ref()
            .ok_or(TensorError::TransferError)?
            .try_gpu()?;
        let cpu_buf = gpu_buf.to_cpu(&self.device).await?;

        Ok(Tensor::new_impl(
            LazyOp::Const,
            self.view.clone(),
            Some(Storage::CPU(cpu_buf)),
            Device::CPU,
            false,
        ))
    }

    /// Transfers the tensor to the specified device.
    ///
    /// If the tensor is already on the specified device, it will be returned as-is,
    /// and the underlying storage will not be copied.
    /// If the tensor is on a different device, it will be copied to the specified device.
    #[maybe_async]
    pub async fn to(&self, device: &Device) -> Result<Tensor, TensorError> {
        match (self.device(), device) {
            (Device::GPU(_), Device::CPU) => self.to_cpu().await,
            (Device::CPU, Device::GPU(_)) => self.to_gpu(device).await,
            _ => Ok(self.clone()),
        }
    }

    #[cfg(feature = "pyo3")]
    pub fn to_py<'s, 'p: 's, T: TensorDType + numpy::Element>(
        &'s self,
        py: &'p pyo3::Python<'p>,
    ) -> &'s PyArrayDyn<T> {
        use numpy::PyArray;
        assert!(
            self.device().is_cpu(),
            "Cannot convert non-CPU tensor to numpy array"
        );
        PyArray::from_owned_array(*py, self.deep_clone().into_ndarray::<T>())
    }
}

pub fn compile_gpu_for_op(
    op: &LazyOp,
    gpu_compile_key: &ComputeCompileKey,
    gpu_device: &WgpuDevice,
    debug: bool,
) -> Option<CompiledOp> {
    match op {
        LazyOp::Binary(b) => b.compile_gpu(gpu_compile_key, gpu_device, debug).ok(),
        LazyOp::Cast(c) => c.compile_gpu(gpu_compile_key, gpu_device, debug).ok(),
        LazyOp::Matmul(m) => m.compile_gpu(gpu_compile_key, gpu_device, debug).ok(),
        LazyOp::Softmax(s) => s.compile_gpu(gpu_compile_key, gpu_device, debug).ok(),
        LazyOp::RoPE(r) => r.compile_gpu(gpu_compile_key, gpu_device, debug).ok(),
        LazyOp::Alibi(a) => a.compile_gpu(gpu_compile_key, gpu_device, debug).ok(),
        LazyOp::Unary(u) => u.compile_gpu(gpu_compile_key, gpu_device, debug).ok(),
        LazyOp::Reindex(r) => r.compile_gpu(gpu_compile_key, gpu_device, debug).ok(),
        LazyOp::Concat(c) => c.compile_gpu(gpu_compile_key, gpu_device, debug).ok(),
        LazyOp::Norm(n) => n.compile_gpu(gpu_compile_key, gpu_device, debug).ok(),
        LazyOp::Affine(a) => a.compile_gpu(gpu_compile_key, gpu_device, debug).ok(),
        LazyOp::Cmp(c) => c.compile_gpu(gpu_compile_key, gpu_device, debug).ok(),
        LazyOp::Powf(p) => p.compile_gpu(gpu_compile_key, gpu_device, debug).ok(),
        LazyOp::WhereCond(w) => w.compile_gpu(gpu_compile_key, gpu_device, debug).ok(),
        LazyOp::Conv(c) => c.compile_gpu(gpu_compile_key, gpu_device, debug).ok(),
        LazyOp::Select(i) => i.compile_gpu(gpu_compile_key, gpu_device, debug).ok(),
        LazyOp::IndexWrite(i) => i.compile_gpu(gpu_compile_key, gpu_device, debug).ok(),
        LazyOp::IndexAdd(i) => i.compile_gpu(gpu_compile_key, gpu_device, debug).ok(),
        LazyOp::ScatterAdd(s) => s.compile_gpu(gpu_compile_key, gpu_device, debug).ok(),
        LazyOp::Trilu(t) => t.compile_gpu(gpu_compile_key, gpu_device, debug).ok(),
        LazyOp::Cache(c) => c.compile_gpu(gpu_compile_key, gpu_device, debug).ok(),
        LazyOp::Reduce(s) => s.compile_gpu(gpu_compile_key, gpu_device, debug).ok(),
        LazyOp::Detach(d) => compile_gpu_for_op(d, gpu_compile_key, gpu_device, debug),
        LazyOp::Gather(g) => g.compile_gpu(gpu_compile_key, gpu_device, debug).ok(),
        LazyOp::FillConstant(f) => f.compile_gpu(gpu_compile_key, gpu_device, debug).ok(),
        LazyOp::FillRandn(f) => f.compile_gpu(gpu_compile_key, gpu_device, debug).ok(),
        LazyOp::Bernoulli(b) => b.compile_gpu(gpu_compile_key, gpu_device, debug).ok(),
        LazyOp::Arange(a) => a.compile_gpu(gpu_compile_key, gpu_device, debug).ok(),
        LazyOp::View(_) | LazyOp::Const => None,
        LazyOp::Copy(_) => panic!("Copy should not have a gpu_compile_key"),
    }
}

#[cfg(feature = "pyo3")]
impl<T: TensorDType + numpy::Element> From<&PyArrayDyn<T>> for Tensor {
    fn from(array: &PyArrayDyn<T>) -> Self {
        Self::from(array.to_owned_array())
    }
}

#[cfg(feature = "testing")]
#[derive(Default)]
struct CloseStats<T> {
    total_error: T,
    max_abs_error: T,
    max_abs_error_idxs: Option<Vec<usize>>,
    element_count: usize,
    fail_count: usize,
    atol: T,
    rtol: T,
}

#[cfg(feature = "testing")]
impl<T: TensorDType + Default + num_traits::Float> CloseStats<T> {
    fn new(atol: T, rtol: T) -> Self {
        Self {
            atol,
            rtol,
            ..Default::default()
        }
    }

    fn update(&mut self, a: &T, b: &T, index: ndarray::IxDyn) {
        let abs_diff = (*a - *b).abs();
        self.total_error = self.total_error + abs_diff;
        self.element_count += 1;

        if abs_diff > self.max_abs_error {
            self.max_abs_error = abs_diff;
            self.max_abs_error_idxs = Some(index.slice().into());
        }

        if !self.is_close(a, b, abs_diff) {
            self.fail_count += 1;
        }
    }

    fn avg_error(&self) -> T {
        self.total_error / T::from(self.element_count).expect("Failed to convert")
    }

    fn is_close(&self, a: &T, b: &T, abs_diff: T) -> bool {
        (a.is_nan() && b.is_nan())
            || (a.is_infinite() && b.is_infinite() && a.signum() == b.signum())
            || abs_diff <= self.atol + self.rtol * b.abs()
    }
}

#[cfg(feature = "testing")]
impl Tensor {
    pub fn read_npy<T, P>(path: P, device: &Device, requires_grad: bool) -> Result<Self>
    where
        T: TensorDType + npyz::Deserialize,
        P: AsRef<Path>,
    {
        Self::from_npy_bytes::<T>(&std::fs::read(path)?, device, requires_grad)
    }

    pub fn write_npy<T, P>(&self, path: P) -> anyhow::Result<()>
    where
        T: TensorDType + npyz::Serialize,
        P: AsRef<Path>,
    {
        let mut out_buf = vec![];
        let shape = self
            .shape()
            .to_vec()
            .iter()
            .map(|x| *x as u64)
            .collect::<Vec<_>>();
        let mut writer = {
            npyz::WriteOptions::new()
                .dtype(self.dtype().into())
                .shape(&shape)
                .writer(&mut out_buf)
                .begin_nd()?
        };
        let ndarray = self.to_ndarray_view::<T>();
        ndarray.iter().for_each(|x| {
            writer.push(x).unwrap();
        });
        writer.finish()?;
        std::fs::write(path, out_buf)?;
        Ok(())
    }

    pub fn from_npy_bytes<T: TensorDType + npyz::Deserialize>(
        bytes: &[u8],
        device: &Device,
        requires_grad: bool,
    ) -> Result<Self> {
        let reader = npyz::NpyFile::new(bytes)?;
        let shape = reader
            .shape()
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<_>>()
            .into();
        let data = reader.into_vec::<T>()?;
        Ok(Tensor::from_data_impl(
            data,
            shape,
            device.clone(),
            requires_grad,
        ))
    }

    pub fn into_ndarray<T: TensorDType>(self) -> ArrayD<T> {
        self.to_ndarray_view().into_owned()
    }

    pub fn to_ndarray_view<T: TensorDType>(&self) -> ArrayViewD<T> {
        ensure_resolved_sync!(self);
        assert!(self.device().is_cpu());
        assert!(self.dtype() == T::dtype());
        let shape = self.shape().to_vec();
        if self.num_bytes() != 0 {
            let storage_guard = self.storage();
            let buffer = storage_guard.as_ref().unwrap().try_cpu().unwrap();
            let (ptr, _) = buffer.inner().into_raw_parts();
            unsafe { ArrayViewD::from_shape_ptr(shape, ptr as *const T) }
        } else {
            ArrayViewD::from_shape(shape, &[]).unwrap()
        }
    }

    pub fn all_close<T>(&self, other: &Self, atol: T, rtol: T) -> anyhow::Result<()>
    where
        T: TensorDType + std::fmt::Display + num_traits::Float + Default,
    {
        if self.shape() != other.shape() {
            anyhow::bail!("Shape mismatch {:?} != {:?}", self.shape(), other.shape())
        }
        assert!(
            self.dtype() == other.dtype(),
            "DType mismatch {:?} != {:?}",
            self.dtype(),
            other.dtype()
        );
        assert!(
            self.dtype() == T::dtype(),
            "DType mismatch {:?} != {:?}",
            self.dtype(),
            T::dtype()
        );

        let self_nd = self.to_ndarray_view::<T>();
        let other_nd = other.to_ndarray_view::<T>();

        let mut stats = CloseStats::new(atol, rtol);
        ndarray::indices_of(&self_nd).into_iter().for_each(|idx| {
            let (a, b) = (self_nd[&idx], other_nd[&idx]);
            stats.update(&a, &b, idx);
        });

        let idx_fmt = stats.max_abs_error_idxs.as_ref();
        if stats.fail_count > 0 {
            anyhow::bail!(
                "\x1b[1;31m{} samples not close \x1b[0m - AVGE={} MAE={} at {:?}",
                stats.fail_count,
                stats.avg_error(),
                stats.max_abs_error,
                idx_fmt
            );
        } else {
            println!(
                "\x1b[1;32mAll close \x1b[0m - AVGE={} MAE={} at {:?}",
                stats.avg_error(),
                stats.max_abs_error,
                idx_fmt
            );
            Ok(())
        }
    }
}

impl<T: TensorDType> From<ArrayD<T>> for Tensor {
    fn from(it: ArrayD<T>) -> Self {
        if it.as_slice().is_some() {
            let layout = std::alloc::Layout::from_size_align(
                it.len() * std::mem::size_of::<T>(),
                std::mem::align_of::<T>(),
            )
            .unwrap();
            let shape = it.shape().to_vec().into();
            let stride = Stride::from(&shape);
            let vec = it.into_raw_vec().into_boxed_slice();
            let ptr = Box::into_raw(vec) as *mut u8;

            let raw_buf = RawCPUBuffer::new(ptr, layout);
            let meta = StorageView::new(shape, T::dtype(), stride);
            Tensor::new_impl(
                LazyOp::Const,
                meta,
                Some(Storage::CPU(CPUBuffer::new(raw_buf))),
                Device::CPU,
                false,
            )
        } else {
            panic!("Cannot convert numpy array with non-contiguous memory layout to tensor");
        }
    }
}

macro_rules! bin_trait {
    ($trait:ident, $fn1:ident, $mul:expr, $add:expr) => {
        impl std::ops::$trait<Tensor> for Tensor {
            type Output = Result<Tensor>;

            fn $fn1(self, rhs: Tensor) -> Self::Output {
                Tensor::$fn1(self, rhs)
            }
        }

        impl std::ops::$trait<Tensor> for Result<Tensor> {
            type Output = Result<Tensor>;

            fn $fn1(self, rhs: Tensor) -> Self::Output {
                Tensor::$fn1(self?, rhs)
            }
        }

        impl std::ops::$trait<Result<Tensor>> for Tensor {
            type Output = Result<Tensor>;

            fn $fn1(self, rhs: Result<Tensor>) -> Self::Output {
                Tensor::$fn1(self, rhs?)
            }
        }

        impl std::ops::$trait<f32> for Tensor {
            type Output = Result<Self>;

            fn $fn1(self, rhs: f32) -> Self::Output {
                self.affine($mul(rhs), $add(rhs))
            }
        }
    };
}

bin_trait!(Add, add, |_| 1., |v| v);
bin_trait!(Sub, sub, |_| 1., |v: f32| -v);
bin_trait!(Mul, mul, |v| v, |_| 0.);
bin_trait!(Div, div, |v| 1. / v, |_| 0.);

impl std::ops::Add<Tensor> for f32 {
    type Output = Result<Tensor>;

    fn add(self, rhs: Tensor) -> Self::Output {
        rhs + self
    }
}

impl std::ops::Mul<Tensor> for f32 {
    type Output = Result<Tensor>;

    fn mul(self, rhs: Tensor) -> Self::Output {
        rhs * self
    }
}

impl std::ops::Sub<Tensor> for f32 {
    type Output = Result<Tensor>;

    fn sub(self, rhs: Tensor) -> Self::Output {
        rhs.affine(-1., self)
    }
}

impl std::ops::Div<Tensor> for f32 {
    type Output = Result<Tensor>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Tensor) -> Self::Output {
        rhs.recip()? * self
    }
}

impl safetensors::View for &Tensor {
    fn dtype(&self) -> safetensors::Dtype {
        match Tensor::dtype(self) {
            DType::F32 => safetensors::Dtype::F32,
            DType::U32 => safetensors::Dtype::U32,
            DType::I32 => safetensors::Dtype::I32,
            DType::F16 => safetensors::Dtype::F16,
            DType::Q8_0F(_) | DType::Q8_0H(_) => safetensors::Dtype::U8,
            DType::BF16 => safetensors::Dtype::BF16,
            DType::Q4_KF(_) | DType::Q4_KH(_) => todo!(),
        }
    }

    fn shape(&self) -> &[usize] {
        Tensor::shape(self).inner()
    }

    fn data(&self) -> Cow<'_, [u8]> {
        assert!(
            self.device().is_cpu(),
            "Cannot convert non-CPU tensor to safetensors"
        );
        let storage_guard = self.storage();
        let buffer = storage_guard.as_ref().unwrap().try_cpu().unwrap();
        let (ptr, _) = buffer.inner().into_raw_parts();
        Cow::from(unsafe { std::slice::from_raw_parts(ptr, self.num_bytes()) })
    }

    fn data_len(&self) -> usize {
        self.num_bytes()
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use crate::{rvec, shape, Device, Tensor};

    #[test]
    fn has_nan_works() {
        let device = Device::request_device(crate::DeviceRequest::GPU).unwrap();
        let rand = Tensor::randn::<f32>(0., 1., shape![1, 1500, 384], device.clone()).unwrap();
        let nans = Tensor::from_data(vec![f32::NAN; 1500 * 384], shape![1, 1500, 384], device);

        let bingo = Tensor::cat(rvec![rand, nans], 2).unwrap();

        let result = bingo.to(&Device::CPU).unwrap();
        println!("RESULT: {:?}", result);
        assert!(result.has_nan::<f32>());
    }
}
