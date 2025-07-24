use crate::dispatch::Value;
use crate::dispatch::{
    dispatch_operator, BoxedKernel, DispatchKeySet, KernelFunction, OperatorHandle,
    OperatorHandlerRegistration, OperatorRegistration, StackValue,
};
use crate::gpu::{BindGroupEntry, CpuUniform, WgpuDevice};
use crate::{
    cpu, get_current_scope, ops::*, rvec, BufferSegment, CPUBuffer, Compiled, CompiledOp,
    ComputeCompileKey, DType, Device, DeviceStorage, Dim, Dims, GPUOperation, GpuCompileKey,
    InvariantError, LazyGraphExecutorError, LazyOp, Operation, OperationError, RVec, RawCPUBuffer,
    ScopePusher, Shape, Storage, Stride, TensorDType, TensorId,
};
#[cfg(not(feature = "debug"))]
use crate::{BufferDescriptor, BufferUsagesExt, GPUBuffer};
use anyhow::Result;
use bitvec::prelude::*;
use derive_new::new;
use maybe_async::maybe_async;
use npyz::WriterBuilder;
use num_traits::AsPrimitive;
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use paste::paste;
use std::borrow::Cow;
use std::io::{BufRead, Seek};
use std::mem::ManuallyDrop;
use std::ops::Bound;
use std::path::Path;
use std::str::FromStr;
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
    #[error(transparent)]
    LazyGraphExecutorError(#[from] Box<LazyGraphExecutorError>),
}

/// A multi-dimensional array of data.
///
/// A tensor is a lazy representation of an operation. The nodes required to compute it's
/// value and it's own value will not be computed until `resolve` is called.
#[derive(Clone)]
pub struct OpTensor {
    pub(crate) inner: Arc<Inner>,
}

unsafe impl Send for OpTensor {}

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

impl OpTensor {
    fn register_with_device(&self) {
        if let Device::GPU(inner) = self.device() {
            log::trace!(
                "Attempting to register tensor {:?} with op {:?} requires_grad={}",
                self.id(),
                self.op().name(),
                self.requires_grad()
            );
            inner.register_tensor(self);
        }
    }

    pub fn new(
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

    pub fn wrap(self) -> Tensor {
        Tensor::wrap(self)
    }

    pub(crate) fn full_impl<T: TensorDType + num_traits::AsPrimitive<f32>, S: Into<Shape>>(
        shape: S,
        value: T,
        device: &Device,
        requires_grad: bool,
    ) -> Result<Self> {
        let shape = shape.into();
        let meta = StorageView {
            shape: shape.clone(),
            dtype: T::dtype(),
            stride: Stride::from(&shape.clone()),
        };
        Ok(Self::new(
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

    pub fn full<T: TensorDType + num_traits::AsPrimitive<f32>, S: Into<Shape>>(
        shape: S,
        value: T,
        device: &Device,
        requires_grad: bool,
    ) -> Result<Self> {
        let shape = shape.into();
        if device.is_cpu() {
            let mut data = Vec::with_capacity(shape.numel());
            data.resize(shape.numel(), value);
            Ok(Self::from_data(data, shape, device.clone(), requires_grad))
        } else {
            Self::full_impl::<T, _>(shape, value, device, requires_grad)
        }
    }

    #[track_caller]
    fn lazy(op: LazyOp, meta: StorageView, device: Device, requires_grad: bool) -> Self {
        op.check_invariants();
        Self::new(op, meta, None, device, requires_grad)
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

    // Helper method to handle broadcasting for multiple tensors
    fn broadcast_tensors(tensors: RVec<Self>) -> Result<RVec<Self>, OperationError> {
        if tensors.is_empty() {
            return Ok(rvec![]);
        }

        let shapes: RVec<Shape> = tensors.iter().map(|t| t.shape().clone()).collect();
        let shape_refs: RVec<&Shape> = shapes.iter().collect();
        let broadcasted = Shape::multi_broadcast(&shape_refs);

        if broadcasted.is_none() {
            return Err(InvariantError::BroadcastingFailed(shapes.to_vec()).into());
        }

        let broadcasted = broadcasted.unwrap();
        let mut result = RVec::with_capacity(tensors.len());

        for (tensor, shape) in tensors.into_iter().zip(shapes.iter()) {
            if shape != &broadcasted {
                result.push(broadcast_to(tensor, broadcasted.clone())?);
            } else {
                result.push(tensor);
            }
        }

        Ok(result)
    }

    // Convenience method for binary operations
    fn broadcast_for_binary_op(self, other: Self) -> Result<(Self, Self), OperationError> {
        let mut result = Self::broadcast_tensors(rvec![self, other])?;
        let rhs = result.pop().unwrap();
        let lhs = result.pop().unwrap();
        Ok((lhs, rhs))
    }

    // Convenience method for ternary operations
    fn broadcast_for_ternary_op(
        self,
        tensor1: Self,
        tensor2: Self,
    ) -> Result<(Self, Self, Self), OperationError> {
        let mut result = Self::broadcast_tensors(rvec![self, tensor1, tensor2])?;
        let t2 = result.pop().unwrap();
        let t1 = result.pop().unwrap();
        let input = result.pop().unwrap();
        Ok((input, t1, t2))
    }
}

impl std::fmt::Debug for OpTensor {
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

impl PartialEq for OpTensor {
    fn eq(&self, other: &Self) -> bool {
        self.inner.id == other.inner.id
    }
}

impl std::ops::Deref for OpTensor {
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
    invalidated: Arc<RwLock<bool>>,
    storage: ManuallyDrop<Arc<RwLock<Option<Storage>>>>,
    grad: Arc<RwLock<Option<Tensor>>>,
    #[cfg(not(feature = "debug"))]
    debug_tensor: Arc<RwLock<Option<OpTensor>>>,
    inplace: RwLock<bool>,
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
            grad: Arc::new(RwLock::new(None)),
            requires_grad,
            invalidated: Arc::new(RwLock::new(false)),
            #[cfg(not(feature = "debug"))]
            debug_tensor: Arc::new(RwLock::new(None)),
            inplace: RwLock::new(false),
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
            grad: Arc::new(RwLock::new(None)),
            requires_grad,
            invalidated: Arc::new(RwLock::new(false)),
            #[cfg(not(feature = "debug"))]
            debug_tensor: Arc::new(RwLock::new(None)),
            inplace: RwLock::new(false),
        }
    }
}

impl OpTensor {
    pub fn id(&self) -> TensorId {
        self.inner.id
    }

    pub fn storage_view(&self) -> &StorageView {
        &self.view
    }

    pub fn dim(&self) -> usize {
        self.view.shape.dim()
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

    pub fn storage(&'_ self) -> RwLockReadGuard<'_, Option<Storage>> {
        self.inner.storage.read()
    }

    pub fn resolved(&self) -> bool {
        self.storage().is_some() || *self.inner.invalidated.read()
    }

    pub fn invalidated(&self) -> bool {
        *self.inner.invalidated.read()
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

    pub fn is_inplace(&self) -> bool {
        *self.inner.inplace.read()
    }

    // TODO: Get rid of these
    /// Sets the content of the inner tensor, this does not require a mutable reference as inner
    /// mutability is used.
    pub fn set_sync(&self, src: Self) -> Result<()> {
        if self.same_storage(&src) {
            panic!("cannot set a variable to a tensor that is derived from its value");
        }
        if self.shape() != src.shape() {
            panic!(
                "shape mismatch: {:?} != {:?} (target id: {:?}, source id: {:?})",
                self.shape(),
                src.shape(),
                self.id(),
                src.id()
            );
        }
        self.update_storage(Storage::GPU(GPUBuffer {
            inner: src.storage().as_ref().unwrap().try_gpu()?.inner.clone(),
            alignment: self.dtype().size_of(),
            cpu_size: Some(self.num_bytes()),
        }));
        Ok(())
    }

    pub fn set(self, src: Self) -> Self {
        copy(self, src).unwrap()
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
        paste! {
            pub fn $method_name<T1: Into<OpTensor>, T2: Into<OpTensor>, T3: TensorTypeOrScalar<T2>>(
                input: T1,
                other: T3,
            ) -> Result<OpTensor> {
                let input = input.into();
                let device = input.device().clone();
                let (lhs, rhs) = match other.map_tensor(|other| other.into()).tensor_or_scalar() {
                    Ok(TensorTypeOrScalarEnum::Tensor(other)) => {
                        let (lhs, rhs) = input.broadcast_for_binary_op(other)?;
                        (lhs, TensorTypeOrScalarEnum::Tensor(rhs))
                    }
                    Ok(TensorTypeOrScalarEnum::Scalar(other)) => {
                        (input, TensorTypeOrScalarEnum::Scalar(other)) // TODO: this is wrong
                    }
                    Err(e) => return Err(e),
                };
                let binary = Binary::new(lhs, rhs, $op);
                let new_view = binary.compute_view()?;
                Ok(OpTensor::lazy(
                    LazyOp::Binary(binary),
                    new_view,
                    device,
                    false,
                ))
            }

            impl Tensor {
                #[allow(clippy::should_implement_trait)]
                pub fn $method_name<T: TensorTypeOrScalar<Self>>(
                    self,
                    other: T,
                ) -> Result<Self> {
                    $method_name(self, other).map(Self::wrap)
                }

                pub fn [<$method_name _>]<T: TensorTypeOrScalar<Self>>(
                    self,
                    other: T,
                ) -> Result<Self> {
                    Ok(self.wrap_inplace($method_name(self.clone(), other)?))
                }
            }
        }
    };
}

macro_rules! impl_binary_op_tensor_only {
    ($method_name:ident, $op:expr) => {
        paste! {
            pub fn $method_name<T1: Into<OpTensor>, T2: Into<OpTensor>>(
                input: T1,
                other: T2,
            ) -> Result<OpTensor> {
                let input = input.into();
                let other = other.into();
                let device = input.device().clone();
                let (lhs, rhs) = input.broadcast_for_binary_op(other)?;
                let binary = Binary::new(lhs, TensorTypeOrScalarEnum::Tensor(rhs), $op);
                let new_view = binary.compute_view()?;
                Ok(OpTensor::lazy(
                    LazyOp::Binary(binary),
                    new_view,
                    device,
                    false,
                ))
            }

            impl Tensor {
                #[allow(clippy::should_implement_trait)]
                pub fn $method_name<T: Into<OpTensor>>(self, other: T) -> Result<Self> {
                    $method_name(self, other).map(Self::wrap)
                }

                pub fn [<$method_name _>] <
                    T: Into<OpTensor>,
                >(
                    self,
                    other: T,
                ) -> Result<Self> {
                    Ok(self.wrap_inplace($method_name(self.clone(), other)?))
                }
            }
        }
    };
}

macro_rules! impl_ternary_op {
    ($method_name:ident, $op:expr) => {
        paste! {
            pub fn $method_name<T1: Into<OpTensor>, T2: Into<OpTensor>, T3: Into<OpTensor>>(
                input: T1,
                tensor1: T2,
                tensor2: T3,
                value: f32,
            ) -> Result<OpTensor> {
                let input = input.into();
                let tensor1 = tensor1.into();
                let tensor2 = tensor2.into();
                let device = input.device().clone();
                let (input, t1, t2) = input.broadcast_for_ternary_op(tensor1, tensor2)?;
                let ternary = Ternary::new(input, t1, t2, value, $op);
                let new_view = ternary.compute_view()?;
                Ok(OpTensor::lazy(
                    LazyOp::Ternary(ternary),
                    new_view,
                    device,
                    false,
                ))
            }

            impl Tensor {
                #[allow(clippy::should_implement_trait)]
                pub fn $method_name(
                    self,
                    tensor1: Self,
                    tensor2: Self,
                    value: f32,
                ) -> Result<Self> {
                    $method_name(self, tensor1, tensor2, value).map(Self::wrap)
                }

                pub fn [<$method_name _>](
                    self,
                    tensor1: Self,
                    tensor2: Self,
                    value: f32,
                ) -> Result<Self> {
                    Ok(self.wrap_inplace($method_name(self.clone(), tensor1, tensor2, value)?))
                }
            }
        }
    };
}

macro_rules! impl_cmp_op {
    ($method_name:ident, $op:expr) => {
        paste! {
            pub fn $method_name<T1: Into<OpTensor>, T2: Into<OpTensor>, T3: TensorTypeOrScalar<T2>>(
                input: T1,
                other: T3,
            ) -> Result<OpTensor> {
                let input = input.into();
                let device = input.device().clone();
                match other.map_tensor(|other| other.into()).tensor_or_scalar() {
                    Ok(TensorTypeOrScalarEnum::Tensor(other)) => {
                        let (lhs, rhs) = input.broadcast_for_binary_op(other)?;
                        let cmp = Cmp::new(lhs, TensorTypeOrScalarEnum::Tensor(rhs), $op);
                        let new_view = cmp.compute_view()?;
                        Ok(OpTensor::lazy(LazyOp::Cmp(cmp), new_view, device, false))
                    }
                    Ok(TensorTypeOrScalarEnum::Scalar(other)) => {
                        let device = input.device.clone();
                        let cmp = Cmp::new(input, TensorTypeOrScalarEnum::Scalar(other), $op);
                        let new_view = cmp.compute_view()?;
                        Ok(OpTensor::lazy(LazyOp::Cmp(cmp), new_view, device, false))
                    }
                    Err(e) => Err(e),
                }
            }

            impl Tensor {
                #[allow(clippy::should_implement_trait)]
                pub fn $method_name<T: TensorTypeOrScalar<Self>>(
                    self,
                    other: T,
                ) -> Result<Self> {
                    $method_name(self, other).map(Self::wrap)
                }

                pub fn [<$method_name _>]<T: TensorTypeOrScalar<Self>>(
                    self,
                    other: T,
                ) -> Result<Self> {
                    Ok(self.wrap_inplace($method_name(self.clone(), other)?))
                }
            }
        }
    };
}

macro_rules! impl_unary_op {
    ($method_name:ident, $op:expr) => {
        paste! {
            pub fn $method_name<T: Into<OpTensor>>(input: T) -> Result<OpTensor> {
                let input = input.into();
                let device = input.device().clone();
                let unary = Unary::new(input, $op);
                let new_view = unary.compute_view()?;
                Ok(OpTensor::lazy(
                    LazyOp::Unary(unary),
                    new_view,
                    device,
                    false,
                ))
            }

            impl Tensor {
                #[allow(clippy::should_implement_trait)]
                pub fn $method_name(self) -> Result<Self> {
                    $method_name(self).map(Self::wrap)
                }

                pub fn [<$method_name _>](self) -> Result<Self> {
                    Ok(self.wrap_inplace($method_name(self.clone())?))
                }
            }
        }
    };
}

impl_binary_op!(add, BinaryOp::Add);
impl_binary_op!(sub, BinaryOp::Sub);
impl_binary_op!(mul, BinaryOp::Mul);
impl_binary_op!(div, BinaryOp::Div);
impl_binary_op!(pow, BinaryOp::Pow);
impl_binary_op_tensor_only!(maximum, BinaryOp::Maximum);
impl_binary_op_tensor_only!(minimum, BinaryOp::Minimum);

impl_ternary_op!(addcdiv, TernaryOp::Addcdiv);
impl_ternary_op!(addcmul, TernaryOp::Addcmul);

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

pub fn cast<T: Into<OpTensor>>(input: T, dst_dtype: DType) -> Result<OpTensor> {
    let input = input.into();
    if input.dtype() == dst_dtype {
        return Ok(input);
    }

    let device = input.device().clone();
    let cast = Cast::new(input, dst_dtype);
    let new_view = cast.compute_view()?;
    Ok(OpTensor::lazy(LazyOp::Cast(cast), new_view, device, false))
}

/// Cast a tensor to full precision (IEEE 754 32-bit floating point).
pub fn float<T: Into<OpTensor>>(input: T) -> Result<OpTensor> {
    cast(input, DType::F32)
}

/// Cast a tensor to half precision (IEEE 754 16-bit floating point).
pub fn half<T: Into<OpTensor>>(input: T) -> Result<OpTensor> {
    cast(input, DType::F16)
}

pub fn group_norm<T1: Into<OpTensor>, T2: Into<OpTensor>, T3: Into<OpTensor>>(
    input: T1,
    num_groups: usize,
    weight: Option<T2>,
    bias: Option<T3>,
    eps: f32,
) -> Result<OpTensor> {
    let input = input.into();
    let weight = weight.map(|w| w.into());
    let bias = bias.map(|b| b.into());
    let device = input.device().clone();
    let group_norm = GroupNorm::new(Norm::new(input, weight, bias, eps), num_groups);
    let norm_op = NormOp::GroupNorm(group_norm);
    let new_view = norm_op.compute_view()?;
    Ok(OpTensor::lazy(
        LazyOp::Norm(norm_op),
        new_view,
        device,
        false,
    ))
}

pub fn layer_norm<T1: Into<OpTensor>, T2: Into<OpTensor>, T3: Into<OpTensor>>(
    input: T1,
    weight: Option<T2>,
    bias: Option<T3>,
    eps: f32,
) -> Result<OpTensor> {
    let input = input.into();
    let weight = weight.map(|w| w.into());
    let bias = bias.map(|b| b.into());
    let device = input.device().clone();
    let layer_norm = Norm::new(input, weight, bias, eps);
    let op = NormOp::LayerNorm(layer_norm);
    let new_view = op.compute_view()?;
    Ok(OpTensor::lazy(LazyOp::Norm(op), new_view, device, false))
}

pub fn rms_norm<T1: Into<OpTensor>, T2: Into<OpTensor>>(
    input: T1,
    weight: Option<T2>,
    eps: f32,
) -> Result<OpTensor> {
    let input = input.into();
    let weight = weight.map(|w| w.into());
    let device = input.device().clone();
    let rms = Norm::new(input, weight, None, eps);
    let op = NormOp::RMSNorm(rms);
    let new_view = op.compute_view()?;
    Ok(OpTensor::lazy(LazyOp::Norm(op), new_view, device, false))
}

pub fn conv1d<T1: Into<OpTensor>, T2: Into<OpTensor>, T3: Into<OpTensor>>(
    input: T1,
    weight: T2,
    bias: Option<T3>,
    stride: usize,
    padding: usize,
) -> Result<OpTensor> {
    let input = input.into();
    let device = input.device().clone();
    let conv = Conv::new(
        input,
        weight.into(),
        bias.map(|b| b.into()),
        stride,
        padding,
    );
    let new_view = conv.compute_view()?;
    Ok(OpTensor::lazy(LazyOp::Conv(conv), new_view, device, false))
}

pub fn softmax<T: Into<OpTensor>, D: Dim>(input: T, dim: D) -> Result<OpTensor> {
    let input = input.into();
    let device = input.device().clone();
    let dim = dim.to_index(input.shape(), "softmax")?;
    let softmax = Softmax::new(input, dim);
    let new_view = softmax.compute_view()?;
    Ok(OpTensor::lazy(
        LazyOp::Softmax(softmax),
        new_view,
        device,
        false,
    ))
}

fn rope_impl<T: Into<OpTensor>>(
    input: T,
    dim: usize,
    base: f32,
    offset: usize,
    is_backward: bool,
) -> Result<OpTensor> {
    let input = input.into();
    let device = input.device().clone();
    let rope = RoPE::new(input, dim, base, offset, is_backward);
    let new_view = rope.compute_view()?;
    Ok(OpTensor::lazy(LazyOp::RoPE(rope), new_view, device, false))
}

pub fn rope<T: Into<OpTensor>>(input: T, dim: usize, base: f32, offset: usize) -> Result<OpTensor> {
    rope_impl(input, dim, base, offset, false)
}

pub(crate) fn rope_backward<T: Into<OpTensor>>(
    input: T,
    dim: usize,
    base: f32,
    offset: usize,
) -> Result<OpTensor> {
    rope_impl(input, dim, base, offset, true)
}

pub fn alibi<T: Into<OpTensor>>(input: T, max_bias: f32) -> Result<OpTensor> {
    let input = input.into();
    let device = input.device().clone();
    let alibi = Alibi::new(input, max_bias);
    let new_view = alibi.compute_view()?;
    Ok(OpTensor::lazy(
        LazyOp::Alibi(alibi),
        new_view,
        device,
        false,
    ))
}

//TODO (vinhowe): figure out how to make this interface more like pytorch
pub fn matmul<T1: Into<OpTensor>, T2: Into<OpTensor>>(
    input: T1,
    rhs: T2,
    trans_lhs: bool,
    trans_rhs: bool,
) -> Result<OpTensor> {
    let input = input.into();
    let rhs = rhs.into();
    let device = input.device().clone();
    let matmul = Matmul::new(input, rhs, None, trans_lhs, trans_rhs, false);
    let new_view = matmul.compute_view()?;
    Ok(OpTensor::lazy(
        LazyOp::Matmul(matmul),
        new_view,
        device,
        false,
    ))
}

pub fn gemm<T1: Into<OpTensor>, T2: Into<OpTensor>, T3: Into<OpTensor>>(
    input: T1,
    rhs: T2,
    bias: Option<T3>,
    trans_lhs: bool,
    trans_rhs: bool,
    trans_out: bool,
) -> Result<OpTensor> {
    let input = input.into();
    let rhs = rhs.into();
    let bias = bias.map(|b| b.into());
    let device = input.device().clone();
    let gemm = Matmul::new(input, rhs, bias, trans_lhs, trans_rhs, trans_out);
    let new_view = gemm.compute_view()?;
    Ok(OpTensor::lazy(
        LazyOp::Matmul(gemm),
        new_view,
        device,
        false,
    ))
}

pub fn affine<T: Into<OpTensor>>(input: T, mul: f32, add: f32) -> Result<OpTensor> {
    let input = input.into();
    let device = input.device().clone();
    let affine = Affine::new(input, mul, add);
    let new_view = affine.compute_view()?;
    Ok(OpTensor::lazy(
        LazyOp::Affine(affine),
        new_view,
        device,
        false,
    ))
}

pub fn lerp<
    T1: Into<OpTensor>,
    T2: Into<OpTensor>,
    T3: Into<OpTensor>,
    T4: TensorTypeOrScalar<T3>,
>(
    input: T1,
    end: T2,
    weight: T4,
) -> Result<OpTensor> {
    let input = input.into();
    let end = end.into();
    let weight = weight.map_tensor(|w| w.into()).tensor_or_scalar()?;
    let device = input.device().clone();
    let lerp = Lerp::new(input, end, weight);
    let new_view = lerp.compute_view()?;
    Ok(OpTensor::lazy(LazyOp::Lerp(lerp), new_view, device, false))
}

fn reduce_impl<T: Into<OpTensor>>(
    input: T,
    dims: RVec<usize>,
    keepdim: bool,
    op: ReduceOp,
) -> Result<OpTensor> {
    let input = input.into();
    let device = input.device().clone();
    let reduce = Reduce::new(input, op, dims, keepdim);
    let new_view = reduce.compute_view()?;
    Ok(OpTensor::lazy(
        LazyOp::Reduce(reduce),
        new_view,
        device,
        false,
    ))
}

fn sum_impl<T: Into<OpTensor>>(input: T, sum_dims: &[usize], keepdim: bool) -> Result<OpTensor> {
    let input = input.into();
    let device = input.device().clone();
    let sum = Reduce::new(input, ReduceOp::Sum, sum_dims.into(), keepdim);
    let new_view = sum.compute_view()?;
    Ok(OpTensor::lazy(LazyOp::Reduce(sum), new_view, device, false))
}

pub fn sum_keepdim<T: Into<OpTensor>, D: Dims>(input: T, dim: D) -> Result<OpTensor> {
    let input = input.into();
    let sum_dims = dim.to_indexes(input.shape(), "sum_keepdim")?;
    sum_impl(input, &sum_dims, true)
}

pub fn sum<T: Into<OpTensor>, D: Dims>(input: T, sum_dims: D) -> Result<OpTensor> {
    let input = input.into();
    let sum_dims = sum_dims.to_indexes(input.shape(), "sum")?;
    sum_impl(input, &sum_dims, false)
}

pub fn sum_all<T: Into<OpTensor>>(input: T) -> Result<OpTensor> {
    let input = input.into();
    let dims: RVec<_> = (0..input.dim()).collect();
    sum_impl(input, &dims, false)
}

fn mean_impl<T: Into<OpTensor>>(input: T, mean_dims: &[usize], keepdim: bool) -> Result<OpTensor> {
    let input = input.into();
    let reduced_dim: usize = mean_dims.iter().map(|i| input.shape()[*i]).product();
    let scale = 1f32 / (reduced_dim as f32);
    mul::<_, OpTensor, _>(sum_impl(input, mean_dims, keepdim)?, scale)
}

pub fn mean_keepdim<T: Into<OpTensor>, D: Dims>(input: T, dim: D) -> Result<OpTensor> {
    let input = input.into();
    let mean_dims = dim.to_indexes(input.shape(), "mean_keepdim")?;
    mean_impl(input, &mean_dims, true)
}

pub fn mean<T: Into<OpTensor>, D: Dims>(input: T, dim: D) -> Result<OpTensor> {
    let input = input.into();
    let mean_dims = dim.to_indexes(input.shape(), "mean")?;
    mean_impl(input, &mean_dims, false)
}

pub fn mean_all<T: Into<OpTensor>>(input: T) -> Result<OpTensor> {
    let input = input.into();
    let dims: RVec<_> = (0..input.dim()).collect();
    mean_impl(input, &dims, false)
}

fn var_impl<T: Into<OpTensor>>(input: T, var_dims: &[usize], keepdim: bool) -> Result<OpTensor> {
    let input = input.into();
    let n: usize = var_dims.iter().map(|&i| input.shape()[i]).product();
    let mean = mean_impl(input.clone(), var_dims, true)?;
    let squares = square(sub(input, mean)?)?;
    div::<_, OpTensor, _>(sum_impl(squares, var_dims, keepdim)?, n as f32 - 1.0)
}

pub fn var_keepdim<T: Into<OpTensor>, D: Dims>(input: T, dim: D) -> Result<OpTensor> {
    let input = input.into();
    let var_dims = dim.to_indexes(input.shape(), "var_keepdim")?;
    var_impl(input, &var_dims, true)
}

pub fn var<T: Into<OpTensor>, D: Dims>(input: T, dim: D) -> Result<OpTensor> {
    let input = input.into();
    let var_dims = dim.to_indexes(input.shape(), "var")?;
    var_impl(input, &var_dims, false)
}

pub fn var_all<T: Into<OpTensor>>(input: T) -> Result<OpTensor> {
    let input = input.into();
    let dims: RVec<_> = (0..input.dim()).collect();
    var_impl(input, &dims, false)
}

fn max_impl<T: Into<OpTensor>>(input: T, max_dims: &[usize], keepdim: bool) -> Result<OpTensor> {
    let input = input.into();
    reduce_impl(input, max_dims.into(), keepdim, ReduceOp::Max)
}

pub fn max_keepdim<T: Into<OpTensor>, D: Dims>(input: T, dim: D) -> Result<OpTensor> {
    let input = input.into();
    let max_dims = dim.to_indexes(input.shape(), "max_keepdim")?;
    max_impl(input, &max_dims, true)
}

pub fn max<T: Into<OpTensor>, D: Dims>(input: T, dim: D) -> Result<OpTensor> {
    let input = input.into();
    let max_dims = dim.to_indexes(input.shape(), "max")?;
    max_impl(input, &max_dims, false)
}

fn min_impl<T: Into<OpTensor>>(input: T, min_dims: &[usize], keepdim: bool) -> Result<OpTensor> {
    let input = input.into();
    reduce_impl(input, min_dims.into(), keepdim, ReduceOp::Min)
}

pub fn min_keepdim<T: Into<OpTensor>, D: Dims>(input: T, dim: D) -> Result<OpTensor> {
    let input = input.into();
    let min_dims = dim.to_indexes(input.shape(), "min_keepdim")?;
    min_impl(input, &min_dims, true)
}

pub fn min<T: Into<OpTensor>, D: Dims>(input: T, dim: D) -> Result<OpTensor> {
    let input = input.into();
    let min_dims = dim.to_indexes(input.shape(), "min")?;
    min_impl(input, &min_dims, false)
}

pub fn argmax_keepdim<T: Into<OpTensor>, D: Dims>(input: T, dim: D) -> Result<OpTensor> {
    let input = input.into();
    let dims = dim.to_indexes(input.shape(), "argmax_keepdim")?;
    reduce_impl(input, dims, true, ReduceOp::ArgMax)
}

pub fn argmax<T: Into<OpTensor>, D: Dims>(input: T, dim: D) -> Result<OpTensor> {
    let input = input.into();
    let dims = dim.to_indexes(input.shape(), "argmax")?;
    reduce_impl(input, dims, false, ReduceOp::ArgMax)
}

pub fn argmin_keepdim<T: Into<OpTensor>, D: Dims>(input: T, dim: D) -> Result<OpTensor> {
    let input = input.into();
    let dims = dim.to_indexes(input.shape(), "argmin_keepdim")?;
    reduce_impl(input, dims, true, ReduceOp::ArgMin)
}

/// Similar to `argmin_keepdim` but the target dimension is squeezed.
pub fn argmin<T: Into<OpTensor>, D: Dims>(input: T, dim: D) -> Result<OpTensor> {
    let input = input.into();
    let dims = dim.to_indexes(input.shape(), "argmin")?;
    reduce_impl(input, dims, false, ReduceOp::ArgMin)
}

fn norm_impl<T: Into<OpTensor>>(
    input: T,
    ord: Option<NormOrd>,
    dim: Option<&[usize]>,
    keepdim: bool,
) -> Result<OpTensor> {
    let input = input.into();
    let device = input.device().clone();

    // Default ord is Frobenius/2-norm
    let ord = ord.unwrap_or_default();

    // Default dim is all dimensions (flatten)
    let dim = match dim {
        Some(d) => d.to_indexes(input.shape(), "norm_impl")?,
        None => (0..input.dim()).collect(),
    };

    match ord {
        NormOrd::Frobenius | NormOrd::P(2.0) => {
            // Use existing Norm2 reduce operation (sqrt(sum of squares))
            let reduce = Reduce::new(input, ReduceOp::Norm2, dim.clone(), keepdim);
            let new_view = reduce.compute_view()?;
            Ok(OpTensor::lazy(
                LazyOp::Reduce(reduce),
                new_view,
                device,
                false,
            ))
        }
        NormOrd::Inf => {
            // Vector ∞-norm : max(|x|)
            // Matrix ∞-norm : max(row_sums(|A|))
            let abs_tensor = abs(input)?;
            let result = if dim.len() == 1 {
                // Vector case
                if keepdim {
                    max_keepdim(abs_tensor, dim.clone())?
                } else {
                    max(abs_tensor, dim.clone())?
                }
            } else if dim.len() == 2 {
                // Matrix case – rows are dim[0], columns are dim[1]
                let row_dim = dim[0];
                let col_dim = dim[1];
                let row_sums = if keepdim {
                    sum_keepdim(abs_tensor, col_dim)?
                } else {
                    sum(abs_tensor, col_dim)?
                };
                let row_dim_adj = if !keepdim && col_dim < row_dim {
                    row_dim - 1
                } else {
                    row_dim
                };
                if keepdim {
                    max_keepdim(row_sums, row_dim_adj)?
                } else {
                    max(row_sums, row_dim_adj)?
                }
            } else {
                // Fallback – reduce max over all specified dims.
                reduce_impl(abs_tensor, dim.clone(), keepdim, ReduceOp::Max)?
            };
            Ok(result)
        }
        NormOrd::NegInf => {
            // Vector −∞-norm : min(|x|)
            // Matrix −∞-norm : min(row_sums(|A|))
            let abs_tensor = abs(input)?;
            let result = if dim.len() == 1 {
                if keepdim {
                    min_keepdim(abs_tensor, dim.clone())?
                } else {
                    min(abs_tensor, dim.clone())?
                }
            } else if dim.len() == 2 {
                let row_dim = dim[0];
                let col_dim = dim[1];
                let row_sums = sum_impl(abs_tensor, &[col_dim], keepdim)?;
                let row_dim_adj = if !keepdim && col_dim < row_dim {
                    row_dim - 1
                } else {
                    row_dim
                };
                min_impl(row_sums, &[row_dim_adj], keepdim)?
            } else {
                reduce_impl(abs_tensor, dim.clone(), keepdim, ReduceOp::Min)?
            };
            Ok(result)
        }
        NormOrd::Zero => {
            // 0-norm : number of non-zero elements (vector only)
            if dim.len() != 1 {
                anyhow::bail!("0-norm is only defined for 1-D tensors");
            }
            sum_impl(ne::<_, OpTensor, _>(input, 0.0f32)?, &dim, keepdim)
        }
        NormOrd::One => {
            // Vector 1-norm : sum(|x|)
            // Matrix 1-norm : max(column_sums(|A|))
            let abs_tensor = abs(input)?;
            let result = if dim.len() == 1 {
                sum_impl(abs_tensor, &dim, keepdim)?
            } else if dim.len() == 2 {
                let row_dim = dim[0];
                let col_dim = dim[1];
                let col_sums = if keepdim {
                    sum_keepdim(abs_tensor, row_dim)?
                } else {
                    sum(abs_tensor, row_dim)?
                };
                let col_dim_adj = if !keepdim && row_dim < col_dim {
                    col_dim - 1
                } else {
                    col_dim
                };
                if keepdim {
                    max_keepdim(col_sums, col_dim_adj)?
                } else {
                    max(col_sums, col_dim_adj)?
                }
            } else {
                anyhow::bail!("1-norm for tensors with more than 2 dims is not supported");
            };
            Ok(result)
        }
        NormOrd::NegOne => {
            // Vector (−1)-norm : harmonic mean norm
            // Matrix (−1)-norm : min(column_sums(|A|))
            let abs_tensor = abs(input)?;
            let result = if dim.len() == 1 {
                let count = abs_tensor.shape()[dim[0]] as f32;
                let sum_recip = sum_impl(recip(abs_tensor)?, &dim, keepdim)?;
                // count / sum(1/|x|)
                mul::<_, OpTensor, _>(recip(sum_recip)?, count)?
            } else if dim.len() == 2 {
                let row_dim = dim[0];
                let col_dim = dim[1];
                let col_sums = if keepdim {
                    sum_keepdim(abs_tensor, row_dim)?
                } else {
                    sum(abs_tensor, row_dim)?
                };
                let col_dim_adj = if !keepdim && row_dim < col_dim {
                    col_dim - 1
                } else {
                    col_dim
                };
                if keepdim {
                    min_keepdim(col_sums, col_dim_adj)?
                } else {
                    min(col_sums, col_dim_adj)?
                }
            } else {
                anyhow::bail!("-1 norm for tensors with more than 2 dims is not supported");
            };
            Ok(result)
        }
        NormOrd::P(p) => {
            if dim.len() != 1 {
                anyhow::bail!("p-norm is only defined for 1-D tensors");
            }
            // General p-norm ‑ (sum(|x|^p))^(1/p)
            let powered = pow::<_, OpTensor, _>(abs(input)?, p)?;
            let summed = sum_impl(powered, &dim, keepdim)?;
            pow::<_, OpTensor, _>(summed, 1.0 / p)
        }
    }
}

pub fn norm<T: Into<OpTensor>>(input: T) -> Result<OpTensor> {
    // Default: Frobenius norm over all dimensions
    let input = input.into();
    let dims: RVec<_> = (0..input.dim()).collect();
    norm_impl(input, None, Some(&dims), false)
}

pub fn norm_keepdim<T: Into<OpTensor>, D: Dims>(input: T, dim: D) -> Result<OpTensor> {
    let input = input.into();
    let dims = dim.to_indexes(input.shape(), "norm_keepdim")?;
    norm_impl(input, None, Some(&dims), true)
}

pub fn norm_ord<T: Into<OpTensor>>(input: T, ord: NormOrd) -> Result<OpTensor> {
    let input = input.into();
    let dims: RVec<_> = (0..input.dim()).collect();
    norm_impl(input, Some(ord), Some(&dims), false)
}

pub fn norm_ord_keepdim<T: Into<OpTensor>>(input: T, ord: NormOrd) -> Result<OpTensor> {
    let input = input.into();
    let dims: RVec<_> = (0..input.dim()).collect();
    norm_impl(input, Some(ord), Some(&dims), true)
}

pub fn norm_ord_dim<T: Into<OpTensor>, D: Dims>(
    input: T,
    ord: NormOrd,
    dim: D,
) -> Result<OpTensor> {
    let input = input.into();
    let dims = dim.to_indexes(input.shape(), "norm_ord_dim")?;
    norm_impl(input, Some(ord), Some(&dims), false)
}

pub fn norm_ord_dim_keepdim<T: Into<OpTensor>, D: Dims>(
    input: T,
    ord: NormOrd,
    dim: D,
) -> Result<OpTensor> {
    let input = input.into();
    let dims = dim.to_indexes(input.shape(), "norm_ord_dim_keepdim")?;
    norm_impl(input, Some(ord), Some(&dims), true)
}

fn flatten_impl<T: Into<OpTensor>>(
    input: T,
    start_dim: Option<usize>,
    end_dim: Option<usize>,
) -> Result<OpTensor> {
    let input = input.into();
    if input.dim() == 0 {
        view(input, 1)
    } else {
        let start_dim = start_dim.unwrap_or(0);
        let end_dim = end_dim.unwrap_or(input.dim() - 1);
        if start_dim < end_dim {
            let dims = input.shape();
            let mut dst_dims = dims[..start_dim].to_vec();
            dst_dims.push(
                dims.to_vec()[start_dim..end_dim + 1]
                    .iter()
                    .product::<usize>(),
            );
            if end_dim + 1 < dims.len() {
                dst_dims.extend(&dims[end_dim + 1..]);
            }
            view(input, dst_dims)
        } else {
            Ok(input)
        }
    }
}

pub fn flatten<T: Into<OpTensor>, D: Dim>(input: T, start_dim: D, end_dim: D) -> Result<OpTensor> {
    let input = input.into();
    let start_dim = start_dim.to_index(input.shape(), "flatten")?;
    let end_dim = end_dim.to_index(input.shape(), "flatten")?;
    flatten_impl(input, Some(start_dim), Some(end_dim))
}

pub fn flatten_to<T: Into<OpTensor>, D: Dim>(input: T, end_dim: D) -> Result<OpTensor> {
    let input = input.into();
    let end_dim = end_dim.to_index(input.shape(), "flatten")?;
    flatten_impl(input, None::<usize>, Some(end_dim))
}

pub fn flatten_from<T: Into<OpTensor>, D: Dim>(input: T, start_dim: D) -> Result<OpTensor> {
    let input = input.into();
    let start_dim = start_dim.to_index(input.shape(), "flatten")?;
    flatten_impl(input, Some(start_dim), None::<usize>)
}

pub fn flatten_all<T: Into<OpTensor>>(input: T) -> Result<OpTensor> {
    let input = input.into();
    flatten_impl(input, None::<usize>, None::<usize>)
}

/// # Slice
///
/// Current slice implementation requires specification of all dimensions.
/// Currently very user hostile, but will be improved.
/// TODO: should allow mixed range types
pub fn slice<T: Into<OpTensor>, D: std::ops::RangeBounds<usize>>(
    input: T,
    ranges: &[D],
) -> Result<OpTensor> {
    let input = input.into();
    let device = input.device().clone();
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
            Bound::Unbounded => input.shape()[ridx],
        };
        resolved_ranges.push(start..end);
    }

    let slice = Slice::new(input, resolved_ranges);
    let out_view = slice.compute_view()?;
    let op = LazyOp::Reindex(Reindex::Slice(slice));
    Ok(OpTensor::lazy(op, out_view, device, false))
}

/// Returns a new tensor that is a narrowed version of the input, the dimension `dim`
/// ranges from `start` to `start + len`.
/// This calls `slice` internally.
pub fn narrow<T: Into<OpTensor>, D: Dim>(
    input: T,
    dim: D,
    start: usize,
    len: usize,
) -> Result<OpTensor> {
    let input = input.into();
    let dims = input.shape().as_slice();
    let device = input.device().clone();
    let dim = dim.to_index(input.shape(), "narrow")?;
    let err = |msg| {
        anyhow::bail!(
            "invalid narrow args: shape {:?}, dim {}, start {}, len {}, {}",
            input.shape(),
            dim,
            start,
            len,
            msg
        )
    };
    if start > dims[dim] {
        err("start > dim_len")?
    }
    if start.saturating_add(len) > dims[dim] {
        err("start + len > dim_len")?
    }
    if start == 0 && dims[dim] == len {
        Ok(input)
    } else {
        // Create ranges for all dimensions, using full range for non-target dimensions
        let mut ranges = rvec![];
        dims.iter().enumerate().for_each(|(i, &_d)| {
            if i == dim {
                ranges.push(start..start + len);
            } else {
                ranges.push(0..dims[i]);
            }
        });

        let slice = Slice::new(input, ranges);
        let out_view = slice.compute_view()?;
        let op = LazyOp::Reindex(Reindex::Slice(slice));
        Ok(OpTensor::lazy(op, out_view, device, false))
    }
}

/// # View
///
/// Creates a new tensor with the same data, but a different shape.
/// The new shape must have the same number of elements as the original shape.
pub fn view<T: Into<OpTensor>, S: crate::shape::ShapeWithOneHole>(
    input: T,
    shape: S,
) -> Result<OpTensor> {
    let input = input.into();
    let shape = shape.into_shape(input.shape().numel())?;
    if input.shape().numel() != shape.numel() {
        anyhow::bail!(
            "Cannot reshape tensor with {} elements to shape {:?} ({} elements)",
            input.shape().numel(),
            shape,
            shape.numel()
        );
    }
    let device = input.device().clone();
    let storage = Arc::clone(&input.storage);
    let op = View::new(input, shape);
    let out_view = op.compute_view()?;

    Ok(OpTensor::shallow(
        LazyOp::View(op),
        out_view,
        storage,
        device,
        false,
    ))
}

// Use view to add a singleton dimension
pub fn unsqueeze<T: Into<OpTensor>, D: Dim>(input: T, dim: D) -> Result<OpTensor> {
    let input = input.into();
    let dim = dim.to_index_plus_one(input.shape(), "unsqueeze")?;
    let mut new_shape = input.shape().clone();
    new_shape.unsqueeze(dim);
    view(input, new_shape)
}

pub fn squeeze<T: Into<OpTensor>, D: Dims>(input: T, dims: D) -> Result<OpTensor> {
    let input = input.into();
    let mut new_shape = input.shape().clone();
    let dims = dims.to_indexes(input.shape(), "squeeze")?;
    new_shape.squeeze(Some(dims));
    view(input, new_shape)
}

pub fn squeeze_all<T: Into<OpTensor>>(input: T) -> Result<OpTensor> {
    let input = input.into();
    let mut new_shape = input.shape().clone();
    new_shape.squeeze(None);
    view(input, new_shape)
}

pub fn cat<T: Into<OpTensor>, D: Dim>(tensors: RVec<T>, dim: D) -> Result<OpTensor> {
    let tensors = tensors
        .into_iter()
        .map(|t| t.into())
        .collect::<RVec<OpTensor>>();
    let dim = dim.to_index(tensors[0].shape(), "cat")?;
    let device = tensors[0].device().clone();
    assert!(tensors.iter().all(|t| t.device == device), "Mixed devices");

    let cat = Concat::new(tensors, dim);
    let new_view = cat.compute_view()?;
    Ok(OpTensor::lazy(LazyOp::Concat(cat), new_view, device, false))
}

fn stack_impl(tensors: RVec<OpTensor>, dim: usize, root: bool) -> Result<OpTensor> {
    match tensors.len() {
        0 => anyhow::bail!("Cannot stack empty list of tensors"),
        1 => {
            if root {
                Ok(unsqueeze(tensors[0].clone(), dim)?)
            } else {
                Ok(tensors[0].clone())
            }
        }
        len => {
            let tensors = if root {
                tensors
                    .iter()
                    .map(|t| unsqueeze(t.clone(), dim))
                    .collect::<anyhow::Result<RVec<OpTensor>>>()?
            } else {
                tensors
            };

            let device = tensors[0].device.clone();
            assert!(tensors.iter().all(|t| t.device == device), "Mixed devices");

            if len <= 4 {
                return cat(tensors, dim);
            }

            // Process tensors in chunks of 4 recursively
            let mut current_level = tensors;

            while current_level.len() > 1 {
                let mut next_level = RVec::with_capacity(current_level.len().div_ceil(4));

                for chunk in current_level.chunks(4) {
                    let chunk_vec = chunk.iter().cloned().collect();
                    let reduced = stack_impl(chunk_vec, dim, false)?;
                    next_level.push(reduced);
                }

                current_level = next_level;
            }

            Ok(current_level.into_iter().next().unwrap())
        }
    }
}

pub fn stack<T: Into<OpTensor>, D: Dim>(tensors: RVec<T>, dim: D) -> Result<OpTensor> {
    let tensors = tensors
        .into_iter()
        .map(|t| t.into())
        .collect::<RVec<OpTensor>>();
    let dim = dim.to_index_plus_one(tensors[0].shape(), "stack")?;
    stack_impl(tensors, dim, true)
}

pub fn permute<T: Into<OpTensor>, D: Dims>(input: T, dims: D) -> Result<OpTensor> {
    let input = input.into();
    let dims = dims.to_indexes(input.shape(), "permute")?;
    let device = input.device().clone();
    let permute = Permute::new(input, dims);
    let out_view = permute.compute_view()?;

    let op = LazyOp::Reindex(Reindex::Permute(permute));
    Ok(OpTensor::lazy(op, out_view, device, false))
}

pub fn transpose<T: Into<OpTensor>, D: Dim>(input: T, dim0: D, dim1: D) -> Result<OpTensor> {
    let input = input.into();
    let dim0 = dim0.to_index(input.shape(), "transpose")?;
    let dim1 = dim1.to_index(input.shape(), "transpose")?;
    let mut dims: RVec<usize> = (0..input.dim()).collect();
    dims.swap(dim0, dim1);
    permute(input, dims)
}

pub fn t<T: Into<OpTensor>>(input: T) -> Result<OpTensor> {
    let input = input.into();
    if input.dim() > 2 {
        anyhow::bail!(
            "t() can only be applied to tensors with 2 or fewer dimensions, got tensor with {} dimensions",
            input.dim()
        );
    }
    transpose(input, 0, 1)
}

/// Returns a tensor with the last two dimensions transposed.
/// For 2D tensors, equivalent to transpose(0, 1).
/// For tensors with any other number of dimensions, throws an error.
pub fn T<T: Into<OpTensor>>(input: T) -> Result<OpTensor> {
    let input = input.into();
    match input.dim() {
        2 => transpose(input, 0, 1),
        _ => anyhow::bail!(
            "T() can only be applied to 2D tensors, got tensor with {} dimensions",
            input.dim()
        ),
    }
}

/// Matrix transpose. Returns a tensor with the last two dimensions transposed.
/// For tensors with fewer than 2 dimensions, returns the tensor unchanged.
pub fn mT<T: Into<OpTensor>>(input: T) -> Result<OpTensor> {
    let input = input.into();
    match input.dim() {
        0 | 1 => Ok(input),
        _ => {
            let dim = input.dim();
            transpose(input, dim - 2, dim - 1)
        }
    }
}

pub fn cache<T: Into<OpTensor>, D: Dim>(
    input: T,
    source: T,
    dim: D,
    offset: usize,
) -> Result<OpTensor> {
    let input = input.into();
    let source = source.into();
    let dim = dim.to_index(input.shape(), "cache")?;
    let device = input.device().clone();
    let cache = Cache::new(input, source, dim, offset);
    let new_view = cache.compute_view()?;
    Ok(OpTensor::lazy(
        LazyOp::Cache(cache),
        new_view,
        device,
        false,
    ))
}

/// Returns a new tensor duplicating data from the original tensor. New dimensions are inserted
/// on the left.
pub fn broadcast_left<T: Into<OpTensor>, S: Into<Shape>>(
    input: T,
    left_shape: S,
) -> Result<OpTensor> {
    let input = input.into();
    let mut dims = left_shape.into().to_vec();
    dims.extend(input.shape().to_vec());
    broadcast_to(input, Shape::from(dims))
}

pub fn broadcast_to<T: Into<OpTensor>, S: Into<Shape>>(input: T, shape: S) -> Result<OpTensor> {
    let input = input.into();
    let device = input.device().clone();
    let broadcast = Broadcast::new(input, shape.into());
    let new_view = broadcast.compute_view()?;

    let op = LazyOp::Reindex(Reindex::Broadcast(broadcast));
    Ok(OpTensor::lazy(op, new_view, device, false))
}

pub fn index_select<T: Into<OpTensor>, D: Dim>(input: T, indices: T, dim: D) -> Result<OpTensor> {
    let input = input.into();
    let indices = indices.into();
    let dim = dim.to_index(input.shape(), "index_select")?;
    let device = input.device().clone();
    let index_select = IndexSelect::new(input, indices, dim);
    let new_view = index_select.compute_view()?;
    Ok(OpTensor::lazy(
        LazyOp::Select(index_select),
        new_view,
        device,
        false,
    ))
}

// TODO(vinhowe): Make this API more like PyTorch's
pub fn index_write<T: Into<OpTensor>, D: Dims>(
    input: T,
    src: T,
    write_start: D,
) -> Result<OpTensor> {
    let input = input.into();
    let src = src.into();
    let write_start = write_start.to_indexes(input.shape(), "index_write")?;
    let device = input.device().clone();
    let index_write = IndexWrite::new(input, src, write_start);
    let new_view = index_write.compute_view()?;
    let op = LazyOp::IndexWrite(index_write);
    Ok(OpTensor::lazy(op, new_view, device, false))
}

pub fn where_cond<
    T1: Into<OpTensor>,
    T2: Into<OpTensor>,
    T3: Into<OpTensor>,
    T4: TensorTypeOrScalar<T3>,
>(
    input: T1,
    condition: T2,
    on_false: T4,
) -> Result<OpTensor> {
    let input = input.into();
    let condition = condition.into();
    let on_false = on_false.map_tensor(|t| t.into())?;
    let device = condition.device().clone();
    let where_cond = WhereCond::new(condition, TensorTypeOrScalarEnum::Tensor(input), on_false);
    let new_view = where_cond.compute_view()?;
    Ok(OpTensor::lazy(
        LazyOp::WhereCond(where_cond),
        new_view,
        device,
        false,
    ))
}

pub fn scatter_add<T1: Into<OpTensor>, T2: Into<OpTensor>, T3: Into<OpTensor>, D: Dim>(
    input: T1,
    indices: T2,
    source: T3,
    dim: D,
) -> Result<OpTensor> {
    let input = input.into();
    let indices = indices.into();
    let source = source.into();
    let dim = dim.to_index(input.shape(), "scatter_add")?;
    let source_dims = source.shape().to_vec();
    let self_dims = input.shape().to_vec();
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
            lhs: input.shape().clone(),
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
    let device = input.device().clone();
    let scatter_add = ScatterAdd::new(input, source, indices, dim);
    let new_view = scatter_add.compute_view()?;
    Ok(OpTensor::lazy(
        LazyOp::ScatterAdd(scatter_add),
        new_view,
        device,
        false,
    ))
}

pub fn index_add<T1: Into<OpTensor>, T2: Into<OpTensor>, T3: Into<OpTensor>, D: Dim>(
    input: T1,
    indices: T2,
    source: T3,
    dim: D,
) -> Result<OpTensor> {
    let input = input.into();
    let indices = indices.into();
    let source = source.into();
    let dim = dim.to_index(input.shape(), "index_add")?;
    let source_dims = source.shape().to_vec();
    let self_dims = input.shape().to_vec();
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
            lhs: input.shape().clone(),
            rhs: source.shape().clone(),
        })?
    }
    if indices.dim() != 1 {
        Err(InvariantError::RankMismatch {
            accepted: 1..=1,
            actual: indices.dim(),
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
    let device = input.device().clone();
    let index_add = IndexAdd::new(input, source, indices, dim);
    let new_view = index_add.compute_view()?;
    Ok(OpTensor::lazy(
        LazyOp::IndexAdd(index_add),
        new_view,
        device,
        false,
    ))
}

pub fn gather<T1: Into<OpTensor>, T2: Into<OpTensor>, D: Dim>(
    input: T1,
    indices: T2,
    dim: D,
) -> Result<OpTensor> {
    let input = input.into();
    let indices = indices.into();
    let dim = dim.to_index(input.shape(), "gather")?;
    let self_dims = input.shape().to_vec();
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
            lhs: input.shape().clone(),
            rhs: indices.shape().clone(),
        })?
    }
    let device = input.device().clone();
    let gather = Gather::new(input, indices, dim);
    let new_view = gather.compute_view()?;
    Ok(OpTensor::lazy(
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
    requires_grad: bool,
) -> Result<OpTensor> {
    arange_step::<T>(start, end, T::one(), device, requires_grad)
}

/// Creates a new 1D tensor with values from the interval `[start, end)` taken with a common
/// difference `step` from `start`.
pub fn arange_step<T: TensorDType + PartialOrd + AsPrimitive<f32>>(
    start: T,
    end: T,
    step: T,
    device: &Device,
    requires_grad: bool,
) -> Result<OpTensor> {
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
        Ok(OpTensor::from_data(
            data,
            len,
            device.clone(),
            requires_grad,
        ))
    } else {
        let arange = Arange::new(start.as_(), end.as_(), step.as_());
        let numel = arange.numel();
        let op = LazyOp::Arange(arange);

        let meta = StorageView {
            shape: numel.into(),
            dtype: T::dtype(),
            stride: Stride::from(&Shape::from(numel)),
        };

        Ok(OpTensor::lazy(op, meta, device.clone(), requires_grad))
    }
}

#[cfg(feature = "rand")]
pub fn randint<T: TensorDType + rand_distr::uniform::SampleUniform + PartialOrd, S: Into<Shape>>(
    low: T,
    high: T,
    shape: S,
    device: Device,
    requires_grad: bool,
) -> Result<OpTensor> {
    let shape = shape.into();
    let rng = device.get_rng();
    let data = (0..shape.numel())
        .map(|_| {
            let sample: T = rng.write().gen_range(low..high);
            sample
        })
        .collect::<Vec<_>>();
    Ok(OpTensor::from_data(data, shape, device, requires_grad))
}

#[cfg(feature = "rand")]
pub fn randn<T: TensorDType + num_traits::Float, S: Into<Shape>>(
    mean: T,
    std: T,
    shape: S,
    device: Device,
    requires_grad: bool,
) -> Result<OpTensor> {
    let shape = shape.into();
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
        Ok(OpTensor::new(
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
        Ok(OpTensor::new(
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
pub fn rand<T: TensorDType + num_traits::Float, S: Into<Shape>>(
    lo: T,
    up: T,
    shape: S,
    device: Device,
    requires_grad: bool,
) -> Result<OpTensor> {
    let shape = shape.into();
    let rng = device.get_rng();
    let distr = Uniform::new(lo.to_f32().unwrap(), up.to_f32().unwrap());
    let data = (0..shape.numel())
        .map(|_| {
            let sample: f32 = distr.sample(&mut *rng.write());
            T::from(sample).expect("Failed to convert sample")
        })
        .collect::<Vec<_>>();

    Ok(OpTensor::from_data(data, shape, device, requires_grad))
}

// TODO(vinhowe): Add inplace
#[cfg(feature = "rand")]
pub fn bernoulli<T: Into<OpTensor>>(input: T) -> Result<OpTensor> {
    let input = input.into();
    let rng = input.device().get_rng();
    let seed = rng.write().next_u32();
    let shape = input.shape();
    let device = input.device().clone();

    let meta = StorageView {
        shape: shape.clone(),
        dtype: DType::F32,
        stride: Stride::from(shape),
    };

    Ok(OpTensor::new(
        LazyOp::Bernoulli(Bernoulli::new(input, Some(seed))),
        meta,
        None,
        device,
        false,
    ))
}

pub(crate) fn full_impl<T: TensorDType + num_traits::AsPrimitive<f32>, S: Into<Shape>>(
    shape: S,
    value: T,
    device: &Device,
    requires_grad: bool,
) -> Result<OpTensor> {
    let shape = shape.into();
    let meta = StorageView {
        shape: shape.clone(),
        dtype: T::dtype(),
        stride: Stride::from(&shape.clone()),
    };
    Ok(OpTensor::new(
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

pub fn zeros<D: TensorDType + num_traits::AsPrimitive<f32>, S: Into<Shape>>(
    shape: S,
    device: &Device,
    requires_grad: bool,
) -> Result<OpTensor> {
    let shape = shape.into();
    if device.is_cpu() {
        let storage = Storage::zeros::<D>(&shape, device);
        let stride = Stride::from(&shape);
        let meta = StorageView::new(shape.clone(), D::dtype(), stride);
        Ok(OpTensor::new(
            LazyOp::Const,
            meta,
            Some(storage),
            device.clone(),
            requires_grad,
        ))
    } else {
        full_impl::<D, _>(shape, D::zero(), device, requires_grad)
    }
}

pub fn zeros_like<T: Into<OpTensor>, D: TensorDType + num_traits::AsPrimitive<f32>>(
    input: T,
    device: Option<&Device>,
    requires_grad: bool,
) -> Result<OpTensor> {
    let input = input.into();
    zeros::<D, _>(
        input.shape(),
        device.unwrap_or(input.device()),
        requires_grad,
    )
}

pub fn ones<D: TensorDType + num_traits::AsPrimitive<f32>, S: Into<Shape>>(
    shape: S,
    device: &Device,
    requires_grad: bool,
) -> Result<OpTensor> {
    let shape = shape.into();
    if device.is_cpu() {
        let storage = Storage::ones::<D>(&shape, device);
        let stride = Stride::from(&shape);
        let meta = StorageView::new(shape.clone(), D::dtype(), stride);
        Ok(OpTensor::new(
            LazyOp::Const,
            meta,
            Some(storage),
            device.clone(),
            requires_grad,
        ))
    } else {
        full_impl::<D, _>(shape, D::one(), device, requires_grad)
    }
}

pub fn ones_like<T: Into<OpTensor>, D: TensorDType + num_traits::AsPrimitive<f32>>(
    input: T,
    device: Option<&Device>,
    requires_grad: bool,
) -> Result<OpTensor> {
    let input = input.into();
    ones::<D, _>(
        input.shape(),
        device.unwrap_or(input.device()),
        requires_grad,
    )
}

pub fn zero_<T: Into<OpTensor>>(input: T) -> Result<OpTensor> {
    let input = input.into();
    mul::<_, OpTensor, _>(input, 0.)
}

// TODO(vinhowe): Add inplace
fn trilu<T: Into<OpTensor>>(input: T, upper: bool, k: Option<i32>) -> Result<OpTensor> {
    let input = input.into();
    let device = input.device().clone();
    let trilu = Trilu::new(input, upper, k);
    let new_view = trilu.compute_view()?;
    Ok(OpTensor::lazy(
        LazyOp::Trilu(trilu),
        new_view,
        device,
        false,
    ))
}

pub fn triu<T: Into<OpTensor>>(input: T, k: Option<i32>) -> Result<OpTensor> {
    trilu(input, true, k)
}

pub fn tril<T: Into<OpTensor>>(input: T, k: Option<i32>) -> Result<OpTensor> {
    trilu(input, false, k)
}

pub fn copy<T1: Into<OpTensor>, T2: Into<OpTensor>>(src: T1, dst: T2) -> Result<OpTensor> {
    let src = src.into();
    let dst = dst.into();
    Ok(OpTensor::new(
        LazyOp::Copy(TensorCopy {
            src: src.clone(),
            dst: dst.clone(),
        }),
        src.view.clone(),
        None,
        src.device.clone(),
        false,
    ))
}

/// Returns a tensor that is in row major order. This is the same as the original tensor if it
/// was already contiguous, otherwise a copy is triggered.
pub fn contiguous<T: Into<OpTensor>>(input: T) -> Result<OpTensor> {
    let input = input.into();
    if input.is_contiguous() {
        Ok(input.clone())
    } else {
        let storage_guard = input.storage();
        let storage = storage_guard.as_ref().unwrap();
        let cloned_storage = storage.deep_clone(input.device()).unwrap();
        Ok(OpTensor::new(
            LazyOp::Const,
            input.view.clone(),
            Some(cloned_storage),
            input.device.clone(),
            false,
        ))
    }
}

impl OpTensor {
    /// Returns true if the data is stored in a C contiguous (aka row major) way.
    pub fn is_contiguous(&self) -> bool {
        self.view.is_contiguous()
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
    pub fn from_data<T: TensorDType, U: AsRef<[T]>, S: Into<Shape>>(
        data: U,
        shape: S,
        device: Device,
        requires_grad: bool,
    ) -> Self {
        let shape = shape.into();
        let storage = Storage::from_slice(data.as_ref(), &shape, &device);
        let stride = Stride::from(&shape);
        let meta = StorageView::new(shape, T::dtype(), stride);
        Self::new(LazyOp::Const, meta, Some(storage), device, requires_grad)
    }

    pub fn from_bytes<S: Into<Shape>>(
        data: &[u8],
        dtype: DType,
        shape: S,
        device: Device,
        requires_grad: bool,
    ) -> Result<Self> {
        let shape = shape.into();
        let storage = Storage::from_bytes(data, dtype.size_of(), &device);
        let stride = Stride::from(&shape);
        let meta = StorageView::new(shape, dtype, stride);
        Ok(Self::new(
            LazyOp::Const,
            meta,
            Some(storage),
            device,
            requires_grad,
        ))
    }

    /// Create a parameter based on the values currently stored in a tensor. The storage is always
    /// copied.
    pub fn requires_grad_(&self, requires_grad: bool) -> Result<Self> {
        if self.requires_grad == requires_grad {
            Ok(self.clone())
        } else {
            let device = self.device.clone();
            let storage = Arc::clone(&self.storage);
            Ok(Self::shallow(
                self.op().clone(),
                self.view.clone(),
                storage,
                device,
                requires_grad,
            ))
        }
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
                let storage = storage_guard.as_ref().cloned();
                Self::new(
                    LazyOp::Detach(Box::new(self.op().clone())),
                    self.view.clone(),
                    storage,
                    self.device.clone(),
                    false,
                )
            }
        }
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

    pub fn from_quantized<T: TensorDType, U: AsRef<[T]>, S: Into<Shape>>(
        data: U,
        dtype: DType,
        shape: S,
        device: Device,
    ) -> Self {
        let shape = shape.into();
        let storage = unsafe { Storage::from_quantized(data.as_ref(), &device) };
        let stride = Stride::from(&shape);
        let meta = StorageView::new(shape, dtype, stride);
        Self::new(LazyOp::Const, meta, Some(storage), device, false)
    }

    pub fn from_disk<T: TensorDType, R: BufRead + Seek, S: Into<Shape>>(
        reader: &mut R,
        shape: S,
        device: Device,
    ) -> Result<Self> {
        let shape = shape.into();
        let storage = Storage::from_disk::<T, R>(reader, &shape, &device)?;
        let stride = Stride::from(&shape);
        let meta = StorageView::new(shape, T::dtype(), stride);
        Ok(Self::new(LazyOp::Const, meta, Some(storage), device, false))
    }

    #[maybe_async]
    pub async fn item<T: TensorDType>(&self) -> T {
        assert!(self.is_scalar());
        ensure_resolved!(self);
        let storage_guard = self.storage();
        let buffer = storage_guard.as_ref().unwrap().try_cpu().unwrap();
        buffer.to_slice::<T>(self.shape())[0]
    }

    /// # Bindings
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

    pub(crate) fn execution_order(&self) -> Vec<&Self> {
        let mut done = BitVec::<u32>::repeat(false, self.id().0 + 1);
        let mut pending = BitVec::<u32>::repeat(false, self.id().0 + 1);
        let mut order = Vec::new();

        let mut stack: Vec<(&OpTensor, usize)> = vec![(self, 0)];
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

            let precursor: &OpTensor = all_srcs[cur_src];
            let precursor_id = precursor.id().0;

            if done[precursor_id] {
                stack.push((cur_t, cur_src + 1));
            } else if pending[precursor_id] {
                panic!(
                    "Cycle detected whilst computing topological order: {precursor_id:?}. Try plotting with feature `plotting`."
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
    pub async fn cpu_apply(self, dst: Self) -> Option<Self> {
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
            LazyOp::Ternary(t) => t.create_gpu_compile_key(self, can_inplace, uniform).ok(),
            LazyOp::Lerp(l) => l.create_gpu_compile_key(self, can_inplace, uniform).ok(),
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

    pub(crate) fn gpu_compile_key(
        &self,
        can_inplace: bool,
        uniform: &mut CpuUniform,
    ) -> Option<GpuCompileKey<'_>> {
        match self.op() {
            LazyOp::Copy(c) => Some(GpuCompileKey::Copy(c.create_gpu_compile_key())),
            _ => self
                .gpu_compile_key_for_op(self.op(), can_inplace, uniform)
                .map(GpuCompileKey::Compute),
        }
    }

    pub(crate) fn compile_gpu<'a>(
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
    async fn resolve_cpu(self) -> Result<Self, TensorError> {
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

        Ok(tensor)
    }

    /// Applies the pending graph to the tensor.
    #[maybe_async]
    async fn apply_pending_graph(&self) -> Result<Self, TensorError> {
        if self.resolved() {
            return Ok(self.clone());
        }

        match self.device() {
            Device::GPU(gpu_device) => {
                #[cfg(target_arch = "wasm32")]
                {
                    Box::pin(gpu_device.sync_tensors_graph(vec![&self]))
                        .await
                        .map_err(|e| TensorError::LazyGraphExecutorError(Box::new(e)))?;
                }
                #[cfg(not(target_arch = "wasm32"))]
                {
                    gpu_device
                        .sync_tensors_graph(vec![&self])
                        .map_err(|e| TensorError::LazyGraphExecutorError(Box::new(e)))?;
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
    async fn to_gpu(&self, dst_device: &Device) -> Result<Self, TensorError> {
        ensure_resolved!(self);
        let storage_guard = self.storage();
        let cpu_buf = storage_guard
            .as_ref()
            .ok_or(TensorError::TransferError)?
            .try_cpu()?;
        let gpu_buf = cpu_buf.to_device(dst_device)?;

        let wgpu_device = dst_device.try_gpu()?;
        Ok(Self::new(
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
        Self::new(
            LazyOp::Const,
            self.view.clone(),
            Some(cloned_storage),
            self.device.clone(),
            false,
        )
    }

    #[maybe_async]
    async fn to_cpu(&self) -> Result<Self, TensorError> {
        ensure_resolved!(self);

        if self.device().is_cpu() {
            return Ok(self.clone());
        }
        let storage_guard = self.storage().clone();
        let gpu_buf = storage_guard
            .as_ref()
            .ok_or(TensorError::TransferError)?
            .try_gpu()?;
        let cpu_buf = gpu_buf.to_cpu(&self.device).await?;

        Ok(Self::new(
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
    pub async fn to(&self, device: &Device) -> Result<Self, TensorError> {
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

    #[cfg(not(feature = "debug"))]
    pub fn debug_tensor(&self) -> Option<Self> {
        self.debug_tensor.read().as_ref().cloned()
    }

    #[cfg(not(feature = "debug"))]
    pub fn get_or_create_debug_tensor(&self) -> Result<Self, TensorError> {
        if self.debug_tensor.read().is_some() {
            return Ok(self.debug_tensor.read().as_ref().unwrap().clone());
        }

        let gpu_device = self.device().try_gpu()?;
        let buffer = gpu_device.get_or_create_buffer(
            &BufferDescriptor {
                size: self.num_bytes() as _,
                // If we want the values in CPU land, we'll eventually have to
                // copy again to a buffer with a usage of COPY_DST | MAP_READ.
                usage: wgpu::BufferUsages::standard(),
                mapped_at_creation: false,
            },
            false,
        )?;
        let tensor = Self::new(
            LazyOp::Const,
            self.view.clone(),
            Some(Storage::GPU(GPUBuffer {
                inner: buffer,
                alignment: self.dtype().size_of(),
                cpu_size: Some(self.num_bytes()),
            })),
            Device::GPU(gpu_device.clone()),
            false,
        );
        *self.debug_tensor.write() = Some(tensor.clone());
        Ok(tensor)
    }

    pub fn grad(&self) -> Option<Tensor> {
        self.grad.read().as_ref().cloned()
    }

    pub fn set_grad(&self, grad: Option<Tensor>) {
        log::trace!(
            "Setting grad for {:?}: {:?}",
            self.id(),
            grad.as_ref().map(|_| "Some").unwrap_or("None")
        );
        *self.grad.write() = grad;
    }

    pub fn take_grad(&self) -> Option<Tensor> {
        log::trace!("Taking grad for {:?}", self.id());
        self.grad.write().take()
    }

    /// Invalidates the tensor by setting its storage to None.
    /// After calling this method, the tensor will no longer be resolved.
    pub fn invalidate(&self) -> Result<(), TensorError> {
        // Fail silently if the tensor is already invalidated or not resolved.
        if self.invalidated() || !self.resolved() {
            return Ok(());
        }
        *self.storage.write() = None;
        *self.invalidated.write() = true;
        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
pub enum NormOrd {
    /// Frobenius norm (default for matrices, same as 2-norm)
    #[default]
    Frobenius,
    /// Infinity norm (max of absolute values)
    Inf,
    /// Negative infinity norm (min of absolute values)
    NegInf,
    /// 0-norm (count non-zero elements, for vectors only)
    Zero,
    /// 1-norm (sum of absolute values)
    One,
    /// -1-norm (harmonic mean norm)
    NegOne,
    /// p-norm for arbitrary p (for vectors only)
    P(f32),
}

impl FromStr for NormOrd {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "fro" => Ok(Self::Frobenius),
            "inf" => Ok(Self::Inf),
            "-inf" => Ok(Self::NegInf),
            "0" => Ok(Self::Zero),
            "1" => Ok(Self::One),
            "-1" => Ok(Self::NegOne),
            _ => Ok(Self::P(s.parse::<f32>()?)),
        }
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
        LazyOp::Ternary(t) => t.compile_gpu(gpu_compile_key, gpu_device, debug).ok(),
        LazyOp::Lerp(l) => l.compile_gpu(gpu_compile_key, gpu_device, debug).ok(),
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
impl<T: TensorDType + numpy::Element> From<&PyArrayDyn<T>> for OpTensor {
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
impl OpTensor {
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
            .collect::<RVec<_>>();
        let data = reader.into_vec::<T>()?;
        Ok(OpTensor::from_data(
            data,
            shape,
            device.clone(),
            requires_grad,
        ))
    }

    pub fn into_ndarray<T: TensorDType>(self) -> ArrayD<T> {
        self.to_ndarray_view().into_owned()
    }

    pub fn to_ndarray_view<T: TensorDType>(&self) -> ArrayViewD<'_, T> {
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

#[cfg(feature = "testing")]
impl Tensor {
    pub fn all_close<T>(&self, other: &Self, atol: T, rtol: T) -> anyhow::Result<()>
    where
        T: TensorDType + std::fmt::Display + num_traits::Float + Default,
    {
        self.inner_or_source()
            .all_close(&other.inner_or_source(), atol, rtol)
    }
}
impl<T: TensorDType> From<ArrayD<T>> for OpTensor {
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
            OpTensor::new(
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

impl safetensors::View for &OpTensor {
    fn dtype(&self) -> safetensors::Dtype {
        match OpTensor::dtype(self) {
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
        OpTensor::shape(self).inner()
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

// Most of the actual work is done in OpTensor; this very manually wraps OpTensor so we can do
// inplace operations similar to PyTorch, which is written in C++, and does inner mutability more
// straightforwardly (/less safely).
#[derive(Clone)]
pub struct Tensor {
    inner: Arc<RwLock<OpTensor>>,
    inplace_source: Arc<RwLock<Option<OpTensor>>>,
}

impl Tensor {
    #[track_caller]
    fn lazy(op: LazyOp, meta: StorageView, device: Device, requires_grad: bool) -> Self {
        OpTensor::lazy(op, meta, device, requires_grad).into()
    }

    pub fn shallow(
        op: LazyOp,
        meta: StorageView,
        storage: Arc<RwLock<Option<Storage>>>,
        device: Device,
        requires_grad: bool,
    ) -> Self {
        OpTensor::shallow(op, meta, storage, device, requires_grad).into()
    }

    pub fn inner_or_source(&self) -> RwLockWriteGuard<'_, OpTensor> {
        let mut inplace_source = self.inplace_source.write();
        let mut inner = self.inner.write();
        if let Some(inplace_source) = inplace_source.take_if(|_| inner.resolved()) {
            *inner = inplace_source.clone();
        }
        inner
    }

    pub fn wrap(op_tensor: OpTensor) -> Self {
        Self {
            inner: Arc::new(RwLock::new(op_tensor)),
            inplace_source: Arc::new(RwLock::new(None)),
        }
    }

    fn wrap_inplace_impl(&self, op_tensor: OpTensor, track: bool) -> Self {
        if track {
            let mut inplace_source = self.inplace_source.write();
            if inplace_source.is_none() {
                *inplace_source = Some(self.inner.read().clone());
            }
        }

        *op_tensor.inplace.write() = true;
        *self.inner.write() = op_tensor;
        self.clone()
    }

    fn wrap_inplace_untracked(&self, op_tensor: OpTensor) -> Self {
        // This is important for making sure we don't erase gradient information.
        // If we wanted to be really careful, we would move this to the tensor right after the
        // inplace source, set the inplace source to the new tensor, then replay all other inplace
        // operations on that tensor.
        self.wrap_inplace_impl(op_tensor, false)
    }

    fn wrap_inplace(&self, op_tensor: OpTensor) -> Self {
        self.wrap_inplace_impl(op_tensor, true)
    }

    pub fn inner(&self) -> &Arc<RwLock<OpTensor>> {
        &self.inner
    }

    pub fn new(
        op: LazyOp,
        meta: StorageView,
        storage: Option<Storage>,
        device: Device,
        requires_grad: bool,
    ) -> Self {
        OpTensor::new(op, meta, storage, device, requires_grad).into()
    }

    pub fn full<T: TensorDType + num_traits::AsPrimitive<f32>, S: Into<Shape>>(
        shape: S,
        value: T,
        device: &Device,
        requires_grad: bool,
    ) -> Result<Self> {
        OpTensor::full(shape, value, device, requires_grad).map(Self::wrap)
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.inner_or_source())
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        let self_tensor = self.inner_or_source();
        let other_tensor = other.inner_or_source();
        *self_tensor == *other_tensor
    }
}

// No deref

impl Tensor {
    pub fn id(&self) -> TensorId {
        self.inner_or_source().id()
    }

    pub fn dim(&self) -> usize {
        self.inner_or_source().dim()
    }

    pub fn dtype(&self) -> DType {
        self.inner_or_source().dtype()
    }

    pub fn shape(&self) -> Shape {
        self.inner_or_source().shape().clone()
    }

    pub fn stride(&self) -> Stride {
        self.inner_or_source().stride().clone()
    }

    pub fn device(&self) -> Device {
        self.inner_or_source().device().clone()
    }

    pub fn resolved(&self) -> bool {
        self.inner_or_source().resolved()
    }

    pub fn op(&self) -> LazyOp {
        self.inner_or_source().op().clone()
    }

    pub fn scope(&self) -> Option<String> {
        self.inner_or_source().scope().clone()
    }

    pub fn is_scalar(&self) -> bool {
        self.inner_or_source().is_scalar()
    }

    pub fn requires_grad(&self) -> bool {
        self.inner_or_source().requires_grad()
    }

    pub fn set_sync(&self, src: Self) -> Result<()> {
        self.inner_or_source()
            .clone()
            .set(src.inner_or_source().clone());
        Ok(())
    }

    pub fn set(&self, src: Self) -> Self {
        self.inner_or_source()
            .clone()
            .set(src.inner_or_source().clone());
        self.clone()
    }

    pub fn num_bytes(&self) -> usize {
        self.inner_or_source().num_bytes()
    }
}

struct CastBackendOp;
impl BoxedKernel for CastBackendOp {
    fn handle(
        &self,
        _op: &OperatorHandle,
        _keys: DispatchKeySet,
        args: &mut RVec<Value>,
    ) -> anyhow::Result<()> {
        let (tensor, dtype) = (Tensor::pop(args), DType::pop(args));
        let result = cast(tensor, dtype).map(Tensor::wrap)?;
        result.push(args);
        Ok(())
    }
}

// SAFETY: `CastBackendOp` is stateless and thus thread-safe.
unsafe impl Sync for CastBackendOp {}

inventory::submit! {
    OperatorRegistration {
        name: "cast",
        handle: &OperatorHandle {
            name: "cast",
            call_fn: |name, stack| {
                // TODO: Evaluate if this is what we want to do here
                let (tensor, dtype) = (Tensor::pop(stack), DType::pop(stack));
                let result = crate::dispatch::dispatch_operator::<(Tensor, DType), Tensor>(name, (tensor, dtype))?;
                result.push(stack);
                Ok(())
            },
        },
    }
}

inventory::submit! {
    OperatorHandlerRegistration {
        name: "cast",
        key: DispatchKeySet::BACKEND,
        handler: &KernelFunction {
            unboxed: Some(|_op, _keys, args| {
                let (tensor, dtype) = (Tensor::pop(args), DType::pop(args));
                let result = cast(tensor, dtype).map(Tensor::wrap)?;
                result.push(args);
                Ok(())
            }),
            boxed: None,
        },
    }
}

impl Tensor {
    pub fn cast(self, dst_dtype: DType) -> Result<Self> {
        dispatch_operator("cast", (self, dst_dtype))
    }

    /// Cast a tensor to full precision (IEEE 754 32-bit floating point).
    pub fn float(self) -> Result<Self> {
        float(self).map(Self::wrap)
    }

    /// Cast a tensor to half precision (IEEE 754 16-bit floating point).
    pub fn half(self) -> Result<Self> {
        half(self).map(Self::wrap)
    }

    pub fn group_norm(
        self,
        num_groups: usize,
        weight: Option<Self>,
        bias: Option<Self>,
        eps: f32,
    ) -> Result<Self> {
        group_norm(self, num_groups, weight, bias, eps).map(Self::wrap)
    }

    pub fn layer_norm(self, weight: Option<Self>, bias: Option<Self>, eps: f32) -> Result<Self> {
        layer_norm(self, weight, bias, eps).map(Self::wrap)
    }

    pub fn rms_norm(self, weight: Option<Self>, eps: f32) -> Result<Self> {
        rms_norm(self, weight, eps).map(Self::wrap)
    }

    pub fn conv1d(
        self,
        weight: Self,
        bias: Option<Self>,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        conv1d(self, weight, bias, stride, padding).map(Self::wrap)
    }

    pub fn softmax<D: Dim>(self, dim: D) -> Result<Self> {
        softmax(self, dim).map(Self::wrap)
    }

    pub fn rope(self, dim: usize, base: f32, offset: usize) -> Result<Self> {
        rope(self, dim, base, offset).map(Self::wrap)
    }

    pub(crate) fn rope_backward(self, dim: usize, base: f32, offset: usize) -> Result<Self> {
        rope_backward(self, dim, base, offset).map(Self::wrap)
    }

    pub fn alibi(self, max_bias: f32) -> Result<Self> {
        alibi(self, max_bias).map(Self::wrap)
    }

    pub fn alibi_inplace(self, max_bias: f32) -> Result<Self> {
        alibi(self, max_bias).map(Self::wrap)
    }

    //TODO (vinhowe): figure out how to make this interface more like pytorch
    pub fn matmul(self, rhs: Self, trans_lhs: bool, trans_rhs: bool) -> Result<Self> {
        matmul(self, rhs, trans_lhs, trans_rhs).map(Self::wrap)
    }

    pub fn gemm(
        self,
        rhs: Self,
        bias: Option<Self>,
        trans_lhs: bool,
        trans_rhs: bool,
        trans_out: bool,
    ) -> Result<Self> {
        gemm(self, rhs, bias, trans_lhs, trans_rhs, trans_out).map(Self::wrap)
    }

    pub fn affine(self, mul: f32, add: f32) -> Result<Self> {
        affine(self, mul, add).map(Self::wrap)
    }

    pub fn affine_(self, mul: f32, add: f32) -> Result<Self> {
        Ok(self.wrap_inplace(affine(self.clone(), mul, add)?))
    }

    pub fn lerp<T: TensorTypeOrScalar<Tensor>>(self, end: Self, weight: T) -> Result<Self> {
        lerp(self, end, weight).map(Self::wrap)
    }

    pub fn lerp_<T: TensorTypeOrScalar<Tensor>>(self, end: Self, weight: T) -> Result<Self> {
        Ok(self.wrap_inplace(lerp(self.clone(), end, weight)?))
    }

    pub fn sum_keepdim<D: Dims>(self, sum_dims: D) -> Result<Self> {
        sum_keepdim(self, sum_dims).map(Self::wrap)
    }

    pub fn sum<D: Dims>(self, sum_dims: D) -> Result<Self> {
        sum(self, sum_dims).map(Self::wrap)
    }

    pub fn sum_all(self) -> Result<Self> {
        sum_all(self).map(Self::wrap)
    }

    pub fn mean_keepdim<D: Dims>(self, dim: D) -> Result<Self> {
        mean_keepdim(self, dim).map(Self::wrap)
    }

    pub fn mean<D: Dims>(self, dim: D) -> Result<Self> {
        mean(self, dim).map(Self::wrap)
    }

    pub fn mean_all(self) -> Result<Self> {
        mean_all(self).map(Self::wrap)
    }

    pub fn var_keepdim<D: Dims>(self, dim: D) -> Result<Self> {
        var_keepdim(self, dim).map(Self::wrap)
    }

    pub fn var<D: Dims>(self, dim: D) -> Result<Self> {
        var(self, dim).map(Self::wrap)
    }

    pub fn var_all(self) -> Result<Self> {
        var_all(self).map(Self::wrap)
    }

    pub fn max_keepdim<D: Dim>(self, dim: D) -> Result<Self> {
        max_keepdim(self, dim).map(Self::wrap)
    }

    pub fn max<D: Dim>(self, dim: D) -> Result<Self> {
        max(self, dim).map(Self::wrap)
    }

    pub fn min_keepdim<D: Dim>(self, dim: D) -> Result<Self> {
        min_keepdim(self, dim).map(Self::wrap)
    }

    pub fn min<D: Dim>(self, dim: D) -> Result<Self> {
        min(self, dim).map(Self::wrap)
    }

    pub fn argmax_keepdim<D: Dim>(self, dim: D) -> Result<Self> {
        argmax_keepdim(self, dim).map(Self::wrap)
    }

    pub fn argmax<D: Dim>(self, dim: D) -> Result<Self> {
        argmax(self, dim).map(Self::wrap)
    }

    pub fn argmin_keepdim<D: Dim>(self, dim: D) -> Result<Self> {
        argmin_keepdim(self, dim).map(Self::wrap)
    }

    /// Similar to `argmin_keepdim` but the target dimension is squeezed.
    pub fn argmin<D: Dim>(self, dim: D) -> Result<Self> {
        argmin(self, dim).map(Self::wrap)
    }

    pub fn norm(self) -> Result<Self> {
        norm(self).map(Self::wrap)
    }

    pub fn norm_keepdim<D: Dims>(self, dim: D) -> Result<Self> {
        norm_keepdim(self, dim).map(Self::wrap)
    }

    pub fn norm_ord(self, ord: NormOrd) -> Result<Self> {
        norm_ord(self, ord).map(Self::wrap)
    }

    pub fn norm_ord_keepdim(self, ord: NormOrd) -> Result<Self> {
        norm_ord_keepdim(self, ord).map(Self::wrap)
    }

    pub fn norm_ord_dim<D: Dims>(self, ord: NormOrd, dim: D) -> Result<Self> {
        norm_ord_dim(self, ord, dim).map(Self::wrap)
    }

    pub fn norm_ord_dim_keepdim<D: Dims>(self, ord: NormOrd, dim: D) -> Result<Self> {
        norm_ord_dim_keepdim(self, ord, dim).map(Self::wrap)
    }

    pub fn flatten<D: Dim>(self, start_dim: D, end_dim: D) -> Result<Self> {
        flatten(self, start_dim, end_dim).map(Self::wrap)
    }

    pub fn flatten_to<D: Dim>(self, end_dim: D) -> Result<Self> {
        flatten_to(self, end_dim).map(Self::wrap)
    }

    pub fn flatten_from<D: Dim>(self, start_dim: D) -> Result<Self> {
        flatten_from(self, start_dim).map(Self::wrap)
    }

    pub fn flatten_all(self) -> Result<Self> {
        flatten_all(self).map(Self::wrap)
    }

    /// # Slice
    ///
    /// Current slice implementation requires specification of all dimensions.
    /// Currently very user hostile, but will be improved.
    /// TODO: should allow mixed range types
    pub fn slice<D: std::ops::RangeBounds<usize>>(self, ranges: &[D]) -> Result<Self> {
        slice(self, ranges).map(Self::wrap)
    }

    /// Returns a new tensor that is a narrowed version of the input, the dimension `dim`
    /// ranges from `start` to `start + len`.
    /// This calls `slice` internally.
    pub fn narrow<D: Dim>(self, dim: D, start: usize, len: usize) -> Result<Self> {
        narrow(self, dim, start, len).map(Self::wrap)
    }

    /// # View
    ///
    /// Creates a new tensor with the same data, but a different shape.
    /// The new shape must have the same number of elements as the original shape.
    pub fn view<S: crate::shape::ShapeWithOneHole>(self, shape: S) -> Result<Self> {
        view(self, shape).map(Self::wrap)
    }

    pub fn unsqueeze<D: Dim>(self, dim: D) -> Result<Self> {
        unsqueeze(self, dim).map(Self::wrap)
    }

    pub fn unsqueeze_<D: Dim>(self, dim: D) -> Result<Self> {
        Ok(self.wrap_inplace(unsqueeze(self.clone(), dim)?))
    }

    pub fn squeeze<D: Dims>(self, dims: D) -> Result<Self> {
        squeeze(self, dims).map(Self::wrap)
    }

    pub fn squeeze_<D: Dims>(self, dims: D) -> Result<Self> {
        Ok(self.wrap_inplace(squeeze(self.clone(), dims)?))
    }

    pub fn squeeze_all(self) -> Result<Self> {
        squeeze_all(self).map(Self::wrap)
    }

    pub fn cat<D: Dim>(tensors: RVec<Self>, dim: D) -> Result<Self> {
        cat(tensors, dim).map(Self::wrap)
    }

    pub fn stack<D: Dim>(tensors: RVec<Self>, dim: D) -> Result<Self> {
        stack(tensors, dim).map(Self::wrap)
    }

    pub fn permute<D: Dims>(self, dims: D) -> Result<Self> {
        permute(self, dims).map(Self::wrap)
    }

    pub fn transpose<D: Dim>(self, dim0: D, dim1: D) -> Result<Self> {
        transpose(self, dim0, dim1).map(Self::wrap)
    }

    pub fn t(self) -> Result<Self> {
        t(self).map(Self::wrap)
    }

    /// Returns a tensor with the last two dimensions transposed.
    /// For 2D tensors, equivalent to transpose(0, 1).
    /// For tensors with any other number of dimensions, throws an error.
    pub fn T(self) -> Result<Self> {
        T(self).map(Self::wrap)
    }

    /// Matrix transpose. Returns a tensor with the last two dimensions transposed.
    /// For tensors with fewer than 2 dimensions, returns the tensor unchanged.
    pub fn mT(self) -> Result<Self> {
        mT(self).map(Self::wrap)
    }

    pub fn cache<D: Dim>(self, source: Self, dim: D, offset: usize) -> Result<Self> {
        cache(self, source, dim, offset).map(Self::wrap)
    }

    /// Returns a new tensor duplicating data from the original tensor. New dimensions are inserted
    /// on the left.
    pub fn broadcast_left<S: Into<Shape>>(self, left_shape: S) -> Result<Self> {
        broadcast_left(self, left_shape).map(Self::wrap)
    }

    pub fn broadcast_to<S: Into<Shape>>(self, shape: S) -> Result<Self> {
        broadcast_to(self, shape).map(Self::wrap)
    }

    pub fn index_select<D: Dim>(self, indices: Self, dim: D) -> Result<Self> {
        index_select(self, indices, dim).map(Self::wrap)
    }

    // TODO(vinhowe): Make this API more like PyTorch's
    pub fn index_write<D: Dims>(self, src: Self, write_start: D) -> Result<Self> {
        index_write(self, src, write_start).map(Self::wrap)
    }

    pub fn where_cond<T: TensorTypeOrScalar<Tensor>>(
        self,
        condition: Self,
        on_false: T,
    ) -> Result<Self> {
        where_cond(self, condition, on_false).map(Self::wrap)
    }

    pub fn scatter_add<D: Dim>(self, indices: Self, source: Self, dim: D) -> Result<Self> {
        scatter_add(self, indices, source, dim).map(Self::wrap)
    }

    pub fn scatter_add_<D: Dim>(self, indices: Self, source: Self, dim: D) -> Result<Self> {
        Ok(self.wrap_inplace(scatter_add(self.clone(), indices, source, dim)?))
    }

    pub fn index_add_<D: Dim>(self, indices: Self, source: Self, dim: D) -> Result<Self> {
        Ok(self.wrap_inplace(index_add(self.clone(), indices, source, dim)?))
    }

    pub fn gather<D: Dim>(self, indices: Self, dim: D) -> Result<Self> {
        gather(self, indices, dim).map(Self::wrap)
    }

    pub fn arange<T: TensorDType + PartialOrd + AsPrimitive<f32>>(
        start: T,
        end: T,
        device: &Device,
        requires_grad: bool,
    ) -> Result<Self> {
        arange(start, end, device, requires_grad).map(Self::wrap)
    }

    /// Creates a new 1D tensor with values from the interval `[start, end)` taken with a common
    /// difference `step` from `start`.
    pub fn arange_step<T: TensorDType + PartialOrd + AsPrimitive<f32>>(
        start: T,
        end: T,
        step: T,
        device: &Device,
        requires_grad: bool,
    ) -> Result<Self> {
        arange_step(start, end, step, device, requires_grad).map(Self::wrap)
    }

    #[cfg(feature = "rand")]
    pub fn randint<
        T: TensorDType + rand_distr::uniform::SampleUniform + PartialOrd,
        S: Into<Shape>,
    >(
        low: T,
        high: T,
        shape: S,
        device: Device,
        requires_grad: bool,
    ) -> Result<Self> {
        randint(low, high, shape, device, requires_grad).map(Self::wrap)
    }

    #[cfg(feature = "rand")]
    pub fn randn<T: TensorDType + num_traits::Float, S: Into<Shape>>(
        mean: T,
        std: T,
        shape: S,
        device: Device,
        requires_grad: bool,
    ) -> Result<Self> {
        randn(mean, std, shape, device, requires_grad).map(Self::wrap)
    }

    #[cfg(feature = "rand")]
    pub fn rand<T: TensorDType + num_traits::Float, S: Into<Shape>>(
        lo: T,
        up: T,
        shape: S,
        device: Device,
        requires_grad: bool,
    ) -> Result<Self> {
        rand(lo, up, shape, device, requires_grad).map(Self::wrap)
    }

    // TODO(vinhowe): Add inplace
    #[cfg(feature = "rand")]
    pub fn bernoulli(self) -> Result<Self> {
        bernoulli(self).map(Self::wrap)
    }

    #[cfg(feature = "rand")]
    pub fn bernoulli_(self) -> Result<Self> {
        Ok(self.wrap_inplace(bernoulli(self.clone())?))
    }

    pub fn zeros<T: TensorDType + num_traits::AsPrimitive<f32>, S: Into<Shape>>(
        shape: S,
        device: &Device,
        requires_grad: bool,
    ) -> Result<Self> {
        zeros::<T, S>(shape, device, requires_grad).map(Self::wrap)
    }

    pub fn zeros_like<D: TensorDType + num_traits::AsPrimitive<f32>>(
        &self,
        device: Option<&Device>,
        requires_grad: bool,
    ) -> Result<Self> {
        zeros_like::<_, D>(self, device, requires_grad).map(Self::wrap)
    }

    pub fn ones<T: TensorDType + num_traits::AsPrimitive<f32>, S: Into<Shape>>(
        shape: S,
        device: &Device,
        requires_grad: bool,
    ) -> Result<Self> {
        ones::<T, S>(shape, device, requires_grad).map(Self::wrap)
    }

    pub fn ones_like<D: TensorDType + num_traits::AsPrimitive<f32>>(
        &self,
        device: Option<&Device>,
        requires_grad: bool,
    ) -> Result<Self> {
        ones_like::<_, D>(self, device, requires_grad).map(Self::wrap)
    }

    pub fn zero_(self) -> Result<Self> {
        Ok(self.wrap_inplace(zero_(self.clone())?))
    }

    pub fn triu(self, k: Option<i32>) -> Result<Self> {
        triu(self, k).map(Self::wrap)
    }

    pub fn triu_(self, k: Option<i32>) -> Result<Self> {
        Ok(self.wrap_inplace(triu(self.clone(), k)?))
    }

    pub fn tril(self, k: Option<i32>) -> Result<Self> {
        tril(self, k).map(Self::wrap)
    }

    pub fn tril_(self, k: Option<i32>) -> Result<Self> {
        Ok(self.wrap_inplace(tril(self.clone(), k)?))
    }

    /// Returns true if the data is stored in a C contiguous (aka row major) way.
    pub fn is_contiguous(&self) -> bool {
        self.inner_or_source().clone().is_contiguous()
    }

    /// Returns a tensor that is in row major order. This is the same as the original tensor if it
    /// was already contiguous, otherwise a copy is triggered.
    pub fn contiguous(self) -> Result<Self> {
        contiguous(self).map(Self::wrap)
    }

    pub fn has_nan<T: TensorDType + num_traits::Float>(&self) -> bool {
        self.inner_or_source().clone().has_nan::<T>()
    }

    /// Creates a new tensor from a chunk of data.
    ///
    /// The Tensor is instantly resolved.
    /// If a non-CPU device is specified, the data will be copied to the device.
    pub fn from_data<T: TensorDType, U: AsRef<[T]>, S: Into<Shape>>(
        data: U,
        shape: S,
        device: Device,
        requires_grad: bool,
    ) -> Self {
        OpTensor::from_data(data, shape, device, requires_grad).into()
    }

    pub fn from_bytes<S: Into<Shape>>(
        data: &[u8],
        dtype: DType,
        shape: S,
        device: Device,
        requires_grad: bool,
    ) -> Result<Self> {
        OpTensor::from_bytes(data, dtype, shape, device, requires_grad).map(Self::wrap)
    }

    /// Create a parameter based on the values currently stored in a tensor. The storage is always
    /// copied.
    pub fn requires_grad_(&self, requires_grad: bool) -> Result<Self> {
        self.inner_or_source()
            .requires_grad_(requires_grad)
            .map(Self::wrap)
    }

    /// Returns a new tensor detached from the current graph, gradient are not propagated through
    /// this new node. The storage of this tensor is shared with the initial tensor.
    ///
    /// If the tensor is already detached from the computation graph, the same tensor is returned.
    pub fn detach(&self) -> Self {
        Self::wrap(self.inner_or_source().detach())
    }

    pub fn detach_(&self) -> Self {
        let inner = self.inner_or_source().clone();
        self.wrap_inplace_untracked(inner.detach())
    }

    pub fn copy(&self, dst: &Self) -> Result<Self> {
        copy(self, dst).map(Self::wrap)
    }

    /// # Safety
    ///
    /// If the tensor has more than 1 reference, you die.
    /// If the tensor has no storage, you die.
    pub fn into_bytes(self) -> anyhow::Result<Vec<u8>> {
        self.inner_or_source().clone().into_bytes()
    }

    pub fn from_quantized<T: TensorDType, U: AsRef<[T]>, S: Into<Shape>>(
        data: U,
        dtype: DType,
        shape: S,
        device: Device,
    ) -> Self {
        Self::wrap(OpTensor::from_quantized(data, dtype, shape, device))
    }

    pub fn from_disk<T: TensorDType, R: BufRead + Seek, S: Into<Shape>>(
        reader: &mut R,
        shape: S,
        device: Device,
    ) -> Result<Self> {
        OpTensor::from_disk::<T, R, S>(reader, shape, device).map(Self::wrap)
    }

    #[maybe_async]
    pub async fn item<T: TensorDType>(&self) -> T {
        let inner = self.inner_or_source().clone();
        inner.item::<T>().await
    }

    /// Converts the tensor into a 1D vector.
    ///
    /// The 1D vector contains the data from the tensor, as it was laid out in memory.
    #[maybe_async]
    pub async fn to_vec<T: TensorDType>(&self) -> anyhow::Result<Vec<T>> {
        let inner = self.inner_or_source().clone();
        inner.to_vec::<T>().await
    }

    #[maybe_async]
    pub async fn cpu_apply(self, dst: Self) -> Option<Self> {
        let inner_clone = self.inner_or_source().clone();
        let dst_inner_clone = dst.inner_or_source().clone();

        inner_clone.cpu_apply(dst_inner_clone).await.map(Self::wrap)
    }

    #[maybe_async]
    pub async fn deep_clone(&self) -> Self {
        let inner_clone = self.inner_or_source().clone();
        Self::wrap(inner_clone.deep_clone().await)
    }

    /// Transfers the tensor to the specified device.
    ///
    /// If the tensor is already on the specified device, it will be returned as-is,
    /// and the underlying storage will not be copied.
    /// If the tensor is on a different device, it will be copied to the specified device.
    #[maybe_async]
    pub async fn to(&self, device: &Device) -> Result<Self, TensorError> {
        let inner_clone = self.inner_or_source().clone();
        inner_clone.to(device).await.map(Self::wrap)
    }

    #[cfg(not(feature = "debug"))]
    pub fn debug_tensor(&self) -> Option<Self> {
        self.inner_or_source().debug_tensor().map(Self::wrap)
    }

    #[cfg(not(feature = "debug"))]
    pub fn get_or_create_debug_tensor(&self) -> Result<Self, TensorError> {
        self.inner_or_source()
            .get_or_create_debug_tensor()
            .map(Self::wrap)
    }

    pub fn grad(&self) -> Option<Self> {
        self.inner_or_source().grad()
    }

    pub fn set_grad(&self, grad: Option<Self>) {
        self.inner_or_source().set_grad(grad);
    }

    pub fn take_grad(&self) -> Option<Self> {
        self.inner_or_source().take_grad()
    }

    pub fn storage_id(&self) -> Option<usize> {
        self.inner_or_source()
            .storage()
            .as_ref()
            .and_then(|s| match s {
                Storage::CPU(_) => None,
                Storage::GPU(g) => Some(g.inner().global_id().inner() as _),
            })
    }

    /// Invalidates the tensor by setting its storage to None.
    /// After calling this method, the tensor will no longer be resolved.
    pub fn invalidate(&self) -> Result<(), TensorError> {
        self.inner_or_source().invalidate()
    }
}

impl<T: TensorDType> From<ArrayD<T>> for Tensor {
    fn from(it: ArrayD<T>) -> Self {
        Self::wrap(OpTensor::from(it))
    }
}

macro_rules! bin_trait_wrapper {
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
            type Output = Result<Tensor>;

            fn $fn1(self, rhs: f32) -> Self::Output {
                self.affine($mul(rhs), $add(rhs))
            }
        }
    };
}

bin_trait_wrapper!(Add, add, |_| 1., |v| v);
bin_trait_wrapper!(Sub, sub, |_| 1., |v: f32| -v);
bin_trait_wrapper!(Mul, mul, |v| v, |_| 0.);
bin_trait_wrapper!(Div, div, |v| 1. / v, |_| 0.);

impl std::ops::Add<Tensor> for f32 {
    type Output = Result<Tensor>;

    fn add(self, rhs: Tensor) -> Self::Output {
        add::<_, Tensor, _>(rhs, self).map(Tensor::wrap)
    }
}

impl std::ops::Mul<Tensor> for f32 {
    type Output = Result<Tensor>;

    fn mul(self, rhs: Tensor) -> Self::Output {
        mul::<_, Tensor, _>(rhs, self).map(Tensor::wrap)
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

impl From<Tensor> for OpTensor {
    fn from(t: Tensor) -> Self {
        t.inner_or_source().clone()
    }
}

impl From<&Tensor> for OpTensor {
    fn from(t: &Tensor) -> Self {
        t.inner_or_source().clone()
    }
}

impl From<OpTensor> for Tensor {
    fn from(t: OpTensor) -> Self {
        Self::wrap(t)
    }
}

#[derive(Debug, Clone)]
pub enum TensorTypeOrScalarEnum<T> {
    Tensor(T),
    Scalar(f32),
}

pub trait TensorTypeOrScalar<TensorType> {
    fn tensor_or_scalar(&self) -> Result<TensorTypeOrScalarEnum<TensorType>>;
    fn tensor(&self) -> Option<TensorType> {
        self.tensor_or_scalar().ok().and_then(|x| match x {
            TensorTypeOrScalarEnum::Tensor(t) => Some(t),
            _ => None,
        })
    }
    fn map_tensor<ReturnTensorType, F: FnOnce(TensorType) -> ReturnTensorType>(
        &self,
        f: F,
    ) -> Result<TensorTypeOrScalarEnum<ReturnTensorType>> {
        match self.tensor_or_scalar()? {
            TensorTypeOrScalarEnum::Tensor(t) => Ok(TensorTypeOrScalarEnum::Tensor(f(t))),
            TensorTypeOrScalarEnum::Scalar(s) => Ok(TensorTypeOrScalarEnum::Scalar(s)),
        }
    }
}

impl TensorTypeOrScalar<OpTensor> for OpTensor {
    fn tensor_or_scalar(&self) -> Result<TensorTypeOrScalarEnum<OpTensor>> {
        Ok(TensorTypeOrScalarEnum::Tensor(self.clone()))
    }
}

impl TensorTypeOrScalar<Tensor> for Tensor {
    fn tensor_or_scalar(&self) -> Result<TensorTypeOrScalarEnum<Tensor>> {
        Ok(TensorTypeOrScalarEnum::Tensor(self.clone()))
    }
}

impl<T: Clone> TensorTypeOrScalar<T> for Result<TensorTypeOrScalarEnum<T>> {
    fn tensor_or_scalar(&self) -> Result<TensorTypeOrScalarEnum<T>> {
        match self {
            Ok(value) => Ok(value.clone()),
            Err(e) => Err(anyhow::anyhow!("{}", e)),
        }
    }
}

impl<T, U: TensorDType + num_traits::Float> TensorTypeOrScalar<T> for U {
    fn tensor_or_scalar(&self) -> Result<TensorTypeOrScalarEnum<T>> {
        Ok(TensorTypeOrScalarEnum::Scalar(
            self.to_f32()
                .ok_or(anyhow::anyhow!("Could not convert to f32"))?,
        ))
    }
}

impl<T: Clone> TensorTypeOrScalar<T> for TensorTypeOrScalarEnum<T> {
    fn tensor_or_scalar(&self) -> Result<TensorTypeOrScalarEnum<T>> {
        Ok(self.clone())
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use crate::{rvec, Device, Tensor};

    #[test]
    fn has_nan_works() {
        let device = Device::request_device(crate::DeviceRequest::GPU).unwrap();
        let rand = Tensor::randn::<f32, _>(0., 1., (1, 1500, 384), device.clone(), false).unwrap();
        let nans = Tensor::from_data(vec![f32::NAN; 1500 * 384], (1, 1500, 384), device, false);

        let bingo = Tensor::cat(rvec![rand, nans], 2).unwrap();

        let result = bingo.to(&Device::CPU).unwrap();
        println!("RESULT: {result:?}");
        assert!(result.has_nan::<f32>());
    }
}
