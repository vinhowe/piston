use crate::cpu::cpu_store_result;
use crate::{
    Binary, BinaryOp, CPUOperation, DType, OpTensor, OperationError, TensorDType,
    TensorTypeOrScalar, TensorTypeOrScalarEnum,
};
use core::marker::PhantomData;
use half::{bf16, f16};
use maybe_async::maybe_async;
use num_traits::NumOps;

// Helper function to cast f32 scalar to the target type T
fn cast_scalar_to_type<T: TensorDType>(scalar: f32) -> T {
    match T::dtype() {
        DType::F32 => unsafe { std::mem::transmute_copy(&scalar) },
        DType::F16 => unsafe { std::mem::transmute_copy(&f16::from_f32(scalar)) },
        DType::BF16 => unsafe { std::mem::transmute_copy(&bf16::from_f32(scalar)) },
        _ => panic!("Unsupported scalar cast to type: {:?}", T::dtype()),
    }
}

#[inline]
pub(crate) fn binary_map<T: TensorDType, U: TensorDType>(
    lhs: &[T],
    rhs: &[T],
    dst: &mut [U],
    f: fn(T, T) -> U,
) {
    assert_eq!(lhs.len(), dst.len());
    assert_eq!(rhs.len(), dst.len());
    for ((l, r), d) in lhs
        .iter()
        .copied()
        .zip(rhs.iter().copied())
        .zip(dst.iter_mut())
    {
        *d = f(l, r);
    }
}

pub(crate) fn binary_map_scalar<T: TensorDType, U: TensorDType>(
    lhs: &[T],
    rhs: T,
    dst: &mut [U],
    f: fn(T, T) -> U,
) {
    assert_eq!(lhs.len(), dst.len());
    for (l, d) in lhs.iter().copied().zip(dst.iter_mut()) {
        *d = f(l, rhs);
    }
}

#[inline]
pub(crate) fn binary_map_inplace<T: TensorDType>(lhs: &mut [T], rhs: &[T], f: fn(T, T) -> T) {
    assert_eq!(lhs.len(), rhs.len());
    lhs.iter_mut().zip(rhs.iter()).for_each(|(l, r)| {
        *l = f(*l, *r);
    });
}

pub(crate) fn binary_map_scalar_inplace<T: TensorDType>(lhs: &mut [T], rhs: T, f: fn(T, T) -> T) {
    lhs.iter_mut().for_each(|l| {
        *l = f(*l, rhs);
    });
}

#[inline]
#[maybe_async]
pub(crate) async fn binary_apply<T: TensorDType, U: TensorDType>(
    lhs: &OpTensor,
    rhs: &OpTensor,
    dst: &OpTensor,
    f: fn(T, T) -> U,
) -> Result<(), OperationError> {
    let lhs = lhs.to_vec::<T>().await?;
    let rhs = rhs.to_vec::<T>().await?;
    let mut result = vec![U::zero(); dst.shape().numel()];
    binary_map(&lhs, &rhs, &mut result, f);
    cpu_store_result(dst, &result);
    Ok(())
}

#[inline]
#[maybe_async]
pub(crate) async fn binary_apply_inplace<T: TensorDType, RHS: TensorTypeOrScalar<OpTensor>>(
    lhs: &OpTensor,
    rhs: &RHS,
    dst: &OpTensor,
    f: fn(T, T) -> T,
) -> Result<(), OperationError> {
    let mut lhs = lhs.to_vec::<T>().await?;
    match rhs.tensor_or_scalar()? {
        TensorTypeOrScalarEnum::Tensor(rhs) => {
            let rhs = rhs.to_vec::<T>().await?;
            binary_map_inplace(&mut lhs, &rhs, f);
        }
        TensorTypeOrScalarEnum::Scalar(rhs) => {
            let rhs_typed = cast_scalar_to_type::<T>(rhs);
            binary_map_scalar_inplace(&mut lhs, rhs_typed, f);
        }
    }
    cpu_store_result(dst, &lhs);
    Ok(())
}

pub struct BinaryOps<T: TensorDType> {
    dtype: PhantomData<T>,
}

macro_rules! impl_cpu_binary_op {
    ($method_name:ident, $dtype:ident, $op:expr) => {
        #[maybe_async]
        async fn $method_name<RHS: TensorTypeOrScalar<OpTensor>>(
            lhs: &OpTensor,
            rhs: &RHS,
            dst: &OpTensor,
        ) -> Result<OpTensor, OperationError> {
            binary_apply_inplace::<$dtype, RHS>(lhs, rhs, dst, $op).await?;
            Ok(dst.clone())
        }
    };
}

macro_rules! cpu_binary_op_fn {
    ($method_name:ident, $op:expr) => {
        #[inline]
        pub(crate) fn $method_name<T: TensorDType + NumOps + PartialOrd>(lhs: &mut [T], rhs: &[T]) {
            binary_map_inplace::<T>(lhs, rhs, $op);
        }
    };
}

cpu_binary_op_fn!(add, |lhs, rhs| lhs + rhs);
cpu_binary_op_fn!(sub, |lhs, rhs| lhs - rhs);
cpu_binary_op_fn!(mul, |lhs, rhs| lhs * rhs);
cpu_binary_op_fn!(div, |lhs, rhs| lhs / rhs);
cpu_binary_op_fn!(maximum, |lhs, rhs| if lhs > rhs { lhs } else { rhs });

macro_rules! impl_cpu_binary {
    ($dtype:ident) => {
        impl BinaryOps<$dtype> {
            impl_cpu_binary_op!(add, $dtype, |lhs, rhs| lhs + rhs);
            impl_cpu_binary_op!(sub, $dtype, |lhs, rhs| lhs - rhs);
            impl_cpu_binary_op!(mul, $dtype, |lhs, rhs| lhs * rhs);
            impl_cpu_binary_op!(div, $dtype, |lhs, rhs| lhs / rhs);
            impl_cpu_binary_op!(maximum, $dtype, |lhs, rhs| if lhs > rhs {
                lhs
            } else {
                rhs
            });
            impl_cpu_binary_op!(minimum, $dtype, |lhs, rhs| if lhs < rhs {
                lhs
            } else {
                rhs
            });
            #[maybe_async]
            pub async fn apply(op: &Binary, dst: OpTensor) -> Result<OpTensor, OperationError> {
                match op.op() {
                    BinaryOp::Add => Self::add(op.lhs(), op.rhs(), &dst).await,
                    BinaryOp::Sub => Self::sub(op.lhs(), op.rhs(), &dst).await,
                    BinaryOp::Mul => Self::mul(op.lhs(), op.rhs(), &dst).await,
                    BinaryOp::Div => Self::div(op.lhs(), op.rhs(), &dst).await,
                    BinaryOp::Pow => Err(OperationError::UnknownError(anyhow::anyhow!(
                        "Pow is not supported for CPU"
                    ))),
                    BinaryOp::Maximum => Self::maximum(op.lhs(), op.rhs(), &dst).await,
                    BinaryOp::Minimum => Self::minimum(op.lhs(), op.rhs(), &dst).await,
                }
            }
        }
    };
}

#[maybe_async(AFIT)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait)]
impl CPUOperation for Binary {
    #[maybe_async]
    async fn apply_cpu(&self, dst: OpTensor) -> Result<OpTensor, OperationError> {
        match dst.dtype() {
            DType::F32 => BinaryOps::<f32>::apply(self, dst).await,
            DType::F16 => BinaryOps::<f16>::apply(self, dst).await,
            DType::BF16 => BinaryOps::<bf16>::apply(self, dst).await,
            _ => todo!(),
        }
    }
}

impl_cpu_binary!(f32);
impl_cpu_binary!(f16);
impl_cpu_binary!(bf16);
