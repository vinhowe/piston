use crate::cpu::cpu_store_result;
use crate::{Binary, BinaryOp, CPUOperation, DType, OperationError, Tensor, TensorDType};
use core::marker::PhantomData;
use half::{bf16, f16};
use maybe_async::maybe_async;
use num_traits::NumOps;

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

#[inline]
pub(crate) fn binary_map_inplace<T: TensorDType>(lhs: &mut [T], rhs: &[T], f: fn(T, T) -> T) {
    assert_eq!(lhs.len(), rhs.len());
    lhs.iter_mut().zip(rhs.iter()).for_each(|(l, r)| {
        *l = f(*l, *r);
    });
}

#[inline]
#[maybe_async]
pub(crate) async fn binary_apply<T: TensorDType, U: TensorDType>(
    lhs: &Tensor,
    rhs: &Tensor,
    dst: &Tensor,
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
pub(crate) async fn binary_apply_inplace<T: TensorDType>(
    lhs: &Tensor,
    rhs: &Tensor,
    dst: &Tensor,
    f: fn(T, T) -> T,
) -> Result<(), OperationError> {
    let mut lhs = lhs.to_vec::<T>().await?;
    let rhs = rhs.to_vec::<T>().await?;
    binary_map_inplace(&mut lhs, &rhs, f);
    cpu_store_result(dst, &lhs);
    Ok(())
}

pub struct BinaryOps<T: TensorDType> {
    dtype: PhantomData<T>,
}

macro_rules! impl_cpu_binary_op {
    ($method_name:ident, $dtype:ident, $op:expr) => {
        #[maybe_async]
        async fn $method_name(
            lhs: &Tensor,
            rhs: &Tensor,
            dst: Tensor,
        ) -> Result<Tensor, OperationError> {
            binary_apply_inplace::<$dtype>(lhs, rhs, &dst, $op).await?;
            Ok(dst)
        }
    };
}

macro_rules! cpu_binary_op_fn {
    ($method_name:ident, $op:expr) => {
        #[inline]
        pub(crate) fn $method_name<T: TensorDType + NumOps>(lhs: &mut [T], rhs: &[T]) {
            binary_map_inplace::<T>(lhs, rhs, $op);
        }
    };
}

cpu_binary_op_fn!(add, |lhs, rhs| lhs + rhs);
cpu_binary_op_fn!(sub, |lhs, rhs| lhs - rhs);
cpu_binary_op_fn!(mul, |lhs, rhs| lhs * rhs);
cpu_binary_op_fn!(div, |lhs, rhs| lhs / rhs);

macro_rules! impl_cpu_binary {
    ($dtype:ident) => {
        impl BinaryOps<$dtype> {
            impl_cpu_binary_op!(add, $dtype, |lhs, rhs| lhs + rhs);
            impl_cpu_binary_op!(sub, $dtype, |lhs, rhs| lhs - rhs);
            impl_cpu_binary_op!(mul, $dtype, |lhs, rhs| lhs * rhs);
            impl_cpu_binary_op!(div, $dtype, |lhs, rhs| lhs / rhs);

            #[maybe_async]
            pub async fn apply(op: &Binary, dst: Tensor) -> Result<Tensor, OperationError> {
                match op.op() {
                    BinaryOp::Add => Self::add(op.lhs(), op.rhs(), dst).await,
                    BinaryOp::Sub => Self::sub(op.lhs(), op.rhs(), dst).await,
                    BinaryOp::Mul => Self::mul(op.lhs(), op.rhs(), dst).await,
                    BinaryOp::Div => Self::div(op.lhs(), op.rhs(), dst).await,
                }
            }
        }
    };
}

#[maybe_async(AFIT)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait)]
impl CPUOperation for Binary {
    #[maybe_async]
    async fn apply_cpu(&self, dst: Tensor) -> Result<Tensor, OperationError> {
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
