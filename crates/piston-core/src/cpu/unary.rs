use crate::cpu::cpu_store_result;
use crate::{CPUOperation, DType, OpTensor, OperationError, TensorDType, Unary, UnaryOp};
use core::marker::PhantomData;
use half::{bf16, f16};
use maybe_async::maybe_async;
use num_traits::Float;

#[inline]
pub(crate) fn unary_apply_fn_helper<T: TensorDType, U: TensorDType>(
    src: &[T],
    dst: &mut [U],
    f: fn(T) -> U,
) {
    assert_eq!(src.len(), dst.len());
    for (s, d) in src.iter().copied().zip(dst.iter_mut()) {
        *d = f(s);
    }
}

#[inline]
pub(crate) fn unary_map_inplace<T: TensorDType>(src: &mut [T], f: fn(T) -> T) {
    for s in src.iter_mut() {
        *s = f(*s);
    }
}

#[inline]
#[maybe_async]
pub(crate) async fn unary_apply_fn<T: TensorDType, U: TensorDType>(
    input: &OpTensor,
    dst: &OpTensor,
    f: fn(T) -> U,
) -> Result<(), OperationError> {
    let input = input.to_vec::<T>().await?;
    let mut result = vec![U::zero(); dst.shape().numel()];
    unary_apply_fn_helper(&input, &mut result, f);
    cpu_store_result(dst, &result);
    Ok(())
}

struct UnaryOps<T: TensorDType> {
    dtype: PhantomData<T>,
}

macro_rules! impl_unary_ops {
    ($dtype:ident, $conv:expr) => {
        impl UnaryOps<$dtype> {
            impl_cpu_unary_op!(gelu, |x: $dtype| $conv(0.5)
                * x
                * ($conv(1.0)
                    + $dtype::tanh(
                        $conv(0.797_884_6) * x * ($conv(1.0) + $conv(0.044715) * x * x)
                    )));

            impl_cpu_unary_op!(tanh, |x: $dtype| x.tanh());
            impl_cpu_unary_op!(exp, |x: $dtype| x.exp());
            impl_cpu_unary_op!(log, |x: $dtype| x.ln());
            impl_cpu_unary_op!(sin, |x: $dtype| x.sin());
            impl_cpu_unary_op!(cos, |x: $dtype| x.cos());
            impl_cpu_unary_op!(abs, |x: $dtype| x.abs());
            impl_cpu_unary_op!(square, |x: $dtype| x * x);
            impl_cpu_unary_op!(sqrt, |x: $dtype| x.sqrt());
            impl_cpu_unary_op!(relu, |x: $dtype| x.max($conv(0.0)));
            impl_cpu_unary_op!(relu2, |x: $dtype| x.max($conv(0.0)) * x.max($conv(0.0)));
            impl_cpu_unary_op!(floor, |x: $dtype| x.floor());
            impl_cpu_unary_op!(ceil, |x: $dtype| x.ceil());
            impl_cpu_unary_op!(neg, |x: $dtype| -x);
            impl_cpu_unary_op!(silu, |x: $dtype| x / ($conv(1.0) + (-x).exp()));
            impl_cpu_unary_op!(sigmoid, |x: $dtype| $conv(1.0) / ($conv(1.0) + (-x).exp()));
            impl_cpu_unary_op!(reciprocal, |x: $dtype| $conv(1.0) / x);
            impl_cpu_unary_op!(swiglu, |x: $dtype| x
                * x
                * ($conv(1.0) / ($conv(1.0) + (-x).exp())));

            #[maybe_async]
            async fn apply(op: &Unary, dst: OpTensor) -> Result<OpTensor, OperationError> {
                match op.op() {
                    UnaryOp::Gelu => Self::gelu(op.input(), dst).await,
                    UnaryOp::Tanh => Self::tanh(op.input(), dst).await,
                    UnaryOp::Exp => Self::exp(op.input(), dst).await,
                    UnaryOp::Log => Self::log(op.input(), dst).await,
                    UnaryOp::Sin => Self::sin(op.input(), dst).await,
                    UnaryOp::Cos => Self::cos(op.input(), dst).await,
                    UnaryOp::Abs => Self::abs(op.input(), dst).await,
                    UnaryOp::Square => Self::square(op.input(), dst).await,
                    UnaryOp::Sqrt => Self::sqrt(op.input(), dst).await,
                    UnaryOp::Relu => Self::relu(op.input(), dst).await,
                    UnaryOp::Relu2 => Self::relu2(op.input(), dst).await,
                    UnaryOp::Floor => Self::floor(op.input(), dst).await,
                    UnaryOp::Ceil => Self::ceil(op.input(), dst).await,
                    UnaryOp::Neg => Self::neg(op.input(), dst).await,
                    UnaryOp::Reciprocal => Self::reciprocal(op.input(), dst).await,
                    UnaryOp::Silu => Self::silu(op.input(), dst).await,
                    UnaryOp::Sigmoid => Self::sigmoid(op.input(), dst).await,
                    UnaryOp::Swiglu => Self::swiglu(op.input(), dst).await,
                }
            }
        }
    };
}

macro_rules! impl_cpu_unary_op {
    ($method_name:ident, $op:expr) => {
        #[maybe_async]
        async fn $method_name(input: &OpTensor, dst: OpTensor) -> Result<OpTensor, OperationError> {
            unary_apply_fn(input, &dst, $op).await?;
            Ok(dst)
        }
    };
}

#[maybe_async(AFIT)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait)]
impl CPUOperation for Unary {
    #[maybe_async]
    async fn apply_cpu(&self, dst: OpTensor) -> Result<OpTensor, OperationError> {
        match dst.dtype() {
            DType::F32 => UnaryOps::<f32>::apply(self, dst).await,
            DType::F16 => UnaryOps::<f16>::apply(self, dst).await,
            DType::BF16 => UnaryOps::<bf16>::apply(self, dst).await,
            _ => todo!(),
        }
    }
}

macro_rules! impl_cpu_unary {
    ($dtype:ident) => {
        impl_cpu_unary!($dtype, |x: $dtype| -> $dtype { x });
    };
    ($dtype:ident, $conv:expr) => {
        impl_unary_ops!($dtype, $conv);
    };
}

impl_cpu_unary!(f32);
impl_cpu_unary!(f16, f16::from_f32);
impl_cpu_unary!(bf16, bf16::from_f32);
