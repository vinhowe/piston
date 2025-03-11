use crate::{
    cpu::cpu_store_result, CPUOperation, DType, InvariantError, Matmul, MatmulSpec, OperationError,
    Shape, Stride, Tensor, TensorDType,
};
use anyhow::{anyhow, Result};
use core::str::FromStr;
use gemm::{gemm as gemm_kernel, Parallelism};
use half::{bf16, f16};
use maybe_async::maybe_async;
use std::num::NonZeroUsize;

fn get_num_threads() -> NonZeroUsize {
    // Respond to the same environment variable as rayon.
    match std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|s| usize::from_str(&s).ok())
    {
        Some(x) if x > 0 => NonZeroUsize::new(x).unwrap(),
        Some(_) | None => std::thread::available_parallelism()
            .unwrap_or_else(|_| NonZeroUsize::new(1usize).unwrap()),
    }
}

fn get_parallelism() -> Parallelism {
    match get_num_threads().get() {
        1 => Parallelism::None,
        n => Parallelism::Rayon(n),
    }
}

fn calculate_skips(
    lhs_shape: &Shape,
    lhs_stride: &[isize],
    rhs_shape: &Shape,
    rhs_stride: &[isize],
    rank: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(usize, usize)> {
    let lhs_skip: usize = match lhs_stride[..rank - 2] {
        [s1, stride] if s1 == stride * lhs_shape[1] as isize => stride as usize,
        [_, stride] if lhs_shape[0] == 1 => stride as usize,
        [stride, _] if lhs_shape[1] == 1 => stride as usize,
        [stride] => stride as usize,
        [] => m * k,
        _ => Err(anyhow!("non-contiguous lhs"))?,
    };
    let rhs_skip: usize = match rhs_stride[..rank - 2] {
        [s1, stride] if s1 == stride * rhs_shape[1] as isize => stride as usize,
        [_, stride] if rhs_shape[0] == 1 => stride as usize,
        [stride, _] if rhs_shape[1] == 1 => stride as usize,
        [stride] => stride as usize,
        [] => n * k,
        _ => Err(anyhow!("non-contiguous rhs"))?,
    };
    Ok((lhs_skip, rhs_skip))
}

pub(crate) fn gemm<T: TensorDType>(
    lhs: &[T],
    lhs_shape: &Shape,
    lhs_stride: &Stride,
    rhs: &[T],
    rhs_shape: &Shape,
    rhs_stride: &Stride,
    dst_stride: &Stride,
    b: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<Vec<T>, OperationError> {
    let lhs_stride = lhs_stride.to_vec();
    let rhs_stride = rhs_stride.to_vec();
    let rank = lhs_shape.rank();

    let lhs_cs = lhs_stride[rank - 1];
    let lhs_rs = lhs_stride[rank - 2];

    let rhs_cs = rhs_stride[rank - 1];
    let rhs_rs = rhs_stride[rank - 2];

    let (lhs_skip, rhs_skip) = calculate_skips(
        lhs_shape,
        &lhs_stride,
        rhs_shape,
        &rhs_stride,
        rank,
        m,
        n,
        k,
    )?;
    let dst_skip: usize = m * n;
    let dst_rs = dst_stride[0];
    let dst_cs = dst_stride[1];

    let mut dst = vec![T::zero(); b * m * n];

    for step in 0..b {
        let lhs_p = &lhs[step * lhs_skip..];
        let rhs_p = &rhs[step * rhs_skip..];
        let dst_p = &mut dst[step * dst_skip..];
        unsafe {
            gemm_kernel(
                m,
                n,
                k,
                dst_p.as_mut_ptr(),
                dst_cs,
                dst_rs,
                false,
                lhs_p.as_ptr(),
                lhs_cs,
                lhs_rs,
                rhs_p.as_ptr(),
                rhs_cs,
                rhs_rs,
                T::zero(),
                T::one(),
                false,
                false,
                false,
                get_parallelism(),
            )
        }
    }
    Ok(dst)
}

fn gemm_impl<T: TensorDType>(
    spec: MatmulSpec,
    lhs: &[T],
    rhs: &[T],
) -> Result<Vec<T>, OperationError> {
    let lhs_shape = spec.lhs_shape();
    let rhs_shape = spec.rhs_shape();
    let lhs_stride = spec.lhs_stride();
    let rhs_stride = spec.rhs_stride();
    let dst_stride = spec.dst_stride();
    let b = spec.stacks();
    let m = spec.m();
    let n = spec.n();
    let k = spec.k();
    gemm(
        lhs, lhs_shape, lhs_stride, rhs, rhs_shape, rhs_stride, dst_stride, b, m, n, k,
    )
}

#[maybe_async(AFIT)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait)]
impl CPUOperation for Matmul {
    #[maybe_async]
    async fn apply_cpu(&self, dst: Tensor) -> Result<Tensor, OperationError> {
        #[maybe_async]
        async fn run_gemm<T: TensorDType>(
            spec: MatmulSpec,
            lhs: &Tensor,
            rhs: &Tensor,
            dst: &Tensor,
        ) -> Result<(), OperationError> {
            let lhs = lhs.to_vec::<T>().await?;
            let rhs = rhs.to_vec::<T>().await?;

            let result = if spec.trans_dst() {
                gemm_impl::<T>(spec, &rhs, &lhs)?
            } else {
                gemm_impl::<T>(spec, &lhs, &rhs)?
            };
            cpu_store_result(dst, &result);
            Ok(())
        }
        let spec = self.compute_spec();

        let Matmul { lhs, rhs, .. } = self;

        match self.lhs.dtype() {
            DType::F32 => run_gemm::<f32>(spec, lhs, rhs, &dst).await,
            DType::F16 => run_gemm::<f16>(spec, lhs, rhs, &dst).await,
            DType::BF16 => run_gemm::<bf16>(spec, lhs, rhs, &dst).await,
            dtype => Err(InvariantError::UnsupportedDType(dtype))?,
        }?;
        Ok(dst)
    }
}
