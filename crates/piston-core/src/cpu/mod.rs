mod binary;
pub mod gemm;
mod norm;
pub mod reindex;
pub mod rope;
mod softmax;
mod unary;
mod utils;

use crate::{
    dequantize, Cast, Concat, DType, IndexSelect, InvariantError, LazyOp, OpTensor, Operation,
    OperationError, RVec, Shape, TensorDType,
};
use anyhow::anyhow;
use half::{bf16, f16};
use maybe_async::maybe_async;
use rope::cpu_rope;
use unary::unary_apply_fn;
use utils::cpu_store_result;

#[maybe_async]
pub async fn apply_operation(op: LazyOp, dst: OpTensor) -> Result<OpTensor, OperationError> {
    match op {
        LazyOp::Binary(b) => b.apply_cpu(dst).await,
        LazyOp::Ternary(_t) => todo!(),
        LazyOp::Cast(c) => cpu_cast(c, dst).await,
        LazyOp::Matmul(m) => m.apply_cpu(dst).await,
        LazyOp::Softmax(s) => s.apply_cpu(dst).await,
        LazyOp::RoPE(r) => cpu_rope(r, dst).await,
        LazyOp::Alibi(_a) => todo!(),
        LazyOp::Unary(u) => u.apply_cpu(dst).await,
        LazyOp::Reindex(r) => r.apply_cpu(dst).await,
        LazyOp::Concat(c) => cpu_concat(c, dst).await,
        LazyOp::Norm(n) => n.apply_cpu(dst).await,
        LazyOp::Affine(_a) => todo!(),
        LazyOp::Lerp(_l) => todo!(),
        LazyOp::Cmp(_c) => todo!(),
        LazyOp::Powf(_p) => todo!(),
        LazyOp::Conv(_c) => todo!(),
        LazyOp::Select(i) => cpu_index_select(i, dst).await,
        LazyOp::IndexWrite(_i) => todo!(),
        LazyOp::Cache(_c) => todo!(),
        LazyOp::Trilu(_t) => todo!(),
        LazyOp::Const => todo!(),
        LazyOp::View(_) => todo!(),
        LazyOp::WhereCond(_w) => todo!(),
        LazyOp::Reduce(_r) => todo!(),
        LazyOp::Gather(_g) => todo!(),
        LazyOp::FillConstant(_f) => todo!(),
        LazyOp::FillRandn(_f) => todo!(),
        LazyOp::Bernoulli(_b) => todo!(),
        LazyOp::Arange(_a) => todo!(),
        LazyOp::IndexAdd(_i) => todo!(),
        LazyOp::ScatterAdd(_s) => todo!(),
        LazyOp::Detach(_d) => todo!(),
        LazyOp::Copy(_c) => todo!(),
    }
}

#[maybe_async(AFIT)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait)]
pub trait CPUOperation: Operation {
    async fn apply_cpu(&self, dst: OpTensor) -> Result<OpTensor, OperationError>;
}

#[maybe_async]
async fn index_select<T: TensorDType>(
    index_select: IndexSelect,
    dst: OpTensor,
) -> Result<OpTensor, OperationError> {
    let src = index_select.src();
    let indices = index_select.indices();
    let dim = index_select.dim();

    // TODO: Add support for other indexing types
    if !matches!(indices.dtype(), DType::I32) {
        return Err(InvariantError::DTypeMismatch {
            expected: DType::I32,
            actual: indices.dtype(),
        }
        .into());
    }

    let mut dst_dims = src.shape().to_vec();
    let indices_dims = indices.shape().to_vec();

    let src_dim = dst_dims[dim];
    let n_ids = indices_dims[0];
    dst_dims[dim] = n_ids;

    let dst_len: usize = dst_dims.iter().product();
    let left_len: usize = dst_dims[..dim].iter().product();
    let right_len: usize = dst_dims[dim + 1..].iter().product();

    let src = src.to_vec::<T>().await?;
    let indices = indices.to_vec::<i32>().await?;
    let mut result = vec![T::zero(); dst_len];

    for left_i in 0..left_len {
        let start_src_idx = left_i * right_len * src_dim;
        let start_dst_idx = left_i * right_len * n_ids;
        for (i, idx) in indices.iter().enumerate().take(n_ids) {
            let src_idx = start_src_idx + *idx as usize * right_len;
            let dst_idx = start_dst_idx + i * right_len;
            result[dst_idx..dst_idx + right_len]
                .copy_from_slice(&src[src_idx..src_idx + right_len]);
        }
    }
    cpu_store_result(&dst, &result);
    Ok(dst)
}

#[maybe_async]
async fn qindex_select(op: IndexSelect, dst: OpTensor) -> Result<OpTensor, OperationError> {
    // NOTE: qindex_select is functional but not optimized at all.
    // Currently we simply dequantize the entire input tensor to f32 and then call index_select.
    // Because of borrowing rules dequantizing also requires a deep clone of the input tensor, which is less than ideal.
    // In the future we would rather directly index the raw buffer of the quantized tensor and dequantize only what is required.
    // TODO: Add support for direct indexing + partial dequantization
    let src = op.src().deep_clone().await;

    // NOTE: Support for other quantization types is dependent on the corresponding dequantization functions.
    let src = dequantize(src);
    let indices = op.indices().clone();
    let dim = op.dim();

    index_select::<f32>(IndexSelect::new(src, indices, dim), dst).await
}

#[maybe_async]
pub async fn cpu_index_select(i: IndexSelect, dst: OpTensor) -> Result<OpTensor, OperationError> {
    match i.src().dtype() {
        DType::F32 => index_select::<f32>(i, dst).await,
        DType::F16 => index_select::<f16>(i, dst).await,
        DType::BF16 => index_select::<bf16>(i, dst).await,
        DType::Q8_0F(_) => qindex_select(i, dst).await,
        dtype => Err(InvariantError::UnsupportedDType(dtype).into()),
    }
}

#[maybe_async]
pub async fn cpu_cast(cast: Cast, dst: OpTensor) -> Result<OpTensor, OperationError> {
    if cast.input().dtype() == cast.dst_dtype() {
        return Ok(cast.input().clone());
    }
    match (cast.input().dtype(), cast.dst_dtype()) {
        // F32 ->
        (DType::F32, DType::F16) => {
            unary_apply_fn::<f32, f16>(cast.input(), &dst, f16::from_f32).await?
        }
        (DType::F32, DType::BF16) => {
            unary_apply_fn::<f32, bf16>(cast.input(), &dst, bf16::from_f32).await?
        }
        (DType::F32, DType::I32) => {
            unary_apply_fn::<f32, i32>(cast.input(), &dst, |x| x as i32).await?
        }
        (DType::F32, DType::U32) => {
            unary_apply_fn::<f32, u32>(cast.input(), &dst, |x| x as u32).await?
        }

        // F16 ->
        (DType::F16, DType::F32) => {
            unary_apply_fn::<f16, f32>(cast.input(), &dst, f32::from).await?
        }

        // BF16 ->
        (DType::BF16, DType::F32) => {
            unary_apply_fn::<bf16, f32>(cast.input(), &dst, f32::from).await?
        }

        // I32 ->
        (DType::I32, DType::F32) => {
            unary_apply_fn::<i32, f32>(cast.input(), &dst, |x| x as f32).await?
        }

        // U32 ->
        (DType::U32, DType::F32) => {
            unary_apply_fn::<u32, f32>(cast.input(), &dst, |x| x as f32).await?
        }

        _ => unimplemented!(
            "Cannot cast {:?} -> {:?}",
            cast.input().dtype(),
            cast.dst_dtype()
        ),
    };

    Ok(dst)
}

pub(crate) fn concat<T: TensorDType>(
    inputs: &[(Shape, Vec<T>)],
    dim: usize,
    dst_shape: &Shape,
    dst: &mut [T],
) -> Result<(), OperationError> {
    let dst_dim_len = dst_shape[dim];
    let block: usize = dst_shape.iter().skip(1 + dim).product();
    let dst_s = block * dst_dim_len;
    let src_o = 0;
    let mut dst_o = 0;
    for (src_s, src) in inputs {
        let a_dim: usize = src_s.iter().take(dim).product();
        let b_dim = block * src_s[dim];
        for idx in 0..a_dim {
            let dst_idx = idx * dst_s + dst_o;
            let src_idx = idx * b_dim + src_o;
            let dst_t = &mut dst[dst_idx..dst_idx + b_dim];
            let src = &src[src_idx..src_idx + b_dim];
            dst_t.copy_from_slice(src)
        }
        dst_o += b_dim;
    }
    Ok(())
}

#[maybe_async]
pub(crate) async fn apply_concat<T: TensorDType>(
    inputs: RVec<OpTensor>,
    dim: usize,
    dst: OpTensor,
) -> Result<OpTensor, OperationError> {
    let dst_size = dst.shape().numel();
    let mut result = vec![T::zero(); dst_size];

    let mut inputs_result = Vec::with_capacity(inputs.len());
    for t in inputs.iter() {
        let result: Result<_, OperationError> = match t.to_vec::<T>().await {
            Ok(v) => Ok((t.shape().clone(), v)),
            Err(e) => Err(e.into()),
        };
        inputs_result.push(result?);
    }
    let inputs = inputs_result;

    concat(&inputs, dim, dst.shape(), &mut result)?;
    cpu_store_result(&dst, &result);
    Ok(dst)
}

#[maybe_async]
pub async fn cpu_concat(
    Concat { inputs, dim }: Concat,
    dst: OpTensor,
) -> Result<OpTensor, OperationError> {
    match dst.dtype() {
        DType::F32 => apply_concat::<f32>(inputs, dim, dst).await,
        DType::F16 => apply_concat::<f16>(inputs, dim, dst).await,
        DType::BF16 => apply_concat::<bf16>(inputs, dim, dst).await,
        dtype => Err(InvariantError::UnsupportedDType(dtype).into()),
    }
}
