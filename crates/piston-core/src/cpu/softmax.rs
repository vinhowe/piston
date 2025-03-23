use crate::cpu::utils::cpu_store_result;
use crate::{CPUOperation, DType, OperationError, Softmax, Tensor, TensorDType};
use half::{bf16, f16};
use maybe_async::maybe_async;
use num::Float;
use num_traits::NumAssignOps;

#[maybe_async(AFIT)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait)]
impl CPUOperation for Softmax {
    #[maybe_async]
    async fn apply_cpu(&self, dst: Tensor) -> Result<Tensor, OperationError> {
        let Softmax { input, dim } = self;
        match input.dtype() {
            DType::F32 => softmax::<f32>(input, *dim, &dst).await?,
            DType::F16 => softmax::<f16>(input, *dim, &dst).await?,
            DType::BF16 => softmax::<bf16>(input, *dim, &dst).await?,
            _ => todo!(),
        }

        Ok(dst)
    }
}

#[maybe_async]
async fn softmax<T>(input: &Tensor, dim: usize, dst: &Tensor) -> Result<(), OperationError>
where
    T: TensorDType + Float + NumAssignOps,
{
    let src_shape = input.shape();
    let mut input = input.to_vec::<T>().await?;
    let N = src_shape[dim];
    input.chunks_mut(N).for_each(|chunk| {
        let mut sum = T::zero();
        for item in chunk.iter_mut().take(N) {
            *item = item.exp();
            sum += *item;
        }
        for item in chunk.iter_mut().take(N) {
            *item /= sum;
        }
    });

    cpu_store_result(dst, &input);

    Ok(())
}
