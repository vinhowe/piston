mod affine;
mod alibi;
mod arange;
mod bernoulli;
mod binary;
mod cache;
mod cast;
mod cmp;
mod concat;
mod conv;
mod eye;
mod fill_pointwise;
mod gather;
mod index_add;
mod index_write;
mod lerp;
mod matmul;
mod multinomial;
mod norm;
mod one_hot;
mod powf;
mod reduce;
mod reindex;
mod rope;
mod scatter_add;
mod select;
mod softmax;
mod ternary;
mod topk;
mod trilu;
mod unary;
mod view;
mod where_cond;

use std::sync::Arc;

pub use affine::*;
pub use alibi::*;
pub use arange::*;
pub use bernoulli::*;
pub use binary::*;
pub use cache::*;
pub use cast::*;
pub use cmp::*;
pub use concat::*;
pub use conv::*;
pub use eye::*;
pub use fill_pointwise::*;
pub use gather::*;
pub use index_add::*;
pub use index_write::*;
pub use lerp::*;
pub use matmul::*;
pub use multinomial::*;
pub use norm::*;
pub use one_hot::*;
use piston_macros::IrFields;
pub use powf::*;
pub use reduce::*;
pub use reindex::*;
pub use rope::*;
pub use scatter_add::*;
pub use select::*;
pub use softmax::*;
pub use ternary::*;
pub use topk::*;
pub use trilu::*;
pub use unary::*;
pub use view::*;
pub use where_cond::*;

use crate::{
    Compiled, CompiledCopy, CopyCompileKey, OpGuards, OpTensor, Operation, OperationError, RVec,
    Storage, StorageView, rvec,
};

/// # KernelElement
///
/// Used to select the largest possible data type for a kernel.
/// If (dimension of interest % KE) == 0, it is safe to use.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum KernelElement {
    Vec4,
    Vec2,
    Scalar,
}

impl KernelElement {
    pub fn as_size(&self) -> usize {
        self.into()
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            KernelElement::Vec4 => "vec4",
            KernelElement::Vec2 => "vec2",
            KernelElement::Scalar => "scalar",
        }
    }
}

impl From<&KernelElement> for usize {
    fn from(item: &KernelElement) -> Self {
        match item {
            KernelElement::Vec4 => 4,
            KernelElement::Vec2 => 2,
            KernelElement::Scalar => 1,
        }
    }
}

#[derive(Debug, derive_new::new, Clone, IrFields)]
pub struct TensorCopy {
    pub src: OpTensor,
    pub dst: OpTensor,
}

impl TensorCopy {
    pub fn src(&self) -> &OpTensor {
        &self.src
    }

    pub fn dst(&self) -> &OpTensor {
        &self.dst
    }

    pub fn compile_gpu(&self) -> Result<Compiled, OperationError> {
        // Ensure we are running on GPU.
        if !self.src.device().is_gpu() || !self.dst.device().is_gpu() {
            panic!("copy_from only supported for GPU tensors");
        }
        // Sanity check: shape and dtype should match.
        if self.src.shape() != self.dst.shape() || self.src.dtype() != self.dst.dtype() {
            panic!("Shape or dtype mismatch for copy_from");
        }
        // Retrieve the underlying GPU buffers.
        let src_buffer = match self.src.storage().as_ref() {
            Some(Storage::GPU(gpu_buf)) => gpu_buf.inner.clone(),
            _ => panic!("Source tensor has no GPU storage"),
        };
        let dst_buffer = match self.dst.storage().as_ref() {
            Some(Storage::GPU(gpu_buf)) => gpu_buf.inner.clone(),
            _ => panic!("Destination tensor has no GPU storage"),
        };

        let size = self.src.num_bytes() as u64;

        Ok(Compiled::Copy(CompiledCopy::new(
            Arc::new(src_buffer),
            Arc::new(dst_buffer),
            size,
        )))
    }

    pub fn create_gpu_compile_key(&self) -> CopyCompileKey<'_> {
        CopyCompileKey {
            src: &self.src,
            dst: &self.dst,
        }
    }
}

impl OpGuards for TensorCopy {
    fn check_shapes(&self) {
        let (src_shape, dst_shape) = (self.src.shape(), self.dst.shape());
        assert_eq!(src_shape.dim(), dst_shape.dim());
        assert_eq!(src_shape.numel(), dst_shape.numel());
    }

    fn check_dtypes(&self) {}
}

impl Operation for TensorCopy {
    fn name(&self) -> &'static str {
        "TensorCopy"
    }

    fn srcs(&self) -> RVec<&OpTensor> {
        rvec![&self.src]
    }

    fn compute_view(&self) -> Result<StorageView, crate::OperationError> {
        Ok(self.src.storage_view().clone())
    }
}
