mod affine;
mod alibi;
mod arange;
mod binary;
mod cache;
mod cast;
mod cmp;
mod concat;
mod conv;
mod fill_constant;
mod fill_randn;
mod gather;
mod index_add;
mod index_write;
mod matmul;
mod norm;
mod powf;
mod reduce;
mod reindex;
mod rope;
mod scatter_add;
mod select;
mod softmax;
mod trilu;
mod unary;
mod view;
mod where_cond;

use std::sync::Arc;

pub use affine::*;
pub use alibi::*;
pub use arange::*;
pub use binary::*;
pub use cache::*;
pub use cast::*;
pub use cmp::*;
pub use concat::*;
pub use conv::*;
pub use fill_constant::*;
pub use fill_randn::*;
pub use gather::*;
pub use index_add::*;
pub use index_write::*;
pub use matmul::*;
pub use norm::*;
pub use powf::*;
use ratchet_macros::IrFields;
pub use reduce::*;
pub use reindex::*;
pub use rope::*;
pub use scatter_add::*;
pub use select::*;
pub use softmax::*;
pub use trilu::*;
pub use unary::*;
pub use view::*;
pub use where_cond::*;

use crate::{
    rvec, Compiled, CompiledCopy, CopyCompileKey, OpGuards, Operation, OperationError, RVec,
    Storage, StorageView, Tensor,
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
    pub src: Tensor,
    pub dst: Tensor,
}

impl TensorCopy {
    pub fn src(&self) -> &Tensor {
        &self.src
    }

    pub fn dst(&self) -> &Tensor {
        &self.dst
    }

    pub fn compile_gpu(&self) -> Result<Compiled, OperationError> {
        // Ensure we are running on GPU.
        if !self.src.device().is_gpu() || !self.dst.device().is_gpu() {
            panic!("copy_from only supported for GPU tensors");
        }
        // Sanity check: shape and dtype should match.
        if self.src.shape() != self.dst.shape() || self.src.dt() != self.dst.dt() {
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

    pub fn create_gpu_compile_key(&self) -> CopyCompileKey {
        CopyCompileKey {
            src: &self.src,
            dst: &self.dst,
        }
    }
}

impl OpGuards for TensorCopy {
    fn check_shapes(&self) {
        let (src_shape, dst_shape) = (self.src.shape(), self.dst.shape());
        assert_eq!(src_shape.rank(), dst_shape.rank());
        assert_eq!(src_shape.numel(), dst_shape.numel());
    }

    fn check_dtypes(&self) {}
}

impl Operation for TensorCopy {
    fn name(&self) -> &'static str {
        "TensorCopy"
    }

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.src]
    }

    fn compute_view(&self) -> Result<StorageView, crate::OperationError> {
        Ok(self.src.storage_view().clone())
    }
}
