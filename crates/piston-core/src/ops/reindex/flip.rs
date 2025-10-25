use encase::ShaderType;
use piston_macros::{IrFields, WgslMetadata};

use crate::{OpGuards, OpTensor, Operation, OperationError, RVec, StorageView, Stride, rvec};

#[derive(Debug, WgslMetadata, ShaderType, derive_new::new)]
pub struct FlipMeta {
    src_shape: glam::UVec4,
    dst_shape: glam::UVec4,
    src_stride: glam::UVec4,
    dst_stride: glam::UVec4,
    src_numel: u32,
    dst_numel: u32,
    flip_mask: glam::UVec4,
}

#[derive(derive_new::new, Debug, Clone, IrFields)]
pub struct Flip {
    pub src: OpTensor,
    pub dims: RVec<usize>,
}

impl Flip {
    /// Promote dims to 4D indexing space by offsetting with pad length
    pub fn promote(&self) -> RVec<usize> {
        let pad_len = 4 - self.src.shape().dim();
        self.dims.iter().map(|&d| d + pad_len).collect()
    }
}

impl OpGuards for Flip {
    fn check_shapes(&self) {
        let rank = self.src.shape().dim();
        assert!(self.dims.iter().all(|&d| d < rank));
        // No duplicate dims
        let mut seen = std::collections::HashSet::new();
        assert!(self.dims.iter().all(|d| seen.insert(*d)));
    }

    fn check_dtypes(&self) {}
}

impl Operation for Flip {
    fn name(&self) -> &'static str {
        "Flip"
    }

    fn compute_view(&self) -> Result<StorageView, OperationError> {
        let shape = self.src.shape().clone();
        let stride = Stride::from(&shape);
        Ok(StorageView::new(shape, self.src.dtype(), stride))
    }

    fn srcs(&self) -> RVec<&OpTensor> {
        rvec![&self.src]
    }
}
