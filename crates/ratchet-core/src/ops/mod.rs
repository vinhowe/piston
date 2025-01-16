mod affine;
mod binary;
mod cache;
mod cast;
mod cmp;
mod concat;
mod conv;
mod fill_constant;
mod fill_randn;
mod index_write;
mod matmul;
mod norm;
mod reindex;
mod rope;
mod select;
mod softmax;
mod trilu;
mod unary;
mod view;
mod where_cond;

pub use affine::*;
pub use binary::*;
pub use cache::*;
pub use cast::*;
pub use cmp::*;
pub use concat::*;
pub use conv::*;
pub use fill_constant::*;
pub use fill_randn::*;
pub use index_write::*;
pub use matmul::*;
pub use norm::*;
pub use reindex::*;
pub use rope::*;
pub use select::*;
pub use softmax::*;
pub use trilu::*;
pub use unary::*;
pub use view::*;
pub use where_cond::*;

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
