use std::ops::RangeInclusive;

use crate::{DType, Shape};

#[derive(Debug, thiserror::Error)]
pub enum InvariantError {
    #[error("Shape mismatch in {op}: lhs: {lhs:?}, rhs: {rhs:?}.")]
    ShapeMismatchBinaryOp {
        lhs: Shape,
        rhs: Shape,
        op: &'static str,
    },
    #[error("Rank mismatch. {accepted:?} != {actual}.")]
    RankMismatch {
        accepted: RangeInclusive<usize>,
        actual: usize,
    },
    #[error("Wrong input arity. Allowed range is {accepted:?}, node has {actual}.")]
    InputArity {
        accepted: RangeInclusive<usize>,
        actual: usize,
    },
    #[error("Wrong output arity. Allowed is {accepted:?}, node has {actual}.")]
    OutputArity {
        accepted: RangeInclusive<usize>,
        actual: usize,
    },
    #[error("DType mismatch, expected {expected:?}, got {actual:?}.")]
    DTypeMismatch { expected: DType, actual: DType },
    #[error("Unsupported DType {0:?}.")]
    UnsupportedDType(DType),
    #[error("Duplicate dims in permutation.")]
    DuplicateDims,
    #[error("Broadcasting failed: {0:?}")]
    BroadcastingFailed(Vec<Shape>),
    #[error("The expanded size of the tensor must match the existing size.")]
    InplaceBroadcast,
    #[error("Dim out of range {dim} in shape {shape:?}.")]
    DimOutOfRange { dim: usize, shape: Shape },
}
