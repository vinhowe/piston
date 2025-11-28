//! Operation pattern classification for fusion analysis.

use crate::LazyOp;

/// Classification of operations for fusion analysis.
///
/// Maps to TVM's `OpPatternKind` with simplifications for WebGPU.
/// The ordering matters: higher values represent more "dominant" patterns
/// that determine the fusion group's behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum OpPattern {
    /// Element-wise operations with 1:1 input-output mapping and same shape.
    ///
    /// Examples: `relu`, `gelu`, `exp`, `add`, `mul` (when shapes match)
    ///
    /// Fusion: Can fuse with anything that comes before.
    Elemwise = 0,

    /// Injective operations: 1:1 mapping but shapes may differ.
    ///
    /// Examples: `broadcast`, `permute`, `slice`, `reshape`, `view`
    ///
    /// Each output element can be computed from exactly one input element.
    /// Fusion: Can absorb elemwise on either side.
    Injective = 1,

    /// Reduction operations with N:1 mapping.
    ///
    /// Examples: `sum`, `mean`, `max`, `softmax`, `layernorm`
    ///
    /// Fusion rules:
    /// - CAN fuse elemwise/injective ops BEFORE the reduction (prologue)
    /// - CANNOT fuse ops AFTER the reduction (would require materialization)
    Reduce = 2,

    /// Compute-intensive operations with specialized memory access.
    ///
    /// Examples: `matmul`, `conv`, `gemm`
    ///
    /// Fusion rules:
    /// - CAN fuse elemwise AFTER as epilogue (bias add, activation)
    /// - CANNOT be fused INTO other ops (tiling would change)
    ComputeIntensive = 3,

    /// Opaque operations that cannot participate in fusion.
    ///
    /// Examples: `gather`, `scatter`, `topk`, `multinomial`, custom ops
    ///
    /// These have irregular memory access patterns or side effects.
    Opaque = 4,
}

impl OpPattern {
    /// Returns true if this pattern can produce intermediate results
    /// that need not be materialized to global memory.
    pub fn can_produce_intermediate(&self) -> bool {
        matches!(self, OpPattern::Elemwise | OpPattern::Injective)
    }

    /// Returns true if this pattern can consume intermediate results
    /// without reading from global memory.
    pub fn can_consume_intermediate(&self) -> bool {
        matches!(
            self,
            OpPattern::Elemwise | OpPattern::Injective | OpPattern::Reduce
        )
    }

    /// Returns the "dominant" pattern when combining two patterns.
    /// The dominant pattern determines the fusion group's codegen strategy.
    pub fn dominant(self, other: OpPattern) -> OpPattern {
        std::cmp::max(self, other)
    }
}

impl LazyOp {
    /// Classify this operation for fusion analysis.
    pub fn pattern(&self) -> OpPattern {
        match self {
            // === Elemwise: 1:1, shape-preserving ===
            LazyOp::Unary(_) => OpPattern::Elemwise,
            LazyOp::Binary(_) => OpPattern::Elemwise, // TODO: Check if shapes match
            LazyOp::Cast(_) => OpPattern::Elemwise,
            LazyOp::Cmp(_) => OpPattern::Elemwise,
            LazyOp::Powf(_) => OpPattern::Elemwise,
            LazyOp::Ternary(_) => OpPattern::Elemwise,
            LazyOp::Lerp(_) => OpPattern::Elemwise,
            LazyOp::WhereCond(_) => OpPattern::Elemwise,
            LazyOp::Affine(_) => OpPattern::Elemwise, // scale + bias is elemwise

            // === Injective: 1:1 but shape may change ===
            LazyOp::Reindex(_) => OpPattern::Injective, // Permute, Slice, Broadcast, Flip
            LazyOp::View(_) => OpPattern::Injective,
            LazyOp::Concat(_) => OpPattern::Injective,

            // === Reduce: N:1 mapping ===
            LazyOp::Reduce(_) => OpPattern::Reduce,
            LazyOp::Softmax(_) => OpPattern::Reduce, // Internal max + sum
            LazyOp::Norm(_) => OpPattern::Reduce,    // LayerNorm/RMSNorm have reductions

            // === Compute-intensive: special tiling ===
            LazyOp::Matmul(_) => OpPattern::ComputeIntensive,
            LazyOp::Conv(_) => OpPattern::ComputeIntensive,

            // === Opaque: irregular access or side effects ===
            LazyOp::Gather(_) | LazyOp::Select(_) => OpPattern::Opaque,
            LazyOp::TopK(_) | LazyOp::Multinomial(_) => OpPattern::Opaque,
            LazyOp::IndexWrite(_) | LazyOp::IndexAdd(_) | LazyOp::ScatterAdd(_) => {
                OpPattern::Opaque
            }
            LazyOp::RoPE(_) | LazyOp::Alibi(_) => OpPattern::Opaque,
            LazyOp::Cache(_) => OpPattern::Opaque,
            LazyOp::Const | LazyOp::Detach(_) | LazyOp::Copy(_) => OpPattern::Opaque,

            // === Generators (injective from indices) ===
            LazyOp::FillPointwise(_) | LazyOp::Arange(_) | LazyOp::Eye(_) => OpPattern::Injective,
            LazyOp::Bernoulli(_) | LazyOp::Trilu(_) | LazyOp::OneHot(_) => OpPattern::Injective,
        }
    }

    /// Returns true if this operation can be fused as an epilogue to GEMM.
    ///
    /// Epilogue ops are cheap element-wise operations that can be applied
    /// directly to GEMM output tiles before writing to global memory.
    pub fn is_gemm_epilogue_candidate(&self) -> bool {
        match self {
            LazyOp::Unary(u) => matches!(
                u.op,
                crate::UnaryOp::Relu
                    | crate::UnaryOp::Gelu
                    | crate::UnaryOp::Silu
                    | crate::UnaryOp::Sigmoid
                    | crate::UnaryOp::Tanh
            ),
            LazyOp::Binary(b) => matches!(
                b.op,
                crate::BinaryOp::Add | crate::BinaryOp::Mul
            ),
            LazyOp::Affine(_) => true,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_ordering() {
        assert!(OpPattern::Elemwise < OpPattern::Injective);
        assert!(OpPattern::Injective < OpPattern::Reduce);
        assert!(OpPattern::Reduce < OpPattern::ComputeIntensive);
        assert!(OpPattern::ComputeIntensive < OpPattern::Opaque);
    }

    #[test]
    fn test_pattern_dominant() {
        assert_eq!(
            OpPattern::Elemwise.dominant(OpPattern::Elemwise),
            OpPattern::Elemwise
        );
        assert_eq!(
            OpPattern::Elemwise.dominant(OpPattern::Reduce),
            OpPattern::Reduce
        );
        assert_eq!(
            OpPattern::ComputeIntensive.dominant(OpPattern::Elemwise),
            OpPattern::ComputeIntensive
        );
    }
}

