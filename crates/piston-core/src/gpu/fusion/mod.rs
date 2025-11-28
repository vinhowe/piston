//! # Operator Fusion IR
//!
//! TVM-inspired fusion analysis and codegen for Piston.
//!
//! ## Overview
//!
//! The fusion module provides:
//! - [`OpPattern`] - Classification of operations by fusion compatibility
//! - [`FusionGraph`] - DAG representation of the computation graph
//! - [`FusionGroup`] - A set of operations that will be fused into one kernel
//!
//! ## Fusion Rules (TVM-style)
//!
//! 1. **Elementwise chains**: `[elem] -> [elem] -> [elem]` fuses into one kernel
//! 2. **Injective fusion**: `[inject] -> [elem]` or `[elem] -> [inject]` can fuse
//! 3. **Reduction prologue**: `[elem] -> [reduce]` fuses, but `[reduce] -> [elem]` does NOT
//! 4. **Compute epilogue**: `[gemm] -> [elem]` fuses (bias+activation in GEMM)
//! 5. **Opaque barrier**: Opaque ops block all fusion

mod analysis;
mod codegen;
mod graph;
mod group;
mod pattern;

pub use analysis::*;
pub use codegen::*;
pub use graph::*;
pub use group::*;
pub use pattern::*;
