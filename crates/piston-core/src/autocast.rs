//! Automatic Mixed Precision (AMP) support for tensor operations.
//!
//! This module provides thread-local autocast state and helpers for automatically
//! casting FP32 tensors to FP16 during forward passes, similar to PyTorch's autocast.

use std::cell::Cell;

use crate::{DType, OpTensor, cast_kernel};
use anyhow::Result;

thread_local! {
    static AUTOCAST_ENABLED: Cell<bool> = const { Cell::new(false) };
}

/// Returns whether autocast is currently enabled for this thread.
pub fn is_autocast_enabled() -> bool {
    AUTOCAST_ENABLED.with(|enabled| enabled.get())
}

/// Sets the autocast enabled state for this thread.
pub fn set_autocast_enabled(enabled: bool) {
    AUTOCAST_ENABLED.with(|cell| cell.set(enabled));
}

/// Conditionally casts a tensor to FP16 if autocast is enabled and the tensor is FP32.
///
/// This is called by tensor operations annotated with `#[tensor_op(autocast)]` to
/// automatically downcast inputs during mixed-precision training.
///
/// # Arguments
/// * `tensor` - The input tensor to potentially cast
///
/// # Returns
/// * The original tensor if autocast is disabled or tensor is not FP32
/// * A new FP16 tensor if autocast is enabled and tensor is FP32
pub fn maybe_autocast(tensor: OpTensor) -> Result<OpTensor> {
    if is_autocast_enabled() && tensor.dtype() == DType::F32 {
        cast_kernel(tensor, DType::F16)
    } else {
        Ok(tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autocast_toggle() {
        assert!(!is_autocast_enabled());
        set_autocast_enabled(true);
        assert!(is_autocast_enabled());
        set_autocast_enabled(false);
        assert!(!is_autocast_enabled());
    }
}
