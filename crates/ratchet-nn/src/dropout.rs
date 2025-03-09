use crate::{current_module_mode, Module, ModuleMode};
use derive_new::new;
use ratchet::Tensor;
use ratchet_macros::scoped_module;

/// Dropout layer that randomly zeroes some of the elements of the input tensor with probability `p`
/// during training, and rescales the remaining elements by a factor of `1/(1-p)`.
///
/// This is a common regularization technique used in neural networks to prevent overfitting.
#[derive(Debug, Clone, new)]
pub struct Dropout {
    /// Probability of an element to be zeroed. Value range: (0, 1)
    p: f32,
}

#[scoped_module]
impl Module for Dropout {
    type Input = Tensor;
    type Output = Tensor;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Self::Output> {
        if self.p <= 0.0 || current_module_mode() == ModuleMode::Eval {
            return Ok(input);
        }

        // Create a tensor of probabilities (1-p) to keep elements
        let keep_prob = 1.0 - self.p;
        let probs = Tensor::full::<f32>(&input.shape(), keep_prob, &input.device());

        // Apply bernoulli sampling to get binary mask
        let mask = probs.bernoulli()?;

        // Scale the mask by 1/(1-p) to maintain the expected value of the output
        let scale = 1.0 / keep_prob;
        let scaled_mask = mask.affine(scale, 0.0)?;

        // Apply the mask by multiplying element-wise with the input
        input.mul(scaled_mask)
    }
}
