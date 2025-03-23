use piston::{rvec, Tensor};
use piston_macros::scoped_module;

use crate::Module;

/// Implements sinusoidal positional encodings as described in "Attention Is All You Need".
#[derive(Clone, Debug)]
pub struct SinusoidalEmbedding {
    #[allow(dead_code)]
    dim: usize,
    inv_freq: Tensor,
}

pub struct SinusoidalInput {
    pub input: Tensor,
    pub offset: usize,
}

impl SinusoidalEmbedding {
    pub fn new(dim: usize, device: &piston::Device) -> anyhow::Result<Self> {
        // Create position frequencies
        let mut freqs = Vec::with_capacity(dim / 2);
        for i in (0..dim).step_by(2) {
            freqs.push(1.0 / (10000f32.powf(i as f32 / dim as f32)));
        }

        // Create inverse frequency tensor
        let inv_freq = Tensor::from_data(&freqs, dim / 2, device.clone());

        Ok(Self { dim, inv_freq })
    }
}

#[scoped_module]
impl Module for SinusoidalEmbedding {
    type Input = SinusoidalInput;
    type Output = Tensor;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Self::Output> {
        let SinusoidalInput { input, offset } = input;

        // Get sequence length from input shape
        let seq_len = input.shape()[1];

        // Create position sequence [0, 1, 2, ..., seq_len-1]
        let pos_seq =
            Tensor::arange::<f32>(offset as f32, (offset + seq_len) as f32, input.device())?;

        // Compute outer product between positions and frequencies
        let sinusoid_inp =
            pos_seq
                .unsqueeze(1)?
                .matmul(self.inv_freq.clone().unsqueeze(0)?, false, false)?;

        // Compute sin and cos
        let sin = sinusoid_inp.clone().sin()?;
        let cos = sinusoid_inp.cos()?;

        let last_dim = sin.shape().len() - 1;
        // Interleave sin and cos values
        let pos_emb = Tensor::cat(rvec![sin, cos], last_dim)?;

        // Add batch dimension if needed
        let pos_emb = pos_emb.unsqueeze(0)?;

        // Add the positional embeddings to the input
        input + pos_emb
    }
}
