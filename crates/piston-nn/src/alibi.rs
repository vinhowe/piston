use piston::Tensor;
use piston_macros::scoped_module;

use crate::Module;

#[derive(Clone, Debug, derive_new::new)]
pub struct AlibiEmbedding {
    n_head: usize,
    max_bias: f32,
}

pub struct AlibiInput {
    pub input: Tensor,
}

#[scoped_module]
impl Module for AlibiEmbedding {
    type Input = AlibiInput;
    type Output = Tensor;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Self::Output> {
        let AlibiInput { input } = input;
        // To make broadcasting work...
        let alibi = Tensor::zeros::<f32, _>(&input.shape()[..3], &input.device(), false)?;
        input + alibi.alibi(self.max_bias)?.unsqueeze(3)
    }
}
