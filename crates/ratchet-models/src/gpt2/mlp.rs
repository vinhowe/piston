use maybe_async::maybe_async;
use ratchet::Tensor;
use ratchet_nn::{Linear, Module, VarBuilder};

use super::{
    linear::{linear_gpt2, linear_gpt2_residual},
    model::Config,
};

#[derive(Debug)]
pub struct MLP {
    c_fc: Linear,
    c_proj: Linear,
    hidden_act: ratchet_nn::Activation,
}

impl MLP {
    #[maybe_async]
    pub async fn new(cfg: &Config, vb: VarBuilder<'_>) -> anyhow::Result<Self> {
        Ok(Self {
            c_fc: linear_gpt2(cfg.n_embd, 4 * cfg.n_embd, vb.pp("c_fc")).await?,
            c_proj: linear_gpt2_residual(4 * cfg.n_embd, cfg.n_embd, cfg.n_layer, vb.pp("c_proj"))
                .await?,
            hidden_act: cfg.hidden_act,
        })
    }
}

impl Module for MLP {
    type Input = Tensor;
    type Output = Tensor;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Self::Output> {
        let x = self.c_fc.schedule(input)?;
        let x = self.hidden_act.schedule(x)?;
        self.c_proj.schedule(x)
    }
}
