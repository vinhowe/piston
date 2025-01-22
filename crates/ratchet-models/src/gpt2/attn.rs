use std::{cell::RefCell, rc::Rc};

use maybe_async::maybe_async;
use ratchet::{prelude::shape, rvec, Tensor};
use ratchet_nn::{KVCache, Linear, Module, VarBuilder};

use super::{linear::linear_gpt2, linear::linear_gpt2_residual, model::Config};

#[derive(Debug)]
pub struct GPT2SelfAttention {
    c_attn: Linear,
    c_proj: Linear,
    softmax_scale: Tensor,
    n_embd: usize,
    n_head: usize,
    h_dim: usize,
}

impl GPT2SelfAttention {
    #[maybe_async]
    pub async fn new(cfg: &Config, vb: VarBuilder<'_>) -> anyhow::Result<Self> {
        let c_attn = linear_gpt2(cfg.n_embd, 3 * cfg.n_embd, vb.pp("c_attn")).await?;
        let c_proj =
            linear_gpt2_residual(cfg.n_embd, cfg.n_embd, cfg.n_layer, vb.pp("c_proj")).await?;
        let h_dim = cfg.head_dim();

        let softmax_scale = Tensor::full(&shape![1], 1.0 / (h_dim as f32).sqrt(), vb.device());

        Ok(Self {
            c_attn,
            c_proj,
            softmax_scale,
            n_head: cfg.n_head,
            n_embd: cfg.n_embd,
            h_dim,
        })
    }
}

pub struct GPT2AttnInput {
    pub input: Tensor,
    pub index_pos: usize,
    pub block_idx: usize,
    pub cache: Rc<RefCell<KVCache>>,
}

impl Module for GPT2SelfAttention {
    type Input = GPT2AttnInput;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let GPT2AttnInput {
            input,
            index_pos,
            block_idx,
            mut cache,
        } = input;
        let [batch_size, q_len, _]: [usize; 3] = input.shape().try_into()?;

        let qkv = self.c_attn.schedule(input)?;

        let query_pos = 0;
        let key_pos = self.n_embd;
        let value_pos = self.n_embd * 2;

        let q = qkv
            .clone()
            .slice(&[0..batch_size, 0..q_len, query_pos..self.n_embd])?;
        let k = qkv
            .clone()
            .slice(&[0..batch_size, 0..q_len, key_pos..key_pos + self.n_embd])?;
        let v =
            qkv.clone()
                .slice(&[0..batch_size, 0..q_len, value_pos..value_pos + self.n_embd])?;

        let qkv_shape = shape![batch_size as _, q_len, self.n_head, self.h_dim];

        let k = k.view(qkv_shape.clone())?.permute(&[0, 2, 1, 3])?;
        let q = q.view(qkv_shape.clone())?.permute(&[0, 2, 1, 3])?;
        let v = v.view(qkv_shape.clone())?.permute(&[0, 2, 1, 3])?;

        let att = q.matmul(k, false, true)?.mul(self.softmax_scale.clone())?;

        let att = if q_len <= 1 {
            att
        } else {
            // let mask = cache.mask(block_idx)?.broadcast_to(att.shape().clone())?;
            let mask = cache
                .borrow_mut()
                .mask(q_len)?
                .broadcast_to(att.shape().clone())?;
            // println!("mask: {:?}", mask);
            // println!("MASK MASK MASK dt: {:?}", mask.dt());
            masked_fill(&att, &mask, -1e9)?
        };

        let att = att.softmax(3)?;
        let y = att
            .matmul(v, false, false)?
            .permute(&[0, 2, 1, 3])?
            .view(shape![batch_size as _, q_len, self.n_embd])?;

        let y = self.c_proj.schedule(y)?;

        Ok(y)
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> anyhow::Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::full(shape, on_true, on_false.device());
    let m = mask.clone().where_cond(on_true, on_false.clone())?;
    Ok(m)
}
