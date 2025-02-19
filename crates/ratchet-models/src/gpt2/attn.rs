use std::{cell::RefCell, rc::Rc};

use maybe_async::maybe_async;
use ratchet::{prelude::shape, rvec, Tensor};
use ratchet_nn::{
    AlibiEmbedding, AlibiInput, KVCache, Linear, Module, RotaryEmbedding, RotaryInput, VarBuilder,
};

use super::{
    linear::linear_gpt2,
    linear::linear_gpt2_residual,
    model::{Config, PositionalEncoding},
};

#[derive(Debug)]
pub struct GPT2SelfAttention {
    c_attn: Linear,
    c_proj: Linear,
    softmax_scale: Tensor,
    n_embd: usize,
    n_head: usize,
    h_dim: usize,
    rope: Option<RotaryEmbedding>,
    alibi: Option<AlibiEmbedding>,
}

impl GPT2SelfAttention {
    #[maybe_async]
    pub async fn new(cfg: &Config, vb: VarBuilder<'_>) -> anyhow::Result<Self> {
        let c_attn = linear_gpt2(cfg.n_embd, 3 * cfg.n_embd, vb.pp("c_attn")).await?;
        let c_proj =
            linear_gpt2_residual(cfg.n_embd, cfg.n_embd, cfg.n_layer, vb.pp("c_proj")).await?;
        let h_dim = cfg.head_dim();

        let softmax_scale = Tensor::full(&shape![1], 1.0 / (h_dim as f32).sqrt(), vb.device());

        let (rope, alibi) = match cfg.positional_encoding {
            PositionalEncoding::RoPE => {
                let rope_base = 10000.0f32;
                (
                    Some(RotaryEmbedding::new(h_dim, false, rope_base, 1.0)),
                    None,
                )
            }
            PositionalEncoding::ALiBi => (
                None,
                Some(AlibiEmbedding::new(cfg.n_head, 8.0)), // max_bias=8.0 is a common default
            ),
            PositionalEncoding::Learned
            | PositionalEncoding::Sinusoidal
            | PositionalEncoding::None => (None, None),
        };

        Ok(Self {
            c_attn,
            c_proj,
            softmax_scale,
            n_head: cfg.n_head,
            n_embd: cfg.n_embd,
            h_dim,
            rope,
            alibi,
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
    type Output = (Tensor, Tensor);

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Self::Output> {
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

        let mut k = k.view(qkv_shape.clone())?.permute(&[0, 2, 1, 3])?;
        let mut q = q.view(qkv_shape.clone())?.permute(&[0, 2, 1, 3])?;
        let v = v.view(qkv_shape.clone())?.permute(&[0, 2, 1, 3])?;

        let cache_entry = if cache.borrow_mut().use_kv_cache() {
            let cache_ref = cache.borrow_mut();
            let entry = cache_ref[block_idx].clone();
            Some(entry)
        } else {
            None
        };
        let offset = cache_entry.as_ref().map(|kv| kv.entries).unwrap_or(0);
        // Apply RoPE if enabled
        if let Some(rope) = &self.rope {
            q = rope.schedule(RotaryInput { input: q, offset })?;
            k = rope.schedule(RotaryInput { input: k, offset })?;
        }

        let (k, v) = if let Some(cache_entry) = cache_entry {
            let k_cache = cache_entry.k_cache.cache(k, 2, offset)?;
            let v_cache = cache_entry.v_cache.cache(v, 2, offset)?;
            (k_cache, v_cache)
        } else {
            (k, v)
        };

        let mut att = q
            .matmul(k.clone(), false, true)?
            .mul(self.softmax_scale.clone())?;

        // Apply ALiBi if enabled
        if let Some(alibi) = &self.alibi {
            att = alibi.schedule(AlibiInput { input: att })?;
        }

        let att = if q_len <= 1 {
            att
        } else {
            let mask = cache
                .borrow_mut()
                .mask(q_len)?
                .broadcast_to(att.shape().clone())?;
            masked_fill(&att, &mask, -1e9)?
        };

        let att = att.softmax(3)?;
        let y = att
            .clone()
            .matmul(v, false, false)?
            .permute(&[0, 2, 1, 3])?
            .view(shape![batch_size as _, q_len, self.n_embd])?;

        let y = self.c_proj.schedule(y)?;

        Ok((y, att))
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> anyhow::Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::full(shape, on_true, on_false.device());
    let m = mask.clone().where_cond(on_true, on_false.clone())?;
    Ok(m)
}
