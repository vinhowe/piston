/// This is not a true GPT2 model, in that we probably couldn't load the weights from a GGML file.
use std::{cell::RefCell, rc::Rc};

use maybe_async::maybe_async;
use ratchet::{shape, Device, Tensor};
use ratchet_nn::{
    embedding, layer_norm, Embedding, KVCache, LayerNorm, Linear, Module, VarBuilder,
};

use super::{
    attn::{GPT2AttnInput, GPT2SelfAttention},
    linear::linear_no_bias_gpt2,
    mlp::MLP,
};

// https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/config.json
#[derive(Debug, Clone)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_act: ratchet_nn::Activation,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub block_size: usize,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }
}

#[derive(Debug)]
pub struct DecoderLayer {
    input_norm: LayerNorm,
    self_attn: GPT2SelfAttention,
    ffn_norm: LayerNorm,
    mlp: MLP,
}

impl DecoderLayer {
    #[maybe_async]
    async fn new(cfg: &Config, vb: VarBuilder<'_>) -> anyhow::Result<Self> {
        let self_attn = GPT2SelfAttention::new(cfg, vb.pp("self_attn")).await?;

        let input_norm =
            layer_norm(cfg.n_embd, Default::default(), vb.pp("input_layernorm")).await?;
        let ffn_norm = layer_norm(
            cfg.n_embd,
            Default::default(),
            vb.pp("post_attention_layernorm"),
        )
        .await?;

        let mlp = MLP::new(cfg, vb.pp("mlp")).await?;

        Ok(Self {
            self_attn,
            mlp,
            input_norm,
            ffn_norm,
        })
    }
}

pub struct DecoderLayerInput {
    pub x: Tensor,
    pub index_pos: usize,
    pub block_idx: usize,
    // pub mask: Option<Tensor>,
    pub cache: Rc<RefCell<KVCache>>,
}

impl Module for DecoderLayer {
    type Input = DecoderLayerInput;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let DecoderLayerInput {
            x,
            index_pos,
            block_idx,
            cache,
        } = input;
        let residual = x.clone();
        let xs = self.input_norm.schedule(x)?;
        let attn_output = self.self_attn.schedule(GPT2AttnInput {
            input: xs.clone(),
            index_pos,
            block_idx,
            cache,
        })?;
        let xs = residual.add(attn_output)?;
        let residual = xs.clone();
        let xs = self.ffn_norm.schedule(xs)?;
        let xs = self.mlp.schedule(xs)?;
        let xs = residual.add(xs)?;
        Ok(xs)
    }
}

#[derive(Debug)]
pub struct GPT2 {
    pub wte: Embedding,
    pub wpe: Embedding,
    pub layers: Vec<DecoderLayer>,
    pub ln_post: LayerNorm,
    pub lm_head: Linear,
    pub kv_cache: Rc<RefCell<KVCache>>,
    pub device: Device,
}

pub struct GPT2Input {
    pub x: Tensor,
    pub index_pos: usize,
}

impl Module for GPT2 {
    type Input = GPT2Input;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let GPT2Input { x, index_pos } = input;
        let [b_size, seq_len]: [usize; 2] = x.shape().try_into()?;

        let pos = Tensor::arange(0, seq_len as i32, x.device())?;
        let pos = pos.unsqueeze(0)?.broadcast_to(shape![b_size, seq_len])?;
        let input_embeds = self.wte.schedule(x)?;
        let position_embeds = self.wpe.schedule(pos)?;

        let mut x = input_embeds.add(position_embeds)?;

        for (block_idx, layer) in self.layers.iter().enumerate() {
            let input = DecoderLayerInput {
                x,
                block_idx,
                index_pos,
                cache: self.kv_cache.clone(),
            };
            x = layer.schedule(input)?;
        }
        x = self.ln_post.schedule(x)?;
        let logits = self.lm_head.schedule(x)?;
        Ok(logits)
    }
}

#[maybe_async]
impl GPT2 {
    const MAX_CACHE: usize = 4096;

    pub async fn new(cfg: &Config, vb: VarBuilder<'_>) -> anyhow::Result<Self> {
        let vb_m = vb.pp("model");
        let wte = embedding(cfg.vocab_size, cfg.n_embd, vb_m.pp("wte")).await?;
        let wpe = embedding(cfg.block_size, cfg.n_embd, vb_m.pp("wpe")).await?;

        let n_layers = cfg.n_layer as _;

        let mut layers = Vec::with_capacity(n_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..n_layers {
            // let layer = DecoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(layer_idx))?;
            let layer = DecoderLayer::new(cfg, vb_l.pp(layer_idx)).await?;
            layers.push(layer)
        }

        let ln_post = layer_norm(cfg.n_embd, Default::default(), vb_m.pp("norm")).await?;
        let lm_head = linear_no_bias_gpt2(cfg.n_embd, cfg.vocab_size, vb.pp("lm_head")).await?;

        let cache_shape = shape![1, n_layers as _, Self::MAX_CACHE, cfg.head_dim() as _];

        Ok(Self {
            wte,
            wpe,
            layers,
            ln_post,
            lm_head,
            kv_cache: Rc::new(RefCell::new(KVCache::new::<f32>(
                n_layers as _,
                false,
                cache_shape,
                vb.device(),
            ))),
            device: vb.device().clone(),
        })
    }
}

#[cfg(all(test, not(target_arch = "wasm32"), feature = "pyo3"))]
mod tests {
    use ratchet::{prelude::shape, DType, Device, DeviceRequest, Tensor, Var};
    use ratchet_datasets::{
        nlp::{
            tinystories::{Dataset, DatasetRandomIter},
            toy::{ToyTaskIter, TwoSumTask},
        },
        Batcher,
    };
    use ratchet_nn::{
        clip_grad_norm, cross_entropy, AdamW, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap,
    };

    use super::GPT2;
    use crate::gpt2::model::{Config, GPT2Input};

    #[test]
    #[cfg_attr(feature = "ci", ignore)]
    fn train_zeros() -> anyhow::Result<()> {
        let _ = env_logger::builder().is_test(true).try_init();

        let device = Device::request_device(DeviceRequest::GPU).unwrap();

        const VOCAB_SIZE: usize = 10;

        let config = Config {
            vocab_size: VOCAB_SIZE,
            hidden_act: ratchet_nn::Activation::Gelu,
            n_embd: 128,
            n_layer: 4,
            n_head: 4,
            block_size: 64,
        };

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, ratchet::DType::F32, &device.clone());

        let model = GPT2::new(&config, vb)?;

        let params = ParamsAdamW {
            lr: 1e-4,
            // lr: 0.0,
            // weight_decay: 0.1,
            ..Default::default()
        };

        let mut opt = AdamW::new(
            varmap
                .all_labeled_vars()
                .iter()
                .map(|(label, var)| (Some(label.to_owned()), var.to_owned()))
                .collect::<Vec<(Option<String>, Var)>>(),
            params,
        )?;

        const BATCH_SIZE: usize = 1;

        for step in 0..100 {
            let input = Tensor::zeros::<i32>(&shape![BATCH_SIZE, config.block_size], &device);
            let tgt = Tensor::zeros::<i32>(&shape![BATCH_SIZE, config.block_size], &device);

            let logits = model.schedule(GPT2Input {
                x: input,
                index_pos: 0,
            })?;

            let loss = cross_entropy(logits.flatten_to(1)?, tgt.flatten_to(1)?)?;

            let grads = loss.backward()?;

            // clip_grad_norm(&mut grads, 1.0f32, &device)?;

            opt.step(&grads)?;

            let loss_vec = loss.clone().resolve()?.to(&Device::CPU)?.to_vec::<f32>()?;

            println!("{:?} loss: {:?}", step, loss_vec[0]);
        }

        Ok(())
    }

    #[test]
    #[cfg_attr(feature = "ci", ignore)]
    fn n_forward_passes() -> anyhow::Result<()> {
        let _ = env_logger::builder().is_test(true).try_init();

        let device = Device::request_device(DeviceRequest::GPU).unwrap();

        const VOCAB_SIZE: usize = 10;

        let config = Config {
            vocab_size: VOCAB_SIZE,
            hidden_act: ratchet_nn::Activation::Gelu,
            n_embd: 128,
            n_layer: 1,
            n_head: 1,
            block_size: 64,
        };

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, ratchet::DType::F32, &device.clone());

        let model = GPT2::new(&config, vb)?;

        const BATCH_SIZE: usize = 1;

        for batch_index in 0..10 {
            let input = Tensor::zeros::<i32>(&shape![BATCH_SIZE, config.block_size], &device);
            let tgt = Tensor::zeros::<i32>(&shape![BATCH_SIZE, config.block_size], &device);

            let logits = model.schedule(GPT2Input {
                x: input,
                index_pos: 0,
            })?;

            let loss = cross_entropy(logits.flatten_to(1)?, tgt.flatten_to(1)?)?;

            let loss_vec = loss.clone().resolve()?.to(&Device::CPU)?.to_vec::<f32>()?;

            println!("{:?} loss: {:?}", batch_index, loss_vec[0]);
        }

        Ok(())
    }

    #[test]
    #[cfg_attr(feature = "ci", ignore)]
    fn train_2_sum() -> anyhow::Result<()> {
        let _ = env_logger::builder().is_test(true).try_init();

        const VOCAB_SIZE: usize = 256;

        const BATCH_SIZE: usize = 4;
        const SEQUENCE_LENGTH: usize = 24;

        let config = Config {
            vocab_size: VOCAB_SIZE,
            hidden_act: ratchet_nn::Activation::Gelu,
            n_embd: 768,
            n_layer: 4,
            n_head: 4,
            block_size: SEQUENCE_LENGTH,
        };

        let device = Device::request_device(ratchet::DeviceRequest::GPU).unwrap();

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device.clone());

        let model = GPT2::new(&config, vb).unwrap();

        let params = ParamsAdamW {
            lr: 1e-4,
            // lr: 0.0,
            // weight_decay: 0.1,
            ..Default::default()
        };

        let mut opt = AdamW::new(
            varmap
                .all_labeled_vars()
                .iter()
                .map(|(label, var)| (Some(label.to_owned()), var.to_owned()))
                .collect::<Vec<(Option<String>, Var)>>(),
            params,
        )
        .unwrap();

        let task = TwoSumTask::new(5, 5, Some(10));
        let dataset_iter = ToyTaskIter::new(task, device.clone());
        let batch_iter = Batcher::new_r2(dataset_iter).batch_size(BATCH_SIZE);

        for (batch_index, batch) in batch_iter.enumerate() {
            if batch_index > 10 {
                break;
            }

            let (input, tgt) = batch?;

            let logits = model.schedule(GPT2Input {
                x: input,
                index_pos: 0,
            })?;

            let loss = cross_entropy(logits.flatten_to(1)?, tgt.flatten_to(1)?)?;

            let grads = loss.backward()?;

            opt.step(&grads)?;

            let loss_vec = loss.clone().resolve()?.to(&Device::CPU)?.to_vec::<f32>()?;

            println!("{:?} loss: {:?}, norm: {:?}", batch_index, loss_vec[0], "?");
        }
        Ok(())
    }

    #[test]
    #[cfg_attr(feature = "ci", ignore)]
    fn train_tinystories() -> anyhow::Result<()> {
        let _ = env_logger::builder().is_test(true).try_init();

        let device = Device::request_device(DeviceRequest::GPU).unwrap();

        // Print current dir:
        println!("Current dir: {:?}", std::env::current_dir().unwrap());

        let dataset = Dataset::new("../../../llama2.c/data/TinyStories_all_data/")?;
        println!(
            "loaded dataset, train: {} files, valid: {} files",
            dataset.train_tokens(),
            dataset.valid_tokens()
        );

        const VOCAB_SIZE: usize = 32000;
        const BATCH_SIZE: usize = 1;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, ratchet::DType::F32, &device.clone());

        // distilgpt2-sized
        let config = Config {
            vocab_size: VOCAB_SIZE,
            hidden_act: ratchet_nn::Activation::Gelu,
            n_embd: 768,
            n_layer: 12,
            n_head: 12,
            block_size: 256,
        };

        let iter = DatasetRandomIter::new(&dataset, false, config.block_size, device.clone());
        let batch_iter = Batcher::new_r2(iter).batch_size(BATCH_SIZE);

        let model = GPT2::new(&config, vb)?;

        let params = ParamsAdamW {
            lr: 0.0001,
            // lr: 0.0,
            // weight_decay: 0.1,
            ..Default::default()
        };

        let mut opt = AdamW::new(
            varmap
                .all_labeled_vars()
                .iter()
                .map(|(label, var)| (Some(label.to_owned()), var.to_owned()))
                .collect::<Vec<(Option<String>, Var)>>(),
            params,
        )?;

        for (batch_index, batch) in batch_iter.enumerate() {
            let (input, tgt) = batch?;
            let logits = model.schedule(GPT2Input {
                x: input.clone(),
                index_pos: 0,
            })?;

            let loss = cross_entropy(logits.flatten_to(1)?, tgt.flatten_to(1)?)?;

            // This is something of a hack; we add references to all tensors that need to be backpropped
            let grads = loss.backward()?;

            opt.step(&grads)?;

            let loss_vec = loss.resolve_deferred()?.to(&Device::CPU)?.to_vec::<f32>()?;

            println!("{:?} loss: {:?}", batch_index, loss_vec[0]);
        }

        Ok(())
    }
}
