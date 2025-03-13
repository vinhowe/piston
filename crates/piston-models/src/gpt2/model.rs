/// GPT2 model, state-dict-compatible with minGPT, except for "bias" buffer, which is easy enough
/// to add back in:
///
/// ```python
/// from safetensors.torch import load_file
/// n_layer = 6
/// state_dict = load_file("our-gpt2.safetensors")
/// bias_buffer = torch.tril(torch.ones(24, 24)).view(1, 1, 24, 24)
/// state_dict["lm_head.bias"] = bias_buffer
/// for i in range(n_layer):
///     state_dict[f"transformer.h.{i}.attn.bias"] = bias_buffer
/// ```
///
/// You can then load the state_dict into minGPT with:
///
/// ```python
/// from mingpt.model import GPT
/// config = GPT.get_default_config()
/// # ... set config appropriately ...
/// model = GPT(config)
/// model.load_state_dict(state_dict)
/// ```
use std::{cell::RefCell, rc::Rc};

use super::{
    attn::{GPT2AttnInput, GPT2SelfAttention},
    linear::linear_no_bias_gpt2,
    mlp::MLP,
};
use maybe_async::maybe_async;
use piston::{shape, Device, Tensor};
use piston_macros::scoped_module;
use piston_nn::{
    embedding, layer_norm, Dropout, Embedding, KVCache, LayerNorm, Linear, Module,
    SinusoidalEmbedding, SinusoidalInput, VarBuilder,
};

#[derive(Debug, Clone, PartialEq)]
pub enum PositionalEncoding {
    Learned,
    RoPE,
    ALiBi,
    Sinusoidal,
    None,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LayerNormPosition {
    Pre,
    Post,
    None,
}

// https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/config.json
#[derive(Debug, Clone)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_act: piston_nn::Activation,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub block_size: usize,
    pub attention_only: bool,
    pub positional_encoding: PositionalEncoding,
    pub layernorm_position: LayerNormPosition,
    pub embd_pdrop: f32,
    pub attn_pdrop: f32,
    pub resid_pdrop: f32,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }
}

#[derive(Debug)]
pub struct DecoderLayer {
    ln_1: LayerNorm,
    attn: GPT2SelfAttention,
    ln_2: LayerNorm,
    mlp: MLP,
    attention_only: bool,
    layernorm_position: LayerNormPosition,
    dropout: Option<Dropout>,
}

impl DecoderLayer {
    #[maybe_async]
    async fn new(cfg: &Config, vb: VarBuilder<'_>) -> anyhow::Result<Self> {
        let attn = GPT2SelfAttention::new(cfg, vb.pp("attn")).await?;

        let ln_1 = layer_norm(cfg.n_embd, Default::default(), vb.pp("ln_1")).await?;
        let ln_2 = layer_norm(cfg.n_embd, Default::default(), vb.pp("ln_2")).await?;

        let mlp = MLP::new(cfg, vb.pp("mlp")).await?;
        let dropout = if cfg.resid_pdrop > 0.0 {
            Some(Dropout::new(cfg.resid_pdrop))
        } else {
            None
        };

        Ok(Self {
            attn,
            mlp,
            ln_1,
            ln_2,
            attention_only: cfg.attention_only,
            layernorm_position: cfg.layernorm_position.clone(),
            dropout,
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

#[scoped_module]
impl Module for DecoderLayer {
    type Input = DecoderLayerInput;
    type Output = (Tensor, Tensor);

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Self::Output> {
        let DecoderLayerInput {
            x,
            index_pos,
            block_idx,
            cache,
        } = input;

        let residual = x.clone();
        let x = match self.layernorm_position {
            LayerNormPosition::Pre => self.ln_1.schedule(x)?,
            LayerNormPosition::Post | LayerNormPosition::None => x,
        };
        let (attn_output, attn_masks) = self.attn.schedule(GPT2AttnInput {
            input: x,
            index_pos,
            block_idx,
            cache,
        })?;
        let x = residual.add(attn_output)?;
        let x = match self.layernorm_position {
            LayerNormPosition::Post => self.ln_1.schedule(x)?,
            LayerNormPosition::Pre | LayerNormPosition::None => x,
        };

        // Skip the feed-forward network if attention_only is true
        if !self.attention_only {
            let residual = x.clone();
            let x = match self.layernorm_position {
                LayerNormPosition::Pre => self.ln_2.schedule(x)?,
                LayerNormPosition::Post | LayerNormPosition::None => x,
            };
            let x = self.mlp.schedule(x)?;
            let x = match &self.dropout {
                Some(dropout) => dropout.schedule(x)?,
                None => x,
            };
            let x = residual.add(x)?;
            let x = match self.layernorm_position {
                LayerNormPosition::Post => self.ln_2.schedule(x)?,
                LayerNormPosition::Pre | LayerNormPosition::None => x,
            };
            Ok((x, attn_masks))
        } else {
            Ok((x, attn_masks))
        }
    }
}

#[derive(Debug)]
pub struct GPT2 {
    pub wte: Embedding,
    pub wpe: Option<Embedding>,
    pub sinusoidal: Option<SinusoidalEmbedding>,
    pub layers: Vec<DecoderLayer>,
    pub ln_f: LayerNorm,
    pub lm_head: Linear,
    pub embd_dropout: Option<Dropout>,
    pub kv_cache: Rc<RefCell<KVCache>>,
    pub device: Device,
}

pub struct GPT2Input {
    pub x: Tensor,
    pub index_pos: usize,
}

#[scoped_module]
impl Module for GPT2 {
    type Input = GPT2Input;
    type Output = (Tensor, Tensor);

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Self::Output> {
        let GPT2Input { x, index_pos } = input;
        let [b_size, seq_len]: [usize; 2] = x.shape().try_into()?;

        let mut x = self.wte.schedule(x)?;

        // Add positional embeddings based on the type
        if let Some(wpe) = &self.wpe {
            // Learned embeddings
            let pos = Tensor::arange(0, seq_len as i32, x.device())?;
            let pos = pos.unsqueeze(0)?.broadcast_to(shape![b_size, seq_len])?;
            let position_embeds = wpe.schedule(pos)?;
            x = x.add(position_embeds)?;
        } else if let Some(sinusoidal) = &self.sinusoidal {
            // Sinusoidal embeddings
            x = sinusoidal.schedule(SinusoidalInput {
                input: x,
                offset: index_pos,
            })?;
        }
        // For RoPE and ALiBi, positional encoding is handled in the attention layer

        let mut x = match &self.embd_dropout {
            Some(dropout) => dropout.schedule(x)?,
            None => x,
        };

        let mut attn_masks = vec![];

        for (block_idx, layer) in self.layers.iter().enumerate() {
            let input = DecoderLayerInput {
                x,
                block_idx,
                index_pos,
                cache: self.kv_cache.clone(),
            };
            let (layer_output, layer_attn_masks) = layer.schedule(input)?;
            x = layer_output;
            attn_masks.push(layer_attn_masks.flatten_from(2)?);
        }

        let attn_masks = Tensor::stack(attn_masks.into(), 0)?;
        x = self.ln_f.schedule(x)?;
        let logits = self.lm_head.schedule(x)?;
        Ok((logits, attn_masks))
    }
}

#[maybe_async]
impl GPT2 {
    const MAX_CACHE: usize = 4096;

    pub async fn new(cfg: &Config, vb: VarBuilder<'_>, use_kv_cache: bool) -> anyhow::Result<Self> {
        let vb_m = vb.pp("transformer");
        let wte = embedding(cfg.vocab_size, cfg.n_embd, vb_m.pp("wte")).await?;

        // Initialize positional encoding based on the type
        let (wpe, sinusoidal) = match cfg.positional_encoding {
            PositionalEncoding::Learned => (
                Some(embedding(cfg.block_size, cfg.n_embd, vb_m.pp("wpe")).await?),
                None,
            ),
            PositionalEncoding::Sinusoidal => (
                None,
                Some(SinusoidalEmbedding::new(cfg.n_embd, vb.device())?),
            ),
            PositionalEncoding::RoPE | PositionalEncoding::ALiBi => (None, None),
            PositionalEncoding::None => (None, None),
        };

        let n_layers = cfg.n_layer as _;

        let mut layers = Vec::with_capacity(n_layers);
        let vb_l = vb_m.pp("h");
        for layer_idx in 0..n_layers {
            let layer = DecoderLayer::new(cfg, vb_l.pp(layer_idx)).await?;
            layers.push(layer)
        }

        let ln_f = layer_norm(cfg.n_embd, Default::default(), vb_m.pp("ln_f")).await?;
        let lm_head = linear_no_bias_gpt2(cfg.n_embd, cfg.vocab_size, vb.pp("lm_head")).await?;
        let embd_dropout = if cfg.embd_pdrop > 0.0 {
            Some(Dropout::new(cfg.embd_pdrop))
        } else {
            None
        };

        let cache_shape = shape![1, cfg.n_head as _, Self::MAX_CACHE, cfg.head_dim() as _];

        Ok(Self {
            wte,
            wpe,
            sinusoidal,
            layers,
            ln_f,
            lm_head,
            embd_dropout,
            kv_cache: Rc::new(RefCell::new(KVCache::new::<f32>(
                n_layers as _,
                use_kv_cache,
                cache_shape,
                vb.device(),
            )?)),
            device: vb.device().clone(),
        })
    }

    pub fn reset(&mut self) {
        self.kv_cache.borrow_mut().reset();
    }

    pub fn cache_mut(&mut self) -> std::cell::RefMut<'_, KVCache> {
        (*self.kv_cache).borrow_mut()
    }
}

#[cfg(all(test, not(target_arch = "wasm32"), feature = "pyo3"))]
mod tests {
    use piston::{prelude::shape, DType, Device, DeviceRequest, Parameter, StepLogConfig, Tensor};
    use piston_datasets::{
        nlp::{
            tinystories::{Dataset, DatasetRandomIter},
            toy::{ToyTaskIter, TwoSumTask},
        },
        Batcher,
    };
    use piston_nn::{
        clip_grad_norm, cross_entropy, AdamW, ConstantLR, LRScheduler, Module, Optimizer,
        ParamsAdamW, VarBuilder, VarMap,
    };

    use super::GPT2;
    use crate::gpt2::{
        generate,
        model::{Config, GPT2Input, PositionalEncoding},
        LayerNormPosition,
    };

    #[test]
    #[cfg_attr(feature = "ci", ignore)]
    fn train_zeros() -> anyhow::Result<()> {
        let _ = env_logger::builder().is_test(true).try_init();

        let device = Device::request_device(DeviceRequest::GPU).unwrap();

        const VOCAB_SIZE: usize = 10;

        let config = Config {
            vocab_size: VOCAB_SIZE,
            hidden_act: piston_nn::Activation::Relu2,
            n_embd: 128,
            n_layer: 4,
            n_head: 4,
            block_size: 64,
            attention_only: false,
            positional_encoding: PositionalEncoding::Learned,
            layernorm_position: LayerNormPosition::Pre,
            embd_pdrop: 0.1,
            attn_pdrop: 0.1,
            resid_pdrop: 0.1,
        };

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, piston::DType::F32, &device.clone());

        let model = GPT2::new(&config, vb, false)?;

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
                .collect::<Vec<(Option<String>, Parameter)>>(),
            params,
        )?;

        const BATCH_SIZE: usize = 1;

        for step in 0..100 {
            let input = Tensor::zeros::<i32>(&shape![BATCH_SIZE, config.block_size], &device)?;
            let tgt = Tensor::zeros::<i32>(&shape![BATCH_SIZE, config.block_size], &device)?;

            let (logits, _) = model.schedule(GPT2Input {
                x: input,
                index_pos: 0,
            })?;

            let loss = cross_entropy(logits.flatten_to(1)?, tgt.flatten_to(1)?)?;

            let grads = loss.backward()?;

            // clip_grad_norm(&mut grads, 1.0f32, &device)?;

            opt.step(&grads, &device)?;

            let loss_vec = loss.clone().to(&Device::CPU)?.to_vec::<f32>()?;

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
            hidden_act: piston_nn::Activation::Relu2,
            n_embd: 128,
            n_layer: 1,
            n_head: 1,
            block_size: 64,
            attention_only: false,
            positional_encoding: PositionalEncoding::Learned,
            layernorm_position: LayerNormPosition::Pre,
            embd_pdrop: 0.1,
            attn_pdrop: 0.1,
            resid_pdrop: 0.1,
        };

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, piston::DType::F32, &device.clone());

        let model = GPT2::new(&config, vb, false)?;

        const BATCH_SIZE: usize = 10;

        for batch_index in 0..10 {
            let input = Tensor::zeros::<i32>(&shape![BATCH_SIZE, config.block_size], &device)?;
            let tgt = Tensor::zeros::<i32>(&shape![BATCH_SIZE, config.block_size], &device)?;

            let (logits, _) = model.schedule(GPT2Input {
                x: input,
                index_pos: 0,
            })?;

            let loss = cross_entropy(logits.flatten_to(1)?, tgt.flatten_to(1)?)?;

            device.try_gpu()?.mark_step()?;

            let loss_vec = loss.clone().to(&Device::CPU)?.to_vec::<f32>()?;

            println!("{:?} loss: {:?}", batch_index, loss_vec[0]);
        }

        Ok(())
    }

    #[test]
    #[cfg_attr(feature = "ci", ignore)]
    fn train_2_sum() -> anyhow::Result<()> {
        let _ = env_logger::builder().is_test(true).try_init();

        const VOCAB_SIZE: usize = 256;

        const BATCH_SIZE: usize = 8;
        const SEQUENCE_LENGTH: usize = 24;

        let config = Config {
            vocab_size: VOCAB_SIZE,
            hidden_act: piston_nn::Activation::Relu2,
            n_embd: 768,
            n_layer: 4,
            n_head: 4,
            block_size: SEQUENCE_LENGTH,
            attention_only: false,
            positional_encoding: PositionalEncoding::Learned,
            layernorm_position: LayerNormPosition::Pre,
            embd_pdrop: 0.1,
            attn_pdrop: 0.1,
            resid_pdrop: 0.1,
        };

        let device = Device::request_device(piston::DeviceRequest::GPU).unwrap();

        // If you want to enable profiling:
        // device
        //     .try_gpu()
        //     .unwrap()
        //     .set_step_log_config(StepLogConfig {
        //         profiling: true,
        //         ..Default::default()
        //     });

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device.clone());

        let mut model = GPT2::new(&config, vb, false).unwrap();

        let params = ParamsAdamW {
            lr: 1e-4,
            // lr: 0.0,
            // weight_decay: 0.1,
            ..Default::default()
        };

        let opt = AdamW::new(
            varmap
                .all_labeled_vars()
                .iter()
                .map(|(label, var)| (Some(label.to_owned()), var.to_owned()))
                .collect::<Vec<(Option<String>, Parameter)>>(),
            params,
        )?;

        let mut lr_scheduler = ConstantLR::new(opt, 1.0, 100);

        let task = TwoSumTask::new(5, 5, Some(10));
        let dataset_iter = ToyTaskIter::new(task, device.clone());
        let batch_iter = Batcher::new_r2(dataset_iter).batch_size(BATCH_SIZE);

        for (batch_index, batch) in batch_iter.enumerate() {
            if batch_index > 100 {
                break;
            }

            let (input, tgt) = batch?;

            let (logits, _) = model.schedule(GPT2Input {
                x: input,
                index_pos: 0,
            })?;

            let loss = cross_entropy(logits.flatten_to(1)?, tgt.flatten_to(1)?)?;

            let grads = loss.backward()?;

            lr_scheduler.step(&grads, &device)?;

            let loss_vec = loss.clone().to(&Device::CPU)?.to_vec::<f32>()?;

            println!("{:?} loss: {:?}, norm: {:?}", batch_index, loss_vec[0], "?");
        }

        model.reset();

        // Uncomment this to generate a sequence from the model:
        // generate(
        //     &mut model,
        //     "12,35,07,99,03:134=".chars().map(|c| c as i32).collect(),
        //     |s, _logits| {
        //         println!("{}", s.iter().map(|&c| c as u8 as char).collect::<String>());
        //         // println!("{:?}", logits);
        //     },
        //     24,
        // )
        // .unwrap();

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
        let vb = VarBuilder::from_varmap(&varmap, piston::DType::F32, &device.clone());

        // distilgpt2-sized
        let config = Config {
            vocab_size: VOCAB_SIZE,
            hidden_act: piston_nn::Activation::Relu2,
            n_embd: 768,
            n_layer: 12,
            n_head: 12,
            block_size: 256,
            attention_only: false,
            positional_encoding: PositionalEncoding::Learned,
            layernorm_position: LayerNormPosition::Pre,
            embd_pdrop: 0.1,
            attn_pdrop: 0.1,
            resid_pdrop: 0.1,
        };

        let iter = DatasetRandomIter::new(&dataset, false, config.block_size, device.clone());
        let batch_iter = Batcher::new_r2(iter).batch_size(BATCH_SIZE);

        let model = GPT2::new(&config, vb, false)?;

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
            let (logits, _) = model.schedule(GPT2Input {
                x: input.clone(),
                index_pos: 0,
            })?;

            let loss = cross_entropy(logits.flatten_to(1)?, tgt.flatten_to(1)?)?;

            // This is something of a hack; we add references to all tensors that need to be backpropped
            let grads = loss.backward()?;

            opt.step(&grads, &device)?;

            let loss_vec = loss.to(&Device::CPU)?.to_vec::<f32>()?;

            println!("{:?} loss: {:?}", batch_index, loss_vec[0]);
        }

        Ok(())
    }

    #[test]
    fn generate_from_initialization() -> anyhow::Result<()> {
        let _ = env_logger::builder().is_test(true).try_init();

        let device = Device::request_device(DeviceRequest::GPU).unwrap();

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, piston::DType::F32, &device.clone());

        let config = Config {
            vocab_size: 256,
            hidden_act: piston_nn::Activation::Relu2,
            n_embd: 128,
            n_layer: 1,
            n_head: 1,
            block_size: 20,
            attention_only: false,
            positional_encoding: PositionalEncoding::Learned,
            layernorm_position: LayerNormPosition::Pre,
            embd_pdrop: 0.1,
            attn_pdrop: 0.1,
            resid_pdrop: 0.1,
        };

        let mut model = GPT2::new(&config, vb, false)?;

        // Uncomment this to generate a uninitialized sequence from the model:
        // generate(
        //     &mut model,
        //     "Hello, world".chars().map(|c| c as i32).collect(),
        //     |s, logits| {
        //         println!("{:?}", s);
        //         println!("{:?}", logits);
        //     },
        //     24,
        // )
        // .unwrap();

        Ok(())
    }
}
