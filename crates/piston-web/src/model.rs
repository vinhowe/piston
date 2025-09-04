use crate::error::IntoJsError;
use anyhow::Result;
use async_trait::async_trait;
use js_sys::Function;
use piston::{
    AllDims, DType, Device, DeviceRequest, StepLog, Tensor, TensorOptions, reset_scope_context,
};
use piston_models::gpt2::generate;
use piston_models::gpt2::{Config, GPT2Input, PositionalEncoding};
use piston_models::gpt2::{GPT2, LayerNormPosition};
use piston_nn::{
    Activation, AdamW, ConstantLR, CosineAnnealingLR, LRScheduler, LRSchedulerCore, LinearLR,
    Module, Optimizer, ParamsAdamW, SGD, VarBuilder, VarMap, label_smoothed_nll, log_softmax,
    nll_masked,
};
use piston_nn::{ModuleMode, ModuleModeGuard};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::iter::Iterator;
use wasm_bindgen::prelude::*;

#[derive(Serialize, Deserialize)]
pub struct OptimizerConfig {
    pub optimizer_type: String, // "adamw" or "sgd"
    pub lr: f64,
    #[serde(default = "default_beta1")]
    pub beta1: f64,
    #[serde(default = "default_beta2")]
    pub beta2: f64,
    #[serde(default = "default_eps")]
    pub eps: f64,
    #[serde(default = "default_weight_decay")]
    pub weight_decay: f64,
    #[serde(default = "default_momentum")]
    pub momentum: f64, // For SGD
    #[serde(default = "default_scheduler_type")]
    pub scheduler_type: String, // "none", "constant", "linear", "cosine"
    #[serde(default = "default_scheduler_factor")]
    pub scheduler_factor: f64, // For constant/linear scheduler
    #[serde(default = "default_scheduler_steps")]
    pub scheduler_steps: usize, // For all schedulers
    #[serde(default = "default_scheduler_eta_min")]
    pub scheduler_eta_min: f64, // For cosine scheduler
}

fn default_beta1() -> f64 {
    0.9
}
fn default_beta2() -> f64 {
    0.999
}
fn default_eps() -> f64 {
    1e-8
}
fn default_weight_decay() -> f64 {
    0.0
}
fn default_momentum() -> f64 {
    0.0
}
fn default_scheduler_type() -> String {
    "none".to_string()
}
fn default_scheduler_factor() -> f64 {
    1.0
}
fn default_scheduler_steps() -> usize {
    1000
}
fn default_scheduler_eta_min() -> f64 {
    0.0
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
pub enum JsDebugSelection {
    String(String),
    Array(Vec<String>),
}

#[derive(Serialize, Deserialize)]
pub struct JsStepLogConfig {
    #[serde(default = "default_profiling")]
    pub profiling: bool,
    #[serde(default = "default_debug_selection")]
    pub debug_selection: Option<JsDebugSelection>,
}

fn default_profiling() -> bool {
    false
}

fn default_debug_selection() -> Option<JsDebugSelection> {
    None
}

#[derive(Serialize, Deserialize)]
pub struct TrainerConfig {
    pub vocab_size: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub block_size: usize,
    pub batch_size: usize,
    #[serde(default = "default_caching_enabled")]
    pub caching_enabled: bool,
    #[serde(default = "default_inplace_support")]
    pub inplace_support: bool,
    #[serde(default = "default_optimizer_config")]
    pub optimizer: OptimizerConfig,
    #[serde(default = "default_activation")]
    pub activation: String,
    #[serde(default = "default_attention_only")]
    pub attention_only: bool,
    #[serde(default = "default_positional_encoding")]
    pub positional_encoding: String,
    #[serde(default = "default_layernorm_position")]
    pub layernorm_position: String,
    #[serde(default = "default_seed")]
    pub seed: Option<u64>,
    #[serde(default = "default_label_smoothing")]
    pub label_smoothing: f32,
    #[serde(default = "default_embd_pdrop")]
    pub embd_pdrop: f32,
    #[serde(default = "default_attn_pdrop")]
    pub attn_pdrop: f32,
    #[serde(default = "default_resid_pdrop")]
    pub resid_pdrop: f32,
}

fn default_caching_enabled() -> bool {
    true
}

fn default_inplace_support() -> bool {
    true
}

fn default_optimizer_config() -> OptimizerConfig {
    OptimizerConfig {
        optimizer_type: "adamw".to_string(),
        lr: 1e-3,
        beta1: default_beta1(),
        beta2: default_beta2(),
        eps: default_eps(),
        weight_decay: default_weight_decay(),
        momentum: default_momentum(),
        scheduler_type: default_scheduler_type(),
        scheduler_factor: default_scheduler_factor(),
        scheduler_steps: default_scheduler_steps(),
        scheduler_eta_min: default_scheduler_eta_min(),
    }
}

fn default_activation() -> String {
    "gelu".to_string()
}

fn default_attention_only() -> bool {
    false
}

fn default_positional_encoding() -> String {
    "learned".to_string()
}

fn default_layernorm_position() -> String {
    "pre".to_string()
}

fn default_embd_pdrop() -> f32 {
    0.1
}

fn default_attn_pdrop() -> f32 {
    0.1
}

fn default_resid_pdrop() -> f32 {
    0.1
}

fn default_seed() -> Option<u64> {
    None
}

fn default_label_smoothing() -> f32 {
    0.0
}

fn string_to_activation(s: &str) -> Activation {
    match s.to_lowercase().as_str() {
        "gelu" => Activation::Gelu,
        "relu" => Activation::Relu,
        "relu2" => Activation::Relu2,
        "silu" => Activation::Silu,
        "sigmoid" => Activation::Sigmoid,
        "swiglu" => Activation::Swiglu,
        _ => Activation::Gelu,
    }
}

#[derive(Debug, Clone)]
pub enum OptimizerConfigEnum {
    AdamW(ParamsAdamW),
    SGD(f64),
}

pub enum OptimizerEnum {
    AdamW(AdamW),
    SGD(SGD),
}

#[async_trait]
impl Optimizer for OptimizerEnum {
    type Config = OptimizerConfigEnum;

    fn new(vars: Vec<Tensor>, config: Self::Config) -> anyhow::Result<Self> {
        match config {
            OptimizerConfigEnum::AdamW(params) => Ok(Self::AdamW(AdamW::new(vars, params)?)),
            OptimizerConfigEnum::SGD(params) => Ok(Self::SGD(SGD::new(vars, params)?)),
        }
    }

    async fn step(&mut self, device: &Device) -> anyhow::Result<()> {
        match self {
            OptimizerEnum::AdamW(opt) => opt.step(device).await,
            OptimizerEnum::SGD(opt) => opt.step(device).await,
        }
    }

    fn learning_rate(&self) -> f64 {
        match self {
            OptimizerEnum::AdamW(opt) => opt.learning_rate(),
            OptimizerEnum::SGD(opt) => opt.learning_rate(),
        }
    }

    fn set_learning_rate(&mut self, lr: f64) {
        match self {
            OptimizerEnum::AdamW(opt) => opt.set_learning_rate(lr),
            OptimizerEnum::SGD(opt) => opt.set_learning_rate(lr),
        }
    }

    fn parameters(&self) -> Vec<&Tensor> {
        match self {
            OptimizerEnum::AdamW(opt) => opt.parameters(),
            OptimizerEnum::SGD(opt) => opt.parameters(),
        }
    }
}

#[wasm_bindgen]
pub struct Trainer {
    model: GPT2,
    varmap: VarMap,
    optimizer: Box<dyn LRScheduler<OptimizerEnum> + Send + Sync>,
    device: Device,
    config: TrainerConfig,
}

#[wasm_bindgen]
impl Trainer {
    #[wasm_bindgen(constructor)]
    pub async fn new(config: JsValue) -> Result<Trainer, JsValue> {
        let cfg: TrainerConfig =
            serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from(e.to_string()))?;

        let device = Device::request_device(DeviceRequest::GPU)
            .await
            .map_err(|e| e.to_string())?;

        // Set seed on device if provided
        if let Some(seed) = cfg.seed {
            device.set_seed(seed);
        }

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let gpt2_config = Config {
            vocab_size: cfg.vocab_size,
            hidden_act: string_to_activation(&cfg.activation),
            n_embd: cfg.n_embd,
            n_layer: cfg.n_layer,
            n_head: cfg.n_head,
            block_size: cfg.block_size,
            attention_only: cfg.attention_only,
            positional_encoding: match cfg.positional_encoding.to_lowercase().as_str() {
                "rope" => PositionalEncoding::RoPE,
                "alibi" => PositionalEncoding::ALiBi,
                "sinusoidal" => PositionalEncoding::Sinusoidal,
                _ => PositionalEncoding::Learned,
            },
            layernorm_position: match cfg.layernorm_position.to_lowercase().as_str() {
                "pre" => LayerNormPosition::Pre,
                "post" => LayerNormPosition::Post,
                "none" => LayerNormPosition::None,
                _ => LayerNormPosition::Pre,
            },
            embd_pdrop: cfg.embd_pdrop,
            attn_pdrop: cfg.attn_pdrop,
            resid_pdrop: cfg.resid_pdrop,
        };

        device
            .try_gpu()
            .unwrap()
            .set_caching_enabled(cfg.caching_enabled);

        device
            .try_gpu()
            .unwrap()
            .set_inplace_support(cfg.inplace_support);

        let model = GPT2::new(&gpt2_config, vb, false)
            .await
            .map_err(|e| e.to_string())?;

        let vars = varmap
            .all_vars()
            .iter()
            .map(|var| var.to_owned())
            .collect::<Vec<_>>();

        let optimizer_config = match cfg.optimizer.optimizer_type.to_lowercase().as_str() {
            "sgd" => OptimizerConfigEnum::SGD(cfg.optimizer.lr),
            _ => OptimizerConfigEnum::AdamW(ParamsAdamW {
                lr: cfg.optimizer.lr,
                beta1: cfg.optimizer.beta1,
                beta2: cfg.optimizer.beta2,
                eps: cfg.optimizer.eps,
                weight_decay: cfg.optimizer.weight_decay,
            }),
        };

        let base_optimizer =
            OptimizerEnum::new(vars, optimizer_config).map_err(|e| e.to_string())?;

        // Wrap the optimizer in the selected scheduler
        let optimizer: Box<dyn LRScheduler<OptimizerEnum> + Send + Sync> =
            match cfg.optimizer.scheduler_type.to_lowercase().as_str() {
                "constant" => Box::new(ConstantLR::new(
                    base_optimizer,
                    cfg.optimizer.scheduler_factor,
                    cfg.optimizer.scheduler_steps,
                )),
                "linear" => Box::new(LinearLR::new(
                    base_optimizer,
                    1.0,
                    cfg.optimizer.scheduler_factor,
                    cfg.optimizer.scheduler_steps,
                )),
                "cosine" => Box::new(CosineAnnealingLR::new(
                    base_optimizer,
                    cfg.optimizer.scheduler_steps,
                    cfg.optimizer.scheduler_eta_min,
                )),
                _ => Box::new(NoopScheduler::new(base_optimizer)),
            };

        Ok(Trainer {
            model,
            varmap,
            optimizer,
            device,
            config: cfg,
        })
    }

    async fn forward(&mut self, input: Vec<Vec<i32>>) -> anyhow::Result<(Tensor, Tensor)> {
        let batch_size = input.len();
        let seq_len = input[0].len();

        let input_flat: Vec<i32> = input.into_iter().flatten().collect();
        let input_tensor = Tensor::from_data(
            input_flat,
            (batch_size, seq_len),
            TensorOptions::new().device(self.device.clone()),
        )?;

        let (logits, attn_masks) = self.model.schedule(GPT2Input {
            x: input_tensor,
            index_pos: 0,
        })?;
        Ok((logits, attn_masks))
    }

    #[wasm_bindgen(js_name = "forward")]
    pub async fn forward_js(&mut self, input: JsValue) -> Result<JsValue, JsValue> {
        reset_scope_context();
        let input: Vec<Vec<i32>> =
            serde_wasm_bindgen::from_value(input).map_err(|e| JsValue::from(e.to_string()))?;
        let (logits, _) = self.forward(input).await.map_err(|e| e.to_string())?;

        let logits_cpu = logits.to(&Device::CPU).await.map_err(|e| e.to_string())?;
        let logits_shape = logits_cpu.shape().to_vec();
        let logits_data = logits_cpu
            .to_vec::<f32>()
            .await
            .map_err(|e| e.to_string())?;

        Ok(
            serde_wasm_bindgen::to_value(&(logits_data, logits_shape))
                .map_err(|e| e.to_string())?,
        )
    }

    /// Train on a single batch of data provided directly from JavaScript
    #[wasm_bindgen]
    pub async fn train_on_batch(
        &mut self,
        input: JsValue,
        target: JsValue,
    ) -> Result<JsValue, JsValue> {
        reset_scope_context();
        let _train_mode_guard = ModuleModeGuard::new(ModuleMode::Train);
        // Convert input and target from JS arrays to Vec<Vec<i32>>
        let input: Vec<Vec<i32>> =
            serde_wasm_bindgen::from_value(input).map_err(|e| JsValue::from(e.to_string()))?;
        let target: Vec<Vec<i32>> =
            serde_wasm_bindgen::from_value(target).map_err(|e| JsValue::from(e.to_string()))?;

        if input.is_empty() || target.is_empty() {
            return Err("Empty batch provided".into());
        };

        let batch_size = input.len();
        let seq_len = input[0].len();
        if batch_size != target.len()
            || input.iter().any(|x| x.len() != seq_len)
            || target.iter().any(|x| x.len() != seq_len)
        {
            return Err("Inconsistent batch dimensions".into());
        }

        let (logits, attn_masks) = self.forward(input).await.map_err(|e| e.to_string())?;

        let target_flat: Vec<i32> = target.into_iter().flatten().collect();
        let target_tensor = Tensor::from_data(
            target_flat,
            (batch_size, seq_len),
            TensorOptions::new().device(self.device.clone()),
        )
        .map_err(|e| e.into_js_error())?;

        // Flatten the logits and targets, compute the cross-entropy loss with label smoothing
        let logits_flat = logits.clone().flatten(0, 1).map_err(|e| e.to_string())?;
        let target_flat = target_tensor.flatten(0, 1).map_err(|e| e.to_string())?;
        let inp = log_softmax(logits_flat, 1).map_err(|e| e.to_string())?;

        let alpha = self.config.label_smoothing;
        let loss = if alpha > 0.0 {
            label_smoothed_nll(inp, target_flat, alpha).map_err(|e| e.to_string())?
        } else {
            nll_masked(inp, target_flat).map_err(|e| e.to_string())?
        };

        // We sum the loss outside of x entropy because we want to see per-token loss
        let loss_summed = loss
            .clone()
            .sum(AllDims, false)
            .map_err(|e| e.to_string())?;

        loss_summed.backward().map_err(|e| e.to_string())?;

        // Run an optimizer step (updating model parameters).
        self.optimizer
            .step(&self.device)
            .await
            .map_err(|e| e.to_string())?;

        // Transfer tensors to CPU
        let loss_cpu = loss.to(&Device::CPU).await.map_err(|e| e.to_string())?;
        let loss_vec = loss_cpu.to_vec::<f32>().await.map_err(|e| e.to_string())?;

        // Transfer attention masks to CPU and get their shape and flattened data
        let attn_masks_cpu = attn_masks
            .to(&Device::CPU)
            .await
            .map_err(|e| e.to_string())?;
        let attn_masks_shape = attn_masks_cpu.shape().to_vec();
        let attn_masks_data = attn_masks_cpu
            .to_vec::<f32>()
            .await
            .map_err(|e| e.to_string())?;

        // Add logits to the result:
        let logits_cpu = logits.to(&Device::CPU).await.map_err(|e| e.to_string())?;
        let logits_shape = logits_cpu.shape().to_vec();
        let logits_data = logits_cpu
            .to_vec::<f32>()
            .await
            .map_err(|e| e.to_string())?;

        let result = serde_json::json!({
            "loss": {
                "tokens": loss_vec,
                "total": loss_vec.iter().sum::<f32>(),
            },
            "learning_rate": self.optimizer.get_lr(),
            "attn_masks": {
                "data": attn_masks_data,
                "shape": attn_masks_shape,
            },
            "logits": {
                "data": logits_data,
                "shape": logits_shape,
            }
        });

        serde_wasm_bindgen::to_value(&result).map_err(|e| e.to_string().into())
    }

    #[wasm_bindgen]
    pub fn usage_bytes(&self) -> u64 {
        self.device.try_gpu().unwrap().usage_bytes()
    }

    #[wasm_bindgen]
    pub async fn save_checkpoint(&self) -> Result<String, JsValue> {
        self.varmap.download_url().await
    }

    #[wasm_bindgen]
    pub fn set_step_log_config(&mut self, config: JsValue) -> Result<(), JsValue> {
        let config: JsStepLogConfig =
            serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from(e.to_string()))?;
        self.device
            .try_gpu()
            .unwrap()
            .set_step_log_config(piston::StepLogConfig {
                profiling: config.profiling,
                debug_selection: match config.debug_selection {
                    Some(JsDebugSelection::String(s)) => match s.as_str() {
                        "all" => Some(piston::DebugSelection::All),
                        _ => None,
                    },
                    Some(JsDebugSelection::Array(a)) => Some(piston::DebugSelection::Some(a)),
                    None => None,
                },
            });
        Ok(())
    }

    #[wasm_bindgen]
    pub fn take_step_log(&mut self) -> Option<StepLog> {
        self.device.try_gpu().unwrap().take_step_log()
    }

    #[wasm_bindgen]
    pub async fn generate(
        &mut self,
        prompt: JsValue,
        max_tokens: usize,
        callback: Option<Function>,
    ) -> Result<JsValue, JsValue> {
        // Convert prompt from JsValue -> Vec<i32>
        let prompt_vec: Vec<i32> =
            serde_wasm_bindgen::from_value(prompt).map_err(|e| JsValue::from(e.to_string()))?;

        // If a callback is provided, we do the streaming approach.
        if let Some(callback_fn) = callback {
            // Call existing GPT-2 generate with a streaming closure:
            generate(
                &mut self.model,
                prompt_vec,
                |tokens, logits_nd, attn_probs_data| {
                    // Convert the tokens to JS
                    let tokens_js = serde_wasm_bindgen::to_value(&tokens).unwrap_or(JsValue::NULL);

                    // Convert the logits to a {shape, data} object
                    let shape = vec![
                        logits_nd.shape()[0],
                        logits_nd.shape()[1],
                        logits_nd.shape()[2],
                    ];
                    let data: Vec<f32> = logits_nd.into_iter().collect();

                    let logits_obj = js_sys::Object::new();
                    let _ = js_sys::Reflect::set(
                        &logits_obj,
                        &JsValue::from_str("shape"),
                        &serde_wasm_bindgen::to_value(&shape).unwrap_or(JsValue::NULL),
                    );
                    let _ = js_sys::Reflect::set(
                        &logits_obj,
                        &JsValue::from_str("data"),
                        &serde_wasm_bindgen::to_value(&data).unwrap_or(JsValue::NULL),
                    );

                    let attn_probs_obj = js_sys::Object::new();
                    let _ = js_sys::Reflect::set(
                        &attn_probs_obj,
                        &JsValue::from_str("data"),
                        &serde_wasm_bindgen::to_value(&attn_probs_data).unwrap_or(JsValue::NULL),
                    );
                    // Finally, call the JS callback with (tokens, logitsObj)
                    let _ =
                        callback_fn.call3(&JsValue::NULL, &tokens_js, &logits_obj, &attn_probs_obj);
                },
                max_tokens,
            )
            .await
            .map_err(|e| JsValue::from(e.to_string()))?;

            // Return undefined if we're streaming via callback
            Ok(JsValue::UNDEFINED)
        } else {
            // No callback was provided, so we accumulate only the final tokens/logits and return them.
            let final_tokens: RefCell<Vec<i32>> = RefCell::new(Vec::new());
            let final_logits_data: RefCell<Vec<f32>> = RefCell::new(Vec::new());
            let final_logits_shape: RefCell<Vec<usize>> = RefCell::new(Vec::new());

            let prompt_len = prompt_vec.len();

            // We capture these as mut so we can overwrite them at each step with the latest pass
            generate(
                &mut self.model,
                prompt_vec,
                |tokens, logits_nd, _attn_probs_data| {
                    // Update tokens
                    final_tokens.borrow_mut().extend_from_slice(&tokens);

                    // Convert the logits into shape/data only for the final pass
                    final_logits_shape.borrow_mut().extend_from_slice(&[
                        logits_nd.shape()[0],
                        logits_nd.shape()[1],
                        logits_nd.shape()[2],
                    ]);

                    final_logits_data.borrow_mut().extend(logits_nd.into_iter());

                    // Note that we don't bother passing along the attn_probs_data here.
                    // This is inconsistent but we'll deal with it if it becomes an issue.
                },
                max_tokens,
            )
            .await
            .map_err(|e| JsValue::from(e.to_string()))?;

            let result = serde_json::json!({
                "tokens": final_tokens.borrow()[prompt_len..].to_vec(),
                "logits": {
                    "shape": final_logits_shape.borrow().to_vec(),
                    "data": final_logits_data.borrow().to_vec(),
                },
            });

            Ok(serde_wasm_bindgen::to_value(&result).map_err(|e| e.to_string())?)
        }
    }
}

/// A scheduler that does nothing, just passes through to the underlying optimizer.
/// This is used when no scheduler is selected.
pub struct NoopScheduler<O: Optimizer + Send + Sync> {
    core: LRSchedulerCore<O>,
}

impl<O: Optimizer + Send + Sync> NoopScheduler<O> {
    pub fn new(optimizer: O) -> Self {
        Self {
            core: LRSchedulerCore::new(optimizer),
        }
    }
}

impl<O: Optimizer + Send + Sync> LRScheduler<O> for NoopScheduler<O> {
    fn optimizer_mut(&mut self) -> &mut O {
        &mut self.core.optimizer
    }

    fn base_lr(&self) -> f64 {
        self.core.base_lr
    }

    fn step_count(&self) -> usize {
        self.core.step_count
    }

    fn set_step_count(&mut self, count: usize) {
        self.core.step_count = count;
    }

    fn compute_lr(&self) -> f64 {
        self.core.base_lr
    }
}
