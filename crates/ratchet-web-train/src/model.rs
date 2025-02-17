use anyhow::Result;
use async_trait::async_trait;
use ratchet::{DType, Device, DeviceRequest, GradStore, Tensor, Var};
use ratchet_datasets::batcher::IterResult2;
use ratchet_datasets::{
    nlp::{
        tinystories::{
            Dataset as TinyStoriesDataset, DatasetRandomIter as TinyStoriesDatasetRandomIter,
        },
        toy::{
            AddTask, CountTask, ModAddTask, SlapjackTask, SortTask, ToyTaskIter, TwoSumTask,
            ZerosTask,
        },
    },
    Batcher,
};
use ratchet_models::gpt2::{Config, GPT2Input, PositionalEncoding};
use ratchet_models::gpt2::{LayerNormPosition, GPT2};
use ratchet_nn::{
    clip_grad_norm, cross_entropy, Activation, AdamW, ConstantLR, CosineAnnealingLR, LRScheduler,
    LRSchedulerCore, LinearLR, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap, SGD,
};
use serde::{Deserialize, Serialize};
use std::iter::Iterator;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    let logger = fern::Dispatch::new()
        .format(|out, message, record| {
            out.finish(format_args!(
                "{}[{}][{}] {}",
                chrono::Local::now().format("[%Y-%m-%d][%H:%M:%S]"),
                record.target(),
                record.level(),
                message
            ))
        })
        .level_for("tokenizers", log::LevelFilter::Off)
        .level(log::LevelFilter::Info)
        .chain(fern::Output::call(console_log::log))
        .apply();
    match logger {
        Ok(_) => log::info!("Logging initialized."),
        Err(error) => eprintln!("Error initializing logging: {:?}", error),
    }
}

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

// -------------------------------------------------------------------
// Trainer configuration that is deserializable from JavaScript.
// Note the additional fields for batch size and dataset selection.
// -------------------------------------------------------------------
#[derive(Serialize, Deserialize)]
pub struct TrainerConfig {
    pub vocab_size: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub block_size: usize,
    pub batch_size: usize,
    /// A string specifying which dataset to use: "two_sum" (default), "zeros", or "tinystories".
    pub dataset: String,
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

fn default_seed() -> Option<u64> {
    None
}

fn string_to_activation(s: &str) -> Activation {
    match s.to_lowercase().as_str() {
        "gelu" => Activation::Gelu,
        "relu" => Activation::Relu,
        "relu2" => Activation::Relu2,
        "silu" => Activation::Silu,
        "sigmoid" => Activation::Sigmoid,
        "swiglu" => Activation::Swiglu,
        _ => Activation::Relu2, // Default to Relu2 for unrecognized values
    }
}

// -------------------------------------------------------------------
// Batch and BatchHandle definitions.
// The inner Batch holds the (input, target) tensors on the current device.
// -------------------------------------------------------------------
pub struct Batch {
    pub input: Tensor,
    pub target: Tensor,
}

#[wasm_bindgen(js_name = "Batch")]
pub struct BatchHandle {
    inner: Batch,
}

#[wasm_bindgen]
impl BatchHandle {
    /// Asynchronously converts the underlying batch data into a JS object.
    /// This operation transfers the tensors to the CPU and then converts them into Vec<f32>.
    #[wasm_bindgen]
    pub async fn to_js(&self) -> Result<JsValue, JsValue> {
        // Transfer the input and target tensors to CPU.
        let input_cpu = self
            .inner
            .input
            .to(&Device::CPU)
            .await
            .map_err(|e| e.to_string())?;
        let target_cpu = self
            .inner
            .target
            .to(&Device::CPU)
            .await
            .map_err(|e| e.to_string())?;
        // Convert the tensors to Vec<f32>.
        let input_vec = input_cpu.to_vec::<f32>().map_err(|e| e.to_string())?;
        let target_vec = target_cpu.to_vec::<f32>().map_err(|e| e.to_string())?;
        // Create a JS-friendly object.
        let batch_obj = serde_json::json!({
            "input": input_vec,
            "target": target_vec,
        });
        serde_wasm_bindgen::to_value(&batch_obj).map_err(|e| e.to_string().into())
    }
}

// Updated to:

enum BatcherType<'a> {
    ToyTwoSum(
        Batcher<
            IterResult2<
                ratchet_datasets::nlp::toy::ToyTaskIter<ratchet_datasets::nlp::toy::TwoSumTask>,
            >,
        >,
    ),
    ToyZeros(
        Batcher<
            IterResult2<
                ratchet_datasets::nlp::toy::ToyTaskIter<ratchet_datasets::nlp::toy::ZerosTask>,
            >,
        >,
    ),
    ToySort(
        Batcher<
            IterResult2<
                ratchet_datasets::nlp::toy::ToyTaskIter<ratchet_datasets::nlp::toy::SortTask>,
            >,
        >,
    ),
    ToyAdd(
        Batcher<
            IterResult2<
                ratchet_datasets::nlp::toy::ToyTaskIter<ratchet_datasets::nlp::toy::AddTask>,
            >,
        >,
    ),
    ToyCount(
        Batcher<
            IterResult2<
                ratchet_datasets::nlp::toy::ToyTaskIter<ratchet_datasets::nlp::toy::CountTask>,
            >,
        >,
    ),
    ToySlapjack(
        Batcher<
            IterResult2<
                ratchet_datasets::nlp::toy::ToyTaskIter<ratchet_datasets::nlp::toy::SlapjackTask>,
            >,
        >,
    ),
    ToyModAdd(
        Batcher<
            IterResult2<
                ratchet_datasets::nlp::toy::ToyTaskIter<ratchet_datasets::nlp::toy::ModAddTask>,
            >,
        >,
    ),
    TinyStories(Batcher<IterResult2<ratchet_datasets::nlp::tinystories::DatasetRandomIter<'a>>>),
}

/// An enum for optimizer configurations
#[derive(Debug, Clone)]
pub enum OptimizerConfigEnum {
    AdamW(ParamsAdamW),
    SGD(f64),
}

/// An enum that wraps the concrete optimizer types
pub enum OptimizerEnum {
    AdamW(AdamW),
    SGD(SGD),
}

#[async_trait]
impl Optimizer for OptimizerEnum {
    type Config = OptimizerConfigEnum;

    fn new(vars: Vec<(Option<String>, Var)>, config: Self::Config) -> anyhow::Result<Self> {
        match config {
            OptimizerConfigEnum::AdamW(params) => Ok(Self::AdamW(AdamW::new(vars, params)?)),
            OptimizerConfigEnum::SGD(params) => Ok(Self::SGD(SGD::new(vars, params)?)),
        }
    }

    async fn step(&mut self, grads: &GradStore, device: &Device) -> anyhow::Result<()> {
        match self {
            OptimizerEnum::AdamW(opt) => opt.step(grads, device).await,
            OptimizerEnum::SGD(opt) => opt.step(grads, device).await,
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
}

// -------------------------------------------------------------------
// The Trainer manages the model, optimizer, device, and dataset batcher.
// -------------------------------------------------------------------
#[wasm_bindgen]
pub struct Trainer {
    model: GPT2,
    optimizer: Box<dyn LRScheduler<OptimizerEnum> + Send + Sync>,
    device: Device,
    config: TrainerConfig,
    batcher: BatcherType<'static>,
}

#[wasm_bindgen]
impl Trainer {
    /// Create a new Trainer from a JavaScript configuration object.
    /// This async constructor initializes the device, model, optimizer,
    /// and dataset iterator (wrapped in a Batcher) based on the config.
    #[wasm_bindgen(constructor)]
    pub async fn new(config: JsValue) -> Result<Trainer, JsValue> {
        // Deserialize the configuration from JS.
        let cfg: TrainerConfig =
            serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from(e.to_string()))?;

        // Request a GPU device.
        let device = Device::request_device(DeviceRequest::GPU)
            .await
            .map_err(|e| e.to_string())?;

        // Set the seed if provided
        if let Some(seed) = cfg.seed {
            device.set_seed(seed);
        }

        // Set up the variable map and variable builder for model initialization.
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // Create the GPT2 model configuration.
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
                _ => LayerNormPosition::Pre,
            },
        };
        // Initialize the GPT2 model.
        let model = GPT2::new(&gpt2_config, vb)
            .await
            .map_err(|e| e.to_string())?;

        // Set up the optimizer based on the configuration
        let vars = varmap
            .all_labeled_vars()
            .iter()
            .map(|(label, var)| (Some(label.to_owned()), var.to_owned()))
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
                    1.0,                            // start_factor
                    cfg.optimizer.scheduler_factor, // end_factor
                    cfg.optimizer.scheduler_steps,
                )),
                "cosine" => Box::new(CosineAnnealingLR::new(
                    base_optimizer,
                    cfg.optimizer.scheduler_steps,
                    cfg.optimizer.scheduler_eta_min,
                )),
                _ => Box::new(NoopScheduler::new(base_optimizer)), // Default to no scheduler
            };

        // Based on the config.dataset value, choose the dataset iterator and wrap it in a Batcher.
        // Supported options (case-insensitive): "two_sum" (default), "zeros", "tinystories", "sort", "add", "count", "slapjack", "mod_add"
        let batcher = match cfg.dataset.to_lowercase().as_str() {
            "zeros" => {
                let task = ZerosTask::new(cfg.block_size);
                let dataset_iter = ToyTaskIter::new(task, device.clone());
                BatcherType::ToyZeros(Batcher::new_r2(dataset_iter).batch_size(cfg.batch_size))
            }
            "sort" => {
                let task = SortTask::new(20, cfg.seed);
                let dataset_iter = ToyTaskIter::new(task, device.clone());
                BatcherType::ToySort(Batcher::new_r2(dataset_iter).batch_size(cfg.batch_size))
            }
            "add" => {
                let task = AddTask::new(1000, cfg.seed);
                let dataset_iter = ToyTaskIter::new(task, device.clone());
                BatcherType::ToyAdd(Batcher::new_r2(dataset_iter).batch_size(cfg.batch_size))
            }
            "count" => {
                let task = CountTask::new(10, 'd', cfg.seed);
                let dataset_iter = ToyTaskIter::new(task, device.clone());
                BatcherType::ToyCount(Batcher::new_r2(dataset_iter).batch_size(cfg.batch_size))
            }
            "slapjack" => {
                let task = SlapjackTask::new(48, cfg.seed);
                let dataset_iter = ToyTaskIter::new(task, device.clone());
                BatcherType::ToySlapjack(Batcher::new_r2(dataset_iter).batch_size(cfg.batch_size))
            }
            "mod_add" => {
                let task = ModAddTask::new(113, cfg.seed);
                let dataset_iter = ToyTaskIter::new(task, device.clone());
                BatcherType::ToyModAdd(Batcher::new_r2(dataset_iter).batch_size(cfg.batch_size))
            }
            // "tinystories" => {
            //     let dataset = TinyStoriesDataset::new();
            //     let dataset_iter = TinyStoriesDatasetRandomIter::new(dataset, device.clone());
            //     BatcherEnum::TinyStories(Batcher::new_r2(dataset_iter).batch_size(cfg.batch_size))
            // }
            // Default (including "two_sum" and unrecognized values): use the TwoSum task.
            _ => {
                let task = TwoSumTask::new(5, 5, cfg.seed);
                let dataset_iter = ToyTaskIter::new(task, device.clone());
                BatcherType::ToyTwoSum(Batcher::new_r2(dataset_iter).batch_size(cfg.batch_size))
            }
        };

        Ok(Trainer {
            model,
            optimizer,
            device,
            config: cfg,
            batcher,
        })
    }

    /// Asynchronously retrieve the next batch as a BatchHandle.
    /// The returned BatchHandle wraps a Batch consisting of (input, target) tensors,
    /// which reside on the device.
    #[wasm_bindgen]
    pub async fn next_batch(&mut self) -> Result<BatchHandle, JsValue> {
        // Pull the next (input, target) tuple from the dataset iterator.
        let (input, target) = match &mut self.batcher {
            BatcherType::ToyTwoSum(batcher) => batcher
                .next()
                .ok_or_else(|| {
                    JsValue::from_str("No more data available in the ToyTwoSum dataset")
                })?
                .map_err(|e| e.to_string())?,
            BatcherType::ToyZeros(batcher) => batcher
                .next()
                .ok_or_else(|| JsValue::from_str("No more data available in the ToyZeros dataset"))?
                .map_err(|e| e.to_string())?,
            BatcherType::ToySort(batcher) => batcher
                .next()
                .ok_or_else(|| JsValue::from_str("No more data available in the ToySort dataset"))?
                .map_err(|e| e.to_string())?,
            BatcherType::ToyAdd(batcher) => batcher
                .next()
                .ok_or_else(|| JsValue::from_str("No more data available in the ToyAdd dataset"))?
                .map_err(|e| e.to_string())?,
            BatcherType::ToyCount(batcher) => batcher
                .next()
                .ok_or_else(|| JsValue::from_str("No more data available in the ToyCount dataset"))?
                .map_err(|e| e.to_string())?,
            BatcherType::ToySlapjack(batcher) => batcher
                .next()
                .ok_or_else(|| {
                    JsValue::from_str("No more data available in the ToySlapjack dataset")
                })?
                .map_err(|e| e.to_string())?,
            BatcherType::ToyModAdd(batcher) => batcher
                .next()
                .ok_or_else(|| {
                    JsValue::from_str("No more data available in the ToyModAdd dataset")
                })?
                .map_err(|e| e.to_string())?,
            BatcherType::TinyStories(batcher) => batcher
                .next()
                .ok_or_else(|| {
                    JsValue::from_str("No more data available in the TinyStories dataset")
                })?
                .map_err(|e| e.to_string())?,
        };
        Ok(BatchHandle {
            inner: Batch { input, target },
        })
    }

    /// Asynchronously train on the provided batch.
    /// Here we assume that `batch.input` and `batch.target` are already on the correct device.
    /// They are passed directly into the model; loss, backpropagation, and the optimizer step
    /// are performed, and the scalar loss value is returned.
    #[wasm_bindgen]
    pub async fn train_step(&mut self, batch_handle: BatchHandle) -> Result<JsValue, JsValue> {
        // Extract the batch from the handle.
        let batch = batch_handle.inner;

        // Forward pass: compute logits from the model.
        let (logits, attn_masks) = self
            .model
            .schedule(GPT2Input {
                x: batch.input,
                index_pos: 0,
            })
            .map_err(|e| e.to_string())?;

        // Flatten the logits and targets.
        let logits_flat = logits.flatten_to(1).map_err(|e| e.to_string())?;
        let target_flat = batch.target.flatten_to(1).map_err(|e| e.to_string())?;

        // Compute the cross-entropy loss.
        let loss = cross_entropy(logits_flat, target_flat).map_err(|e| e.to_string())?;

        // Backpropagate to compute gradients.
        let grads = loss.backward().map_err(|e| e.to_string())?;

        // Run an optimizer step (updating model parameters).
        self.optimizer
            .step(&grads, &self.device)
            .await
            .map_err(|e| e.to_string())?;

        log::debug!("Done training step");

        // Transfer the loss to CPU and extract its scalar value.
        let loss_cpu = loss.to(&Device::CPU).await.map_err(|e| e.to_string())?;
        let loss_vec = loss_cpu.to_vec::<f32>().map_err(|e| e.to_string())?;

        // Transfer attention masks to CPU and get their shape and flattened data
        let attn_masks_cpu = attn_masks
            .to(&Device::CPU)
            .await
            .map_err(|e| e.to_string())?;
        let attn_masks_shape = attn_masks_cpu.shape().to_vec();
        let attn_masks_data = attn_masks_cpu.to_vec::<f32>().map_err(|e| e.to_string())?;

        // Create a JS-friendly object with both loss and attention masks
        let result = serde_json::json!({
            "loss": loss_vec[0],
            "learning_rate": self.optimizer.get_lr(),
            "attn_masks": {
                "data": attn_masks_data,
                "shape": attn_masks_shape,
            }
        });

        serde_wasm_bindgen::to_value(&result).map_err(|e| e.to_string().into())
    }

    #[wasm_bindgen]
    pub fn usage_bytes(&self) -> u64 {
        self.device.try_gpu().unwrap().usage_bytes()
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

#[cfg(all(test, target_arch = "wasm32"))]
mod tests {
    use crate::test_utils::log_init;

    use super::*;
    use ratchet::{
        shape,
        test_utils::{to_vec0_round, to_vec1_round},
    };
    use ratchet_nn::Linear;
    use wasm_bindgen_test::*;

    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    async fn train_browser() -> Result<(), JsValue> {
        log_init();
        let json_config = r#"{
                "vocab_size": 1024,
                "n_embd": 768,
                "n_layer": 6,
                "n_head": 6,
                "block_size": 24,
                "batch_size": 1,
                "dataset": "two_sum",
                "optimizer": {
                    "lr": 1e-3
                }
            }"#;
        // Parse the JSON string into a TrainerConfig first to validate it
        let config: TrainerConfig =
            serde_json::from_str(json_config).map_err(|e| JsValue::from(e.to_string()))?;
        // Convert to JsValue
        let config_js =
            serde_wasm_bindgen::to_value(&config).map_err(|e| JsValue::from(e.to_string()))?;

        let mut trainer = Trainer::new(config_js).await.unwrap();
        for i in 0..10 {
            let batch = trainer.next_batch().await.unwrap();
            let loss = trainer.train_step(batch).await.unwrap();
            log::error!("step {}: loss: {:?}", i, loss);
        }
        // log::error!("config_js: {:?}", config_js);
        Ok(())
    }
}
