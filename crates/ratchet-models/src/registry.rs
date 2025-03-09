#![allow(non_local_definitions)]
//! # Registry
//!
//! The registry is responsible for surfacing available models to the user in both the CLI & WASM interfaces.

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::wasm_bindgen;

#[derive(Debug, Clone)]
#[cfg_attr(
    target_arch = "wasm32",
    derive(tsify::Tsify, serde::Serialize, serde::Deserialize),
    tsify(from_wasm_abi),
    serde(rename_all = "snake_case")
)]
#[cfg_attr(not(target_arch = "wasm32"), derive(clap::ValueEnum))]
pub enum PhiVariants {
    Phi2,
    Phi3,
}

/// # Available Models
///
/// This is a type safe way to surface models to users,
/// providing autocomplete **within** model families.
#[derive(Debug, Clone)]
#[non_exhaustive]
#[cfg_attr(
    target_arch = "wasm32",
    derive(tsify::Tsify, serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(target_arch = "wasm32", tsify(from_wasm_abi))]
pub enum AvailableModels {
    Phi(PhiVariants),
    Moondream,
}

impl AvailableModels {
    pub fn repo_id(&self) -> String {
        let id = match self {
            AvailableModels::Phi(p) => match p {
                PhiVariants::Phi2 => "FL33TW00D-HF/phi2",
                PhiVariants::Phi3 => "FL33TW00D-HF/phi3",
            },
            AvailableModels::Moondream => "ratchet-community/ratchet-moondream-2",
        };
        id.to_string()
    }

    pub fn model_id(&self, quantization: Quantization) -> String {
        let model_stem = match self {
            AvailableModels::Phi(p) => match p {
                PhiVariants::Phi2 => "phi2",
                PhiVariants::Phi3 => "phi3-mini-4k",
            },
            AvailableModels::Moondream => "moondream",
        };
        match quantization {
            Quantization::Q8_0 => format!("{}_q8_0.gguf", model_stem),
            Quantization::F16 => format!("{}_f16.gguf", model_stem),
            Quantization::F32 => format!("{}_f32.gguf", model_stem),
        }
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
#[cfg_attr(not(target_arch = "wasm32"), derive(clap::ValueEnum))]
pub enum Quantization {
    Q8_0,
    F16,
    F32,
}
