mod attn;
// mod generate;
mod linear;
mod mlp;
mod model;

pub use model::Config;
pub use model::GPT2Input;
pub use model::LayerNormPosition;
pub use model::PositionalEncoding;
pub use model::GPT2;

// #[cfg(target_arch = "wasm32")]
// pub use generate::generate;
