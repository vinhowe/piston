mod activation;
mod alibi;
mod embedding;
mod groupnorm;
mod init;
mod kv_cache;
mod linear;
mod loss;
mod lr_scheduler;
mod norm;
mod optim;
mod rope;
mod sinusoidal;
mod util;
mod var_builder;
mod var_map;

pub use activation::*;
pub use alibi::*;
pub use embedding::*;
pub use groupnorm::*;
pub use init::*;
pub use kv_cache::*;
pub use linear::*;
pub use loss::*;
pub use lr_scheduler::*;
pub use norm::*;
pub use optim::*;
pub use rope::*;
pub use sinusoidal::*;
pub use util::*;
pub use var_builder::*;
pub use var_map::*;

use ratchet::Tensor;

/// # Module
///
/// Analagous to `torch.nn.Module` in PyTorch, a `Module` is a trait that represents a neural network
/// module. However, it has 1 key difference.
///
/// In PyTorch, `forward` performs the computation when called. In Ratchet, `schedule` is used to
/// schedule the computation for future execution. The Tensor returned is lazy, in that it
/// represents the result of the computation, but the computation itself has not been performed.
pub trait Module {
    type Input;
    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor>;
}

/// # MutableModule
///
/// Ditto above, but can mutate self.
pub trait MutableModule {
    type Input;
    fn schedule(&mut self, input: Self::Input) -> anyhow::Result<Tensor>;
}
