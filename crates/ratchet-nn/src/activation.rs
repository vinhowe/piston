use ratchet::Tensor;

use crate::Module;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Activation {
    #[default]
    Gelu,
    Relu,
    Relu2,
    Silu,
    Sigmoid,
    Swiglu,
}

impl Module for Activation {
    type Input = Tensor;
    type Output = Tensor;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Self::Output> {
        match self {
            Self::Gelu => input.gelu(),
            Self::Relu => input.relu(),
            Self::Relu2 => input.relu2(),
            Self::Silu => input.silu(),
            Self::Sigmoid => input.sigmoid(),
            Self::Swiglu => input.swiglu(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct PReLU {
    weight: Tensor,
    is_scalar: bool,
}

impl PReLU {
    pub fn new(weight: Tensor, is_scalar: bool) -> Self {
        Self { weight, is_scalar }
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn is_scalar(&self) -> bool {
        self.is_scalar
    }
}
