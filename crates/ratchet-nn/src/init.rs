//! Variable initialization.
// This is based on:
// https://github.com/pytorch/pytorch/blob/07107919297db3f8ab37f11c12666b6d6d5f692e/torch/nn/init.py#
use ratchet::{Device, Shape, Tensor, Var};

/// Number of features as input or output of a layer.
/// In Kaiming initialization, choosing `FanIn` preserves
/// the magnitude of the variance of the weights in the
/// forward pass, choosing `FanOut` preserves this
/// magnitude in the backward pass.
#[derive(Debug, Copy, Clone)]
pub enum FanInOut {
    FanIn,
    FanOut,
}

impl FanInOut {
    /// Compute the fan-in or fan-out value for a weight tensor of
    /// the specified dimensions.
    /// <https://github.com/pytorch/pytorch/blob/dbeacf11820e336e803bb719b7aaaf2125ae4d9c/torch/nn/init.py#L284>
    pub fn for_shape(&self, shape: &Shape) -> usize {
        let dims = shape.to_vec();
        let receptive_field_size: usize = dims.iter().skip(2).product();
        match &self {
            FanInOut::FanIn => {
                if dims.len() < 2 {
                    1
                } else {
                    dims[1] * receptive_field_size
                }
            }
            FanInOut::FanOut => {
                if dims.is_empty() {
                    1
                } else {
                    dims[0] * receptive_field_size
                }
            }
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum NormalOrUniform {
    Normal,
    Uniform,
}

/// The non-linear function that follows this layer. ReLU is the
/// recommended value.
#[derive(Debug, Copy, Clone)]
pub enum NonLinearity {
    ReLU,
    Linear,
    Sigmoid,
    Tanh,
    SELU,
    ExplicitGain(f32),
}

impl NonLinearity {
    // https://github.com/pytorch/pytorch/blob/07107919297db3f8ab37f11c12666b6d6d5f692e/torch/nn/init.py#L67
    pub fn gain(&self) -> f32 {
        match *self {
            NonLinearity::ReLU => 2f32.sqrt(),
            NonLinearity::Tanh => 5. / 3.,
            NonLinearity::Linear | NonLinearity::Sigmoid => 1.,
            NonLinearity::SELU => 0.75,
            NonLinearity::ExplicitGain(g) => g,
        }
    }
}

/// Variable initializations.
#[derive(Debug, Copy, Clone)]
pub enum Init {
    /// Constant value.
    Const(f32),

    /// Random normal with some mean and standard deviation.
    Randn { mean: f32, stdev: f32 },

    /// Uniform initialization between some lower and upper bounds.
    Uniform { lo: f32, up: f32 },

    /// Kaiming uniform initialization.
    /// See "Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification"
    /// He, K. et al. (2015). This uses a uniform distribution.
    Kaiming {
        dist: NormalOrUniform,
        fan: FanInOut,
        non_linearity: NonLinearity,
    },
}

pub const ZERO: Init = Init::Const(0.);
pub const ONE: Init = Init::Const(1.);

pub const DEFAULT_KAIMING_UNIFORM: Init = Init::Kaiming {
    dist: NormalOrUniform::Uniform,
    fan: FanInOut::FanIn,
    non_linearity: NonLinearity::ReLU,
};

pub const DEFAULT_KAIMING_NORMAL: Init = Init::Kaiming {
    dist: NormalOrUniform::Normal,
    fan: FanInOut::FanIn,
    non_linearity: NonLinearity::ReLU,
};

impl Init {
    /// Creates a new tensor with the specified shape, device, and initialization.
    pub fn var(&self, s: &Shape, device: Device) -> anyhow::Result<Var> {
        match self {
            Self::Const(v) if *v == 0. => Ok(Var::zeros::<f32>(s, &device)),
            Self::Const(v) if *v == 1. => Ok(Var::ones::<f32>(s, &device)),
            Self::Const(cst) => Ok(Var::full(s, *cst, &device)),
            Self::Uniform { lo, up } => Ok(Var::rand::<f32>(*lo, *up, s.clone(), device)),
            Self::Randn { mean, stdev } => Ok(Var::randn::<f32>(*mean, *stdev, s.clone(), device)),
            Self::Kaiming {
                dist,
                fan,
                non_linearity,
            } => {
                let fan = fan.for_shape(s);
                let gain = non_linearity.gain();
                let std = gain / (fan as f32).sqrt();
                match dist {
                    NormalOrUniform::Uniform => {
                        let bound = 3f32.sqrt() * std;
                        Ok(Var::rand::<f32>(-bound, bound, s.clone(), device))
                    }
                    NormalOrUniform::Normal => Ok(Var::randn::<f32>(0., std, s.clone(), device)),
                }
            }
        }
    }
}

impl Default for Init {
    fn default() -> Self {
        Self::Const(0.)
    }
}
