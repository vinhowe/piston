use maybe_async::maybe_async;
use piston::Tensor;
use piston_macros::scoped_module;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerNormConfig {
    pub eps: f32,
    pub remove_mean: bool,
}

impl Default for LayerNormConfig {
    fn default() -> Self {
        Self {
            eps: 1e-5,
            remove_mean: true,
        }
    }
}

impl From<f32> for LayerNormConfig {
    fn from(eps: f32) -> Self {
        Self {
            eps,
            remove_mean: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LayerNorm {
    weight: Tensor,
    bias: Option<Tensor>,
    remove_mean: bool,
    eps: f32,
}

impl LayerNorm {
    pub fn new(weight: Tensor, bias: Option<Tensor>, eps: f32) -> Self {
        Self {
            weight,
            bias,
            remove_mean: true,
            eps,
        }
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

#[scoped_module]
impl crate::Module for LayerNorm {
    type Input = Tensor;
    type Output = Tensor;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        input.layer_norm(Some(self.weight.clone()), self.bias.clone(), self.eps)
    }
}

#[maybe_async]
pub async fn layer_norm(
    size: usize,
    config: LayerNormConfig,
    vb: crate::VarBuilder<'_>,
) -> anyhow::Result<LayerNorm> {
    let weight = vb
        .get_with_hints(size, "weight", crate::Init::Const(1.))
        .await?;
    let bias = vb
        .get_with_hints(size, "bias", crate::Init::Const(0.))
        .await?;
    Ok(LayerNorm {
        weight,
        bias: Some(bias),
        remove_mean: config.remove_mean,
        eps: config.eps,
    })
}

/// RMSNorm
///
/// https://github.com/NVIDIA/apex/pull/1274/files
#[derive(Clone, Debug, derive_new::new)]
pub struct RMSNorm {
    weight: Tensor,
    eps: f32,
}

impl RMSNorm {
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
}

#[scoped_module]
impl crate::Module for RMSNorm {
    type Input = Tensor;
    type Output = Tensor;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Self::Output> {
        let src_dtype = input.dtype();
        input
            .float()?
            .rms_norm(Some(self.weight.clone()), self.eps)?
            .cast(src_dtype)
    }
}

#[maybe_async]
pub async fn rms_norm(size: usize, eps: f32, vb: crate::VarBuilder<'_>) -> anyhow::Result<RMSNorm> {
    let weight = vb
        .get_with_hints(size, "weight", crate::Init::Const(1.))
        .await?;
    Ok(RMSNorm::new(weight, eps))
}

#[cfg(feature = "pyo3")]
mod tests {
    use super::{layer_norm, LayerNorm, LayerNormConfig};
    use crate::{Module, VarBuilder, VarMap};
    use piston::{
        prelude::shape,
        test_util::{run_py_prg, run_py_prg_multiple},
        Device, DeviceRequest, Parameter, Tensor,
    };
    use test_strategy::{proptest, Arbitrary};

    thread_local! {
        static GPU_DEVICE: Device = Device::request_device(DeviceRequest::GPU).unwrap();
    }

    fn ground_truth_forward(
        x: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        eps: f32,
    ) -> anyhow::Result<Tensor> {
        let prg = format!(
            r#"
import torch
import torch.nn as nn

def layer_norm(x, weight, bias=None):
    x_tensor = torch.from_numpy(x)
    weight_tensor = torch.from_numpy(weight)
    bias_tensor = torch.from_numpy(bias) if bias is not None else None
    
    normalized_shape = x_tensor.shape[-1]
    ln = nn.LayerNorm(normalized_shape, eps={}, elementwise_affine=False)
    ln.weight = nn.Parameter(weight_tensor)
    ln.bias = nn.Parameter(bias_tensor) if bias_tensor is not None else None
    
    out = ln(x_tensor)
    
    return out.float().detach().numpy()
"#,
            eps,
        );

        let result = match bias {
            Some(b) => run_py_prg(prg, &[x, weight, b], &[], x.dtype())?,
            None => run_py_prg(prg, &[x, weight], &[], x.dtype())?,
        };
        Ok(result)
    }

    fn ground_truth_backward(
        x: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        eps: f32,
    ) -> anyhow::Result<(Tensor, Tensor, Option<Tensor>)> {
        let prg = format!(
            r#"
import torch
import torch.nn as nn

def layer_norm_backward(x, weight, bias = None):
    x_tensor = torch.tensor(torch.from_numpy(x), requires_grad=True)
    weight_tensor = torch.tensor(torch.from_numpy(weight), requires_grad=True)
    bias_tensor = torch.tensor(torch.from_numpy(bias), requires_grad=True) if bias is not None else None
    
    normalized_shape = x_tensor.shape[-1]
    ln = nn.LayerNorm(normalized_shape, eps={}, elementwise_affine=False)
    ln.weight = nn.Parameter(weight_tensor)
    ln.bias = nn.Parameter(bias_tensor) if bias_tensor is not None else None
    
    out = ln(x_tensor)
    
    out.backward(torch.ones_like(out))

    return (
        x_tensor.grad.numpy(),
        ln.weight.grad.numpy(),
        ln.bias.grad.numpy() if bias_tensor is not None else None
    )
"#,
            eps
        );

        let result = match bias {
            Some(b) => run_py_prg_multiple(prg, &[x, weight, b], &[])?,
            None => run_py_prg_multiple(prg, &[x, weight], &[])?,
        };
        Ok((
            result[0].clone(),
            result[1].clone(),
            bias.map(|_| result[2].clone()),
        ))
    }

    #[derive(Arbitrary, Debug)]
    struct LayerNormProblem {
        #[strategy(1..=64usize)]
        batch_size: usize,
        #[strategy(1..=64usize)]
        seq_len: usize,
        #[strategy(1..=64usize)]
        hidden_size: usize,
        #[strategy(1e-5f32..=1e5f32)]
        eps: f32,
    }

    fn run_layernorm_forward_trial(problem: LayerNormProblem) -> anyhow::Result<()> {
        let device = GPU_DEVICE.with(|d| d.clone());
        let LayerNormProblem {
            batch_size,
            seq_len,
            hidden_size,
            eps,
        } = problem;

        let x = Tensor::randn::<f32, _>(0., 1., (batch_size, seq_len, hidden_size), Device::CPU)?;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, piston::DType::F32, &device);

        let config = LayerNormConfig {
            eps,
            remove_mean: true,
        };
        let layer_norm = layer_norm(hidden_size, config, vb.clone())?;

        let weight = layer_norm.weight();
        let bias = layer_norm.bias();
        let weight_cpu = weight.to(&Device::CPU)?;
        let bias_cpu = bias.map(|b| b.to(&Device::CPU).unwrap());

        let ground = ground_truth_forward(&x, &weight_cpu, bias_cpu.as_ref(), eps)?;

        let x_gpu = x.to(&device)?;
        let layer_norm_gpu = LayerNorm {
            weight: weight.to(&device)?,
            bias: bias.map(|b| b.to(&device)).transpose()?,
            eps,
            remove_mean: true,
        };
        let result_gpu = layer_norm_gpu.schedule(x_gpu)?;

        let ours = result_gpu.to(&Device::CPU)?;
        ground.all_close(&ours, 1e-4, 1e-4)?;
        Ok(())
    }

    fn run_layernorm_backward_trial(problem: LayerNormProblem) -> anyhow::Result<()> {
        let device = GPU_DEVICE.with(|d| d.clone());
        let LayerNormProblem {
            batch_size,
            seq_len,
            hidden_size,
            eps,
        } = problem;

        let x = Tensor::randn::<f32, _>(0., 1., (batch_size, seq_len, hidden_size), Device::CPU)?;
        let x_gpu = x.to(&device)?;
        let x_var = Parameter::from_tensor(&x_gpu)?;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, piston::DType::F32, &device);

        let config = LayerNormConfig {
            eps,
            remove_mean: true,
        };
        let layer_norm = layer_norm(hidden_size, config, vb.clone())?;

        let weight = layer_norm.weight();
        let bias = layer_norm.bias();
        let weight_cpu = weight.to(&Device::CPU)?;
        let bias_cpu = bias.map(|b| b.to(&Device::CPU).unwrap());

        let (ground_x_grad, ground_weight_grad, ground_bias_grad) =
            ground_truth_backward(&x, &weight_cpu, bias_cpu.as_ref(), eps)?;

        let result_gpu = layer_norm.schedule(x_var.as_tensor().clone())?;

        result_gpu.backward()?;
        device.try_gpu()?.mark_step()?;

        let x_grad = x_var.as_tensor().grad().unwrap().to(&Device::CPU)?;
        let weight_grad = weight.grad().unwrap().to(&Device::CPU)?;
        let bias_grad = bias
            .as_ref()
            .map(|b| b.grad().unwrap().to(&Device::CPU).unwrap());

        ground_x_grad.all_close(&x_grad, 1e-4, 1e-4)?;
        ground_weight_grad.all_close(&weight_grad, 1e-4, 1e-4)?;

        if let Some(bias_grad) = &bias_grad {
            ground_bias_grad
                .as_ref()
                .unwrap()
                .all_close(bias_grad, 1e-4, 1e-4)?;
        }

        Ok(())
    }

    #[proptest(cases = 16)]
    fn test_layernorm_forward(prob: LayerNormProblem) {
        run_layernorm_forward_trial(prob).unwrap();
    }

    #[proptest(cases = 16)]
    fn test_layernorm_backward(prob: LayerNormProblem) {
        run_layernorm_backward_trial(prob).unwrap();
    }
}
