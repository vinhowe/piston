use maybe_async::maybe_async;
use ratchet::{shape, Tensor};
use ratchet_macros::scoped_module;

use crate::Module;

/// # Linear
///
/// PyTorch case: y = xW^T + b
/// If your weights are already in the correct layout, you can set `transpose` to `false` to avoid the transpose operation.
#[derive(derive_new::new, Debug)]
pub struct Linear {
    pub w: Tensor,
    b: Option<Tensor>,
}

#[scoped_module]
impl Module for Linear {
    type Input = Tensor;
    type Output = Tensor;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Self::Output> {
        let w = match *input.shape().to_vec() {
            [b1, b2, _, _] => self.w.clone().broadcast_left(shape![b1, b2])?,
            [bsize, _, _] => self.w.clone().broadcast_left(shape![bsize])?,
            _ => self.w.clone(),
        };
        let x = input.matmul(w, false, true)?;

        match &self.b {
            None => Ok(x),
            Some(b) => x.clone() + b.clone().cast(x.dt())?,
        }
    }
}

#[maybe_async]
pub async fn linear(
    in_dim: usize,
    out_dim: usize,
    vb: crate::VarBuilder<'_>,
) -> anyhow::Result<Linear> {
    let init_ws = crate::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb
        .get_with_hints(shape![out_dim, in_dim], "weight", init_ws)
        .await?;
    let bound = 1. / (in_dim as f32).sqrt();
    let init_bs = crate::Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let bs = vb.get_with_hints(shape![out_dim], "bias", init_bs).await?;
    Ok(Linear::new(ws, Some(bs)))
}

#[maybe_async]
/// Create or initialize a new linear layer without biases.
pub async fn linear_no_bias(
    in_dim: usize,
    out_dim: usize,
    vb: crate::VarBuilder<'_>,
) -> anyhow::Result<Linear> {
    let init_ws = crate::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb
        .get_with_hints(shape![out_dim, in_dim], "weight", init_ws)
        .await?;
    Ok(Linear::new(ws, None))
}

#[maybe_async]
pub async fn linear_b(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: crate::VarBuilder<'_>,
) -> anyhow::Result<Linear> {
    if bias {
        linear(in_dim, out_dim, vb).await
    } else {
        linear_no_bias(in_dim, out_dim, vb).await
    }
}

#[cfg(test)]
mod tests {
    use crate::{Module, VarBuilder, VarMap};

    use super::{linear, linear_no_bias, Linear};
    use ratchet::{
        prelude::shape, test_util::run_py_prg_multiple, Device, DeviceRequest, Tensor, Var,
    };
    use test_strategy::{proptest, Arbitrary};

    thread_local! {
        static GPU_DEVICE: Device = Device::request_device(DeviceRequest::GPU).unwrap();
    }

    fn ground_truth_forward(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> anyhow::Result<Tensor> {
        match b {
            Some(bias) => {
                let prg = r#"
import torch
def linear(x, w, b):
    x_tensor = torch.from_numpy(x)
    w_tensor = torch.from_numpy(w)
    b_tensor = torch.from_numpy(b)
    return (torch.nn.functional.linear(x_tensor, w_tensor, b_tensor).float().numpy(),)
"#;
                let result = run_py_prg_multiple(prg.to_string(), &[x, w, bias], &[])?;
                Ok(result.into_iter().next().unwrap())
            }
            None => {
                let prg = r#"
import torch
def linear(x, w):
    x_tensor = torch.from_numpy(x)
    w_tensor = torch.from_numpy(w)
    return (torch.nn.functional.linear(x_tensor, w_tensor).float().numpy(),)
"#;
                let result = run_py_prg_multiple(prg.to_string(), &[x, w], &[])?;
                Ok(result.into_iter().next().unwrap())
            }
        }
    }

    fn run_linear_forward_trial(
        batch_size: usize,
        in_features: usize,
        out_features: usize,
        with_bias: bool,
    ) -> anyhow::Result<()> {
        let device = GPU_DEVICE.with(|d| d.clone());
        let x = Tensor::randn::<f32>(0., 1., shape![batch_size, in_features], Device::CPU);
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, ratchet::DType::F32, &device.clone());

        let linear = if with_bias {
            linear(in_features, out_features, vb.clone())?
        } else {
            linear_no_bias(in_features, out_features, vb.clone())?
        };

        let w = &linear.w;
        let b = linear.b.clone();

        let ground = ground_truth_forward(&x, w, b.as_ref())?;

        let x_gpu = x.to(&device)?;
        let linear_gpu = Linear::new(
            linear.w.to(&device)?,
            linear.b.map(|b| b.to(&device)).transpose()?,
        );
        let result_gpu = linear_gpu.schedule(x_gpu)?;

        let ours = result_gpu.to(&Device::CPU)?;
        println!("x = {:?}", x);
        println!("w = {:?}", w);
        println!("b = {:?}", b);
        println!("ours = {:?}", ours);
        println!("ground = {:?}", ground);
        ground.all_close(&ours, 1e-4, 1e-4)?;
        Ok(())
    }

    #[derive(Arbitrary, Debug)]
    struct LinearProblem {
        #[strategy(1..=64usize)]
        batch_size: usize,
        #[strategy(1..=64usize)]
        in_features: usize,
        #[strategy(1..=64usize)]
        out_features: usize,
        with_bias: bool,
    }

    #[proptest(cases = 32)]
    fn test_linear_forward(prob: LinearProblem) {
        let LinearProblem {
            batch_size,
            in_features,
            out_features,
            with_bias,
        } = prob;
        println!(
            "batch_size = {}, in_features = {}, out_features = {}, with_bias = {}",
            batch_size, in_features, out_features, with_bias
        );
        run_linear_forward_trial(batch_size, in_features, out_features, with_bias).unwrap();
    }

    fn ground_truth_backward(
        x: &Tensor,
        w: &Tensor,
        b: Option<&Tensor>,
    ) -> anyhow::Result<(Tensor, Tensor, Option<Tensor>)> {
        match b {
            Some(bias) => {
                let prg = r#"
import torch
def linear_backward(x, w, b):
    x_tensor = torch.tensor(torch.from_numpy(x), requires_grad=True)
    w_tensor = torch.tensor(torch.from_numpy(w), requires_grad=True)
    b_tensor = torch.tensor(torch.from_numpy(b), requires_grad=True)
    result = torch.nn.functional.linear(x_tensor, w_tensor, b_tensor)
    result.backward(torch.ones_like(result))
    return x_tensor.grad.numpy(), w_tensor.grad.numpy(), b_tensor.grad.numpy()
"#;
                let result = run_py_prg_multiple(prg.to_string(), &[x, w, bias], &[])?;
                Ok((
                    result[0].clone(),
                    result[1].clone(),
                    Some(result[2].clone()),
                ))
            }
            None => {
                let prg = r#"
import torch
def linear_backward(x, w):
    x_tensor = torch.tensor(torch.from_numpy(x), requires_grad=True)
    w_tensor = torch.tensor(torch.from_numpy(w), requires_grad=True)
    result = torch.nn.functional.linear(x_tensor, w_tensor)
    result.backward(torch.ones_like(result))
    return x_tensor.grad.numpy(), w_tensor.grad.numpy()
"#;
                let result = run_py_prg_multiple(prg.to_string(), &[x, w], &[])?;
                Ok((result[0].clone(), result[1].clone(), None))
            }
        }
    }

    fn run_linear_backward_trial(problem: LinearProblem) -> anyhow::Result<()> {
        let device = GPU_DEVICE.with(|d| d.clone());
        let LinearProblem {
            batch_size,
            in_features,
            out_features,
            with_bias,
        } = problem;

        let x = Tensor::randn::<f32>(0., 1., shape![batch_size, in_features], Device::CPU);
        let x_gpu = x.to(&device)?;
        let x_var = Var::from_tensor(&x_gpu)?;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, ratchet::DType::F32, &device.clone());

        let linear = if with_bias {
            linear(in_features, out_features, vb.clone())?
        } else {
            linear_no_bias(in_features, out_features, vb.clone())?
        };

        let w = &linear.w;
        let b = linear.b.as_ref();
        let w_cpu = w.to(&Device::CPU)?;
        let b_cpu = b.map(|b| b.to(&Device::CPU).unwrap());

        let (ground_x_grad, ground_w_grad, ground_b_grad) =
            ground_truth_backward(&x, &w_cpu, b_cpu.as_ref())?;

        let result_gpu = linear.schedule(x_var.as_tensor().clone())?;

        let grads = result_gpu.backward()?;
        device.try_gpu()?.mark_step()?;

        let x_grad = grads.get(x_var.as_tensor()).unwrap().to(&Device::CPU)?;
        let w_grad = grads.get(&linear.w).unwrap().to(&Device::CPU)?;
        let b_grad = match &linear.b {
            Some(b) => Some(grads.get(b).unwrap().to(&Device::CPU)?),
            None => None,
        };

        println!("x_grad (ours) = {:?}", x_grad);
        println!("x_grad (ground) = {:?}", ground_x_grad);
        println!("w_grad (ours) = {:?}", w_grad);
        println!("w_grad (ground) = {:?}", ground_w_grad);
        if let Some(b_grad) = &b_grad {
            println!("b_grad (ours) = {:?}", b_grad);
            println!("b_grad (ground) = {:?}", ground_b_grad.as_ref().unwrap());
        }

        ground_x_grad.all_close(&x_grad, 1e-4, 1e-4)?;
        ground_w_grad.all_close(&w_grad, 1e-4, 1e-4)?;
        if let Some(b_grad) = &b_grad {
            ground_b_grad
                .as_ref()
                .unwrap()
                .all_close(b_grad, 1e-4, 1e-4)?;
        }

        Ok(())
    }

    #[proptest(cases = 16)]
    fn test_linear_backward(prob: LinearProblem) {
        let LinearProblem {
            batch_size,
            in_features,
            out_features,
            with_bias,
        } = prob;
        println!(
            "batch_size = {}, in_features = {}, out_features = {}, with_bias = {}",
            batch_size, in_features, out_features, with_bias
        );
        run_linear_backward_trial(prob).unwrap();
    }
}
