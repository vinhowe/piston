use ratchet::Tensor;

pub fn nll(inp: Tensor, target: Tensor) -> anyhow::Result<Tensor> {
    let b_sz = match &target.shape().to_vec()[..] {
        &[b_sz] => b_sz,
        dims => anyhow::bail!("the target tensor should have a single dimension ({dims:?})"),
    };
    match &inp.shape().to_vec()[..] {
        &[inp_b_sz, _] => {
            if inp_b_sz != b_sz {
                anyhow::bail!("batch size mismatch between inp ({inp_b_sz}) and target ({b_sz})")
            }
        }
        dims => anyhow::bail!("the target tensor should have two dimensions ({dims:?})"),
    }
    inp.gather(target.clone().unsqueeze(1)?, 1)?
        .sum_all()?
        .affine(-1f32 / b_sz as f32, 0.)
}

pub fn log_softmax(xs: Tensor, d: usize) -> anyhow::Result<Tensor> {
    let max = xs.clone().max_keepdim(d)?;
    let diff = xs.clone().sub(max)?;
    let sum_exp = diff.clone().exp()?.sum_keepdim(&[d])?;
    let log_sm = diff.sub(sum_exp.log()?)?;
    Ok(log_sm)
}

pub fn cross_entropy(inp: Tensor, target: Tensor) -> anyhow::Result<Tensor> {
    if inp.rank() != 2 {
        anyhow::bail!("cross_entropy expects an input tensor of rank 2")
    }
    let inp = log_softmax(inp, 1)?;
    nll(inp, target)
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use super::*;
    use anyhow::Result;
    use ratchet::{prelude::shape, DType, Device, DeviceRequest, Var};
    use test_strategy::{proptest, Arbitrary};

    thread_local! {
        static GPU_DEVICE: Device = Device::request_device(DeviceRequest::GPU).unwrap();
    }

    fn ground_truth_cross_entropy(input: &Tensor, target: &Tensor) -> Result<(Tensor, Tensor)> {
        let grad_prg = r#"
import torch
import torch.nn.functional as F

def cross_entropy_with_grad(input, target):
    input_tensor = torch.tensor(torch.from_numpy(input), requires_grad=True)
    target_tensor = torch.tensor(torch.from_numpy(target), dtype=torch.long)
    loss = F.cross_entropy(input_tensor, target_tensor, reduction='mean')
    loss.backward()
    return input_tensor.grad.numpy()
"#;
        let grad = ratchet::test_util::run_py_prg(
            grad_prg.to_string(),
            &[input, target],
            &[],
            DType::F32,
        )?;

        let loss_prg = r#"
import torch
import torch.nn.functional as F

def cross_entropy(input, target):
    input_tensor = torch.tensor(torch.from_numpy(input))
    target_tensor = torch.tensor(torch.from_numpy(target), dtype=torch.long)
    return F.cross_entropy(input_tensor, target_tensor, reduction='mean').numpy()
"#;
        let loss = ratchet::test_util::run_py_prg(
            loss_prg.to_string(),
            &[input, target],
            &[],
            DType::F32,
        )?;

        Ok((loss, grad))
    }

    #[derive(Arbitrary, Debug)]
    struct CrossEntropyProblem {
        #[strategy(1..=32usize)]
        batch_size: usize,
        #[strategy(2..=10usize)]
        num_classes: usize,
    }

    fn run_cross_entropy_trial(problem: CrossEntropyProblem) -> Result<()> {
        let device = GPU_DEVICE.with(|d| d.clone());
        let CrossEntropyProblem {
            batch_size,
            num_classes,
        } = problem;

        // Generate random input and target tensors
        let input = Tensor::randn::<f32>(0., 1., shape![batch_size, num_classes], Device::CPU);
        let target = Tensor::randint(0, num_classes as i32, shape![batch_size], Device::CPU);

        // Compute ground truth
        let (ground_loss, ground_grad) = ground_truth_cross_entropy(&input, &target)?;

        // Compute our implementation
        let input_gpu = input.to(&device)?;
        let target_gpu = target.to(&device)?;
        let input_var = Var::from_tensor(&input_gpu)?;
        let our_loss = cross_entropy(input_var.as_tensor().clone(), target_gpu)?;

        // Compute gradients
        let grads = our_loss.backward()?;
        device.try_gpu()?.mark_step()?;
        let our_grad = grads.get(input_var.as_tensor()).unwrap().clone();

        // Compare results
        let our_loss_cpu = our_loss.to(&Device::CPU)?;
        let our_grad_cpu = our_grad.to(&Device::CPU)?;
        let ground_grad = ground_grad.to(&Device::CPU)?;
        let ground_loss = ground_loss.to(&Device::CPU)?;

        println!("Input shape: {:?}", input.shape());
        println!("Target shape: {:?}", target.shape());
        println!("Our loss: {:?}", our_loss_cpu.to_vec::<f32>());
        println!("Ground truth loss: {:?}", ground_loss.to_vec::<f32>());
        println!("Our grad: {:?}", our_grad_cpu.to_vec::<f32>());
        println!("Ground truth grad: {:?}", ground_grad.to_vec::<f32>());
        println!("Our grad shape: {:?}", our_grad_cpu.shape());
        println!("Ground truth grad shape: {:?}", ground_grad.shape());

        ground_loss.all_close(&our_loss_cpu, 1e-5, 1e-5)?;
        ground_grad.all_close(&our_grad_cpu, 1e-5, 1e-5)?;

        Ok(())
    }

    #[proptest(cases = 10)]
    fn test_cross_entropy(prob: CrossEntropyProblem) {
        let CrossEntropyProblem {
            batch_size,
            num_classes,
        } = prob;
        println!(
            "Testing with batch_size = {}, num_classes = {}",
            batch_size, num_classes
        );
        run_cross_entropy_trial(prob).unwrap();
    }
}
