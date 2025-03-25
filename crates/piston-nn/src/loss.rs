use piston::{DType, ScopePusher, Tensor};

pub fn nll(inp: Tensor, target: Tensor) -> anyhow::Result<Tensor> {
    let _scope_guard = ScopePusher::new("loss:nll");
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
        .affine(-1f32 / b_sz as f32, 0.)
}

pub fn nll_masked(inp: Tensor, target: Tensor) -> anyhow::Result<Tensor> {
    let _scope_guard = ScopePusher::new("loss:nll_masked");
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

    // We do a bunch of work here to allow ignoring tokens in the target.
    let ignore_index = -100;
    let mask = target.clone().ne(Tensor::full(
        target.shape(),
        ignore_index,
        target.device(),
        false,
    )?)?;

    // Note here that we seem to be able to get away with passing negative indices to gather.
    // If we were more careful about this, we'd replace the indices with 0s where the mask is 0,
    // before passing them to gather.

    let per_sample_loss = inp
        .gather(target.clone().unsqueeze(1)?, 1)?
        .affine(-1f32, 0.)?;
    let mask_unsqueezed = mask.clone().unsqueeze(1)?;
    let masked_loss = per_sample_loss
        .clone()
        .mul(mask_unsqueezed.cast(DType::F32)?)?;

    let valid_count = mask.cast(DType::F32)?.sum_all()?;

    masked_loss.div(valid_count.cast(DType::F32)?)
}

/// Computes label-smoothed cross-entropy for a flattened `[batch_size, vocab_size]` log-softmax
/// tensor `log_probs`, together with a corresponding `target` of shape `[batch_size]`.
///
/// `alpha` is the smoothing parameter in `[0, 1]`. If `alpha == 0.0`, this is just ordinary NLL.
pub fn label_smoothed_nll(log_probs: Tensor, target: Tensor, alpha: f32) -> anyhow::Result<Tensor> {
    let _scope_guard = ScopePusher::new("loss:nll_label_smoothed");
    let b_sz = match &target.shape().to_vec()[..] {
        &[b_sz] => b_sz,
        dims => {
            anyhow::bail!("label_smoothed_nll: target must be [batch_size], got shape {dims:?}")
        }
    };
    let shape_lp = log_probs.shape().to_vec();
    if shape_lp.len() != 2 {
        anyhow::bail!(
            "label_smoothed_nll: log_probs must be rank-2 [batch_size, vocab_size], got {shape_lp:?}"
        );
    }
    let (inp_b_sz, vocab_size) = (shape_lp[0], shape_lp[1]);
    if inp_b_sz != b_sz {
        anyhow::bail!(
            "label_smoothed_nll: batch size mismatch between log_probs ({inp_b_sz}) and target ({b_sz})"
        );
    }

    // Check for ignored tokens (often `-100` in NLP).
    let ignore_index = -100;
    let mask = target
        .clone()
        .ne(Tensor::full(
            target.shape(),
            ignore_index,
            target.device(),
            false,
        )?)?
        .cast(piston::DType::F32)?;

    // Gather the negative log-prob for the correct class:
    // nll_loss[i] = -log_probs[i, target[i]]  (for each token i).
    let nll_gathered = log_probs
        .clone()
        .gather(target.clone().unsqueeze(1)?, 1)? // shape [batch_size, 1]
        .affine(-1.0, 0.0)?; // multiply by -1

    // Mask out ignored tokens (multiply by 0 where masked=0).
    let nll_masked = nll_gathered.mul(mask.clone().unsqueeze(1)?)?;

    // We'll also average over only the valid tokens:
    let valid_count = mask.clone().sum_all()?; // shape []

    // (1) Ordinary cross-entropy term (averaged).
    let nll_loss = nll_masked.div(valid_count.clone())?;

    // (2) Uniform penalty term, also masked.
    //
    // For label smoothing, we pretend a small fraction α of the time
    // we want the “average” log-prob over all classes, not just the correct one.
    // i.e. uniform_loss = - average over (vocab_size) of log_probs, restricted to masked positions.
    let all_probs_masked = log_probs.mul(mask.unsqueeze(1)?)?;
    let sum_log_probs = all_probs_masked.sum(1)?;
    // Now shape [batch_size], each entry is sum_{v in vocab} log_probs[i, v].
    // Negative average per token:
    let neg_avg_log_prob = sum_log_probs.affine(-1.0 / vocab_size as f32, 0.0)?;
    let uniform_loss = neg_avg_log_prob.sum_all()?.div(valid_count)?;

    // Combine with alpha
    // final = (1 - alpha) * nll + alpha * uniform_term
    let final_loss = nll_loss
        .affine(1.0 - alpha, 0.0)?
        .add(uniform_loss.affine(alpha, 0.0)?)?;

    Ok(final_loss)
}

pub fn log_softmax(xs: Tensor, d: usize) -> anyhow::Result<Tensor> {
    let _scope_guard = ScopePusher::new("loss:log_softmax");
    let max = xs.clone().max_keepdim(d)?;
    let diff = xs.clone().sub(max)?;
    let sum_exp = diff.clone().exp()?.sum_keepdim(d)?;
    let log_sm = diff.sub(sum_exp.log()?)?;
    Ok(log_sm)
}

pub fn cross_entropy(inp: Tensor, target: Tensor) -> anyhow::Result<Tensor> {
    let _scope_guard = ScopePusher::new("loss:cross_entropy");
    if inp.rank() != 2 {
        anyhow::bail!("cross_entropy expects an input tensor of rank 2")
    }
    let inp = log_softmax(inp, 1)?;
    nll(inp, target)?.sum_all()
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use super::*;
    use anyhow::Result;
    use piston::{DType, Device, DeviceRequest, Parameter};
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
        let grad =
            piston::test_util::run_py_prg(grad_prg.to_string(), &[input, target], &[], DType::F32)?;

        let loss_prg = r#"
import torch
import torch.nn.functional as F

def cross_entropy(input, target):
    input_tensor = torch.tensor(torch.from_numpy(input))
    target_tensor = torch.tensor(torch.from_numpy(target), dtype=torch.long)
    return F.cross_entropy(input_tensor, target_tensor, reduction='mean').numpy()
"#;
        let loss =
            piston::test_util::run_py_prg(loss_prg.to_string(), &[input, target], &[], DType::F32)?;

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
        let input = Tensor::randn::<f32, _>(0., 1., (batch_size, num_classes), Device::CPU)?;
        let target = Tensor::randint(0, num_classes as i32, batch_size, Device::CPU)?;

        // Compute ground truth
        let (ground_loss, ground_grad) = ground_truth_cross_entropy(&input, &target)?;

        // Compute our implementation
        let input_gpu = input.to(&device)?;
        let target_gpu = target.to(&device)?;
        let input_param = Parameter::from_tensor(&input_gpu)?;
        let our_loss = cross_entropy(input_param.as_tensor().clone(), target_gpu)?;

        // Compute gradients
        our_loss.backward()?;
        device.try_gpu()?.mark_step()?;
        let our_grad = input_param.as_tensor().grad().unwrap().clone();

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
