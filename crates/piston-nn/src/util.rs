use piston::{Device, GradStore, Tensor};

// This seems to work, but it is very slow. Norming and adding all the gradients is probaby quite slow;
// there are probably some easy wins here, like maybe a fused norm op?
pub fn clip_grad_norm(
    grads: &mut GradStore,
    max_norm: f32,
    device: &Device,
) -> anyhow::Result<Tensor> {
    let mut total_norm = Tensor::full(1, 0., device)?;
    let mut any_grads = false;

    for (_, grad) in grads.iter() {
        total_norm = (total_norm + grad.clone().norm()?)?;
        any_grads = true;
    }

    if !any_grads {
        return Ok(total_norm);
    }

    let clip_coef = (max_norm / (total_norm.clone() + 1e-6)?)?;
    let ones_max = Tensor::ones::<f32, _>(1, device)?;
    let clip_coef = (clip_coef.clone().lt(ones_max.clone()))?
        .where_cond(clip_coef.clone(), ones_max)?
        .detach();

    for (_, grad) in grads.iter_mut() {
        *grad = grad.clone().mul(clip_coef.clone())?.detach();
    }

    Ok(total_norm)
}
