use piston::{AllDims, Device, Tensor, TensorOptions, ones};

// This seems to work, but it is very slow. Norming and adding all the gradients is probaby quite slow;
// there are probably some easy wins here, like maybe a fused norm op?
pub fn clip_grad_norm(vars: Vec<Tensor>, max_norm: f32, device: &Device) -> anyhow::Result<Tensor> {
    let mut total_norm = Tensor::full(1, 0., device, false)?;
    let mut any_grads = false;

    for var in vars.iter() {
        total_norm = (total_norm + var.grad().unwrap().norm(None, AllDims, false)?)?;
        any_grads = true;
    }

    if !any_grads {
        return Ok(total_norm);
    }

    let clip_coef = (max_norm / (total_norm.clone() + 1e-6)?)?;
    let ones_max = ones(1, TensorOptions::new().device(device.clone()))?;
    let clip_coef = clip_coef
        .clone()
        .where_cond(clip_coef.clone().lt(ones_max.clone())?, ones_max)?;

    for var in vars.iter() {
        var.set_grad(Some(var.grad().unwrap().mul(clip_coef.clone())?));
    }

    Ok(total_norm)
}
