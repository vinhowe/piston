use piston::{Device, Parameter, Tensor};

// This seems to work, but it is very slow. Norming and adding all the gradients is probaby quite slow;
// there are probably some easy wins here, like maybe a fused norm op?
pub fn clip_grad_norm(
    vars: Vec<Parameter>,
    max_norm: f32,
    device: &Device,
) -> anyhow::Result<Tensor> {
    let mut total_norm = Tensor::full(1, 0., device)?;
    let mut any_grads = false;

    for var in vars.iter() {
        total_norm = (total_norm + var.as_tensor().grad().unwrap().norm()?)?;
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

    for var in vars.iter() {
        var.as_tensor()
            .set_grad(var.as_tensor().grad().unwrap().mul(clip_coef.clone())?);
    }

    Ok(total_norm)
}
