use maybe_async::maybe_async;
use piston_nn::{Linear, VarBuilder};

#[maybe_async]
pub async fn linear_gpt2(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilder<'_>,
) -> anyhow::Result<Linear> {
    let init_ws = piston_nn::Init::Randn {
        mean: 0.0,
        stdev: 0.02,
    };
    let ws = vb
        .get_with_hints((out_dim, in_dim), "weight", init_ws)
        .await?;
    let init_bs = piston_nn::Init::Const(0.0);
    let bs = vb.get_with_hints(out_dim, "bias", init_bs).await?;
    Ok(Linear::new(ws, Some(bs)))
}

#[maybe_async]
pub async fn linear_no_bias_gpt2(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilder<'_>,
) -> anyhow::Result<Linear> {
    let init_ws = piston_nn::Init::Randn {
        mean: 0.0,
        stdev: 0.02,
    };
    let ws = vb
        .get_with_hints((out_dim, in_dim), "weight", init_ws)
        .await?;
    Ok(Linear::new(ws, None))
}

#[maybe_async]
pub async fn linear_b_gpt2(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: VarBuilder<'_>,
) -> anyhow::Result<Linear> {
    if bias {
        linear_gpt2(in_dim, out_dim, vb).await
    } else {
        linear_no_bias_gpt2(in_dim, out_dim, vb).await
    }
}

#[maybe_async]
pub async fn linear_gpt2_residual(
    in_dim: usize,
    out_dim: usize,
    n_layer: usize,
    vb: VarBuilder<'_>,
) -> anyhow::Result<Linear> {
    let init_ws = piston_nn::Init::Randn {
        mean: 0.0,
        stdev: 0.02 / (2f32 * (n_layer as f32)).sqrt(),
    };
    let ws = vb
        .get_with_hints((out_dim, in_dim), "weight", init_ws)
        .await?;
    let init_bs = piston_nn::Init::Const(0.0);
    let bs = vb.get_with_hints(out_dim, "bias", init_bs).await?;
    Ok(Linear::new(ws, Some(bs)))
}
