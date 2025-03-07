use crate::{Device, Tensor};
use maybe_async::maybe_async;

#[maybe_async]
pub async fn to_vec0_round(t: &Tensor, digits: i32) -> anyhow::Result<f32> {
    let b = 10f32.powi(digits);
    let t = t.to(&Device::CPU).await?.to_vec::<f32>().await?[0];
    Ok(f32::round(t * b) / b)
}

#[maybe_async]
pub async fn to_vec1_round(t: &Tensor, digits: i32) -> anyhow::Result<Vec<f32>> {
    let b = 10f32.powi(digits);
    let t = t.to(&Device::CPU).await?.to_vec::<f32>().await?;
    let t = t.iter().map(|t| f32::round(t * b) / b).collect();
    Ok(t)
}
