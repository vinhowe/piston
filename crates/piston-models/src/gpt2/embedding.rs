use maybe_async::maybe_async;
use piston_nn::{Embedding, Init, VarBuilder};

#[maybe_async]
pub async fn embedding_gpt2(
    in_size: usize,
    out_size: usize,
    vb: VarBuilder<'_>,
) -> anyhow::Result<Embedding> {
    let embeddings = vb
        .get_with_hints(
            (in_size, out_size),
            "weight",
            Init::Randn {
                mean: 0.,
                stdev: 0.02,
            },
        )
        .await?;
    Ok(Embedding::new(embeddings, out_size))
}
