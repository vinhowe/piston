use crate::Module;
use ratchet::{shape, Shape, Tensor};

/// # Embedding
///
/// Standard `torch.nn.Embedding` module.
#[derive(Debug, derive_new::new)]
pub struct Embedding {
    pub weight: Tensor,
    hidden_size: usize,
}

impl Embedding {
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Get the hidden size of the embedding matrix
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

impl Module for Embedding {
    type Input = Tensor;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let mut final_dims = input.shape().to_vec();
        final_dims.push(self.hidden_size);
        let indexes = input.flatten_all()?;
        let values = self.weight.clone().index_select(indexes.clone(), 0)?;
        let values = values.view(Shape::from(final_dims))?;
        Ok(values)
    }
}

#[cfg(target_arch = "wasm32")]
pub async fn embedding(
    in_size: usize,
    out_size: usize,
    vb: crate::VarBuilder<'_>,
) -> anyhow::Result<Embedding> {
    let embeddings = vb
        .get_with_hints(
            shape![in_size, out_size],
            "weight",
            crate::Init::Randn {
                mean: 0.,
                stdev: 1.,
            },
        )
        .await?;
    Ok(Embedding::new(embeddings, out_size))
}

#[cfg(not(target_arch = "wasm32"))]
pub fn embedding(
    in_size: usize,
    out_size: usize,
    vb: crate::VarBuilder,
) -> anyhow::Result<Embedding> {
    let embeddings = vb.get_with_hints(
        shape![in_size, out_size],
        "weight",
        crate::Init::Randn {
            mean: 0.,
            stdev: 1.,
        },
    )?;
    Ok(Embedding::new(embeddings, out_size))
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use hf_hub::api::sync::Api;
    use proptest::arbitrary::Arbitrary;
    use proptest::strategy::{BoxedStrategy, Just, Strategy};
    use ratchet_loader::gguf::gguf::Header;
    use test_strategy::proptest;
    use tokenizers::Tokenizer;

    use ratchet::test_util::run_py_prg;
    use ratchet::{rvec, shape, Device, DeviceRequest, Shape, Tensor};

    use crate::{Embedding, Module};

    impl Arbitrary for EmbeddingProblem {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            {
                let args = vec![1..512usize, 1..16usize];
                args.prop_map(Into::<Shape>::into).boxed()
            }
            .prop_flat_map(|vocab_shape| (Just(vocab_shape), 1..64usize))
            .prop_map(|(vocab_shape, num_indices)| {
                let indices =
                    Tensor::randint(0, vocab_shape[0] as i32, shape![num_indices], Device::CPU);
                EmbeddingProblem {
                    vocab_shape,
                    indices,
                }
            })
            .boxed()
        }
    }

    fn ground_truth(weight: &Tensor, indices: &Tensor) -> anyhow::Result<Tensor> {
        let arg = "torch.from_numpy(weight)";

        let prg = format!(
            r#"
import torch
def embedding(weight, indices):
    embedding = torch.nn.Embedding.from_pretrained({})
    return embedding(torch.from_numpy(indices)).numpy()
"#,
            arg
        );
        run_py_prg(prg.to_string(), &[weight, indices], &[], weight.dt())
    }

    fn run_embedding_trial(problem: EmbeddingProblem) {
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        println!("Embedding problem: {:?}", problem);
        let EmbeddingProblem {
            vocab_shape,
            indices,
        } = problem;
        let weight = Tensor::randn::<f32>(vocab_shape, Device::CPU);

        let ground_truth = ground_truth(&weight, &indices).unwrap();

        let weight = weight.to(&device).unwrap();
        let indices = indices.to(&device).unwrap();

        let embedding = Embedding::new(weight);
        let result = embedding.schedule(indices).unwrap().resolve().unwrap();
        let x = result.to(&Device::CPU).unwrap();
        ground_truth.all_close(&x, 1e-6, 1e-6).unwrap();
    }

    #[derive(Debug, Clone)]
    struct EmbeddingProblem {
        vocab_shape: Shape,
        indices: Tensor,
    }

    #[test]
    fn debug_embedding() {
        let prob = EmbeddingProblem {
            vocab_shape: shape![10000, 384],
            indices: Tensor::from_data([400i32, 9001i32, 5555i32], shape![1, 3], Device::CPU),
        };
        run_embedding_trial(prob);
    }

    #[proptest(cases = 16)]
    fn test_embedding(prob: EmbeddingProblem) {
        run_embedding_trial(prob);
    }
}
