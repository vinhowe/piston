use crate::db::*;
use futures::stream::TryStreamExt;
use futures::StreamExt;
use ratchet_hub::{Api, ApiBuilder, RepoType};
use ratchet_loader::gguf::gguf::{self, Header, TensorInfo};
use ratchet_models::moondream::{self, Moondream};
use ratchet_models::phi2;
use ratchet_models::phi2::Phi2;
use ratchet_models::phi3::{self, Phi3};
use ratchet_models::registry::{AvailableModels, PhiVariants, Quantization};
use ratchet_models::TensorMap;
use tokenizers::Tokenizer;
use wasm_bindgen::prelude::*;

#[derive(Debug)]
pub enum WebModel {
    Phi2(Phi2),
    Phi3(Phi3),
    Moondream(Moondream),
}

impl WebModel {
    pub async fn run(&mut self, input: JsValue) -> Result<JsValue, JsValue> {
        match self {
            WebModel::Phi2(model) => {
                let input: PhiInputs = serde_wasm_bindgen::from_value(input)?;
                let rs_callback = |output: String| {
                    let _ = input.callback.call1(&JsValue::NULL, &output.into());
                };
                let prompt = input.prompt;

                let model_repo = ApiBuilder::from_hf("microsoft/phi-2", RepoType::Model).build();
                let model_bytes = model_repo.get("tokenizer.json").await?;
                let tokenizer = Tokenizer::from_bytes(model_bytes.to_vec()).unwrap();
                phi2::generate(model, tokenizer, prompt, rs_callback)
                    .await
                    .unwrap();
                Ok(JsValue::NULL)
            }
            WebModel::Phi3(model) => {
                let input: PhiInputs = serde_wasm_bindgen::from_value(input)?;
                let rs_callback = |output: String| {
                    let _ = input.callback.call1(&JsValue::NULL, &output.into());
                };
                let prompt = input.prompt;

                let model_repo =
                    ApiBuilder::from_hf("microsoft/Phi-3-mini-4k-instruct", RepoType::Model)
                        .build();
                let model_bytes = model_repo.get("tokenizer.json").await?;
                let tokenizer = Tokenizer::from_bytes(model_bytes.to_vec()).unwrap();
                phi3::generate(model, tokenizer, prompt, rs_callback)
                    .await
                    .unwrap();
                Ok(JsValue::NULL)
            }
            WebModel::Moondream(model) => {
                let input: MoondreamInputs = serde_wasm_bindgen::from_value(input)?;
                let rs_callback = |output: String| {
                    let _ = input.callback.call1(&JsValue::NULL, &output.into());
                };
                let model_repo =
                    ApiBuilder::from_hf("tgestson/ratchet-moondream2", RepoType::Model).build();
                let model_bytes = model_repo.get("tokenizer.json").await?;
                let tokenizer = Tokenizer::from_bytes(model_bytes.to_vec()).unwrap();
                moondream::generate(
                    model,
                    input.image_bytes,
                    input.question,
                    tokenizer,
                    rs_callback,
                )
                .await
                .unwrap();
                Ok(JsValue::NULL)
            }
        }
    }

    pub async fn from_stored(
        model_record: ModelRecord,
        tensor_map: TensorMap,
    ) -> Result<WebModel, anyhow::Error> {
        let header = serde_wasm_bindgen::from_value::<Header>(model_record.header).unwrap();
        match model_record.model {
            AvailableModels::Phi(variant) => match variant {
                PhiVariants::Phi2 => {
                    let model = Phi2::from_web(header, tensor_map).await?;
                    Ok(WebModel::Phi2(model))
                }
                PhiVariants::Phi3 => {
                    let model = Phi3::from_web(header, tensor_map).await?;
                    Ok(WebModel::Phi3(model))
                }
            },
            AvailableModels::Moondream => {
                let model = Moondream::from_web(header, tensor_map).await?;
                Ok(WebModel::Moondream(model))
            }
            _ => Err(anyhow::anyhow!("Unknown model type")),
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct PhiInputs {
    pub prompt: String,
    #[serde(with = "serde_wasm_bindgen::preserve")]
    pub callback: js_sys::Function,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct MoondreamInputs {
    pub question: String,
    pub image_bytes: Vec<u8>,
    #[serde(with = "serde_wasm_bindgen::preserve")]
    pub callback: js_sys::Function,
}

#[wasm_bindgen]
#[derive(Debug)]
pub struct Model {
    inner: WebModel,
}

#[wasm_bindgen]
impl Model {
    /// The main JS entrypoint into the library.
    ///
    /// Loads a model with the provided ID.
    /// This key should be an enum of supported models.
    #[wasm_bindgen]
    pub async fn load(
        model: AvailableModels,
        quantization: Quantization,
        progress: &js_sys::Function,
    ) -> Result<Model, JsValue> {
        let model_key = ModelKey::from_available(&model, quantization);
        let model_repo = ApiBuilder::from_hf(&model_key.repo_id(), RepoType::Model).build();

        let webModel = Self::load_inner(model, &model_repo, model_key, progress).await?;

        Ok(Model { inner: webModel })
    }

    #[wasm_bindgen]
    pub async fn load_custom(
        endpoint: String,
        model: AvailableModels,
        quantization: Quantization,
        progress: &js_sys::Function,
    ) -> Result<Model, JsValue> {
        let model_key = ModelKey::from_available(&model, quantization);
        let model_repo = ApiBuilder::from_custom(endpoint).build();

        let webModel = Self::load_inner(model, &model_repo, model_key, progress).await?;

        Ok(Model { inner: webModel })
    }

    async fn load_inner(
        model: AvailableModels,
        model_repo: &Api,
        model_key: ModelKey,
        progress: &js_sys::Function,
    ) -> Result<WebModel, JsValue> {
        let db = RatchetDB::open().await.map_err(|e| {
            let e: JsError = e.into();
            Into::<JsValue>::into(e)
        })?;

        log::warn!("Loading model: {:?}", model_key);
        if let None = db.get_model(&model_key).await.map_err(|e| {
            let e: JsError = e.into();
            Into::<JsValue>::into(e)
        })? {
            let header: gguf::Header = serde_wasm_bindgen::from_value(
                model_repo.fetch_gguf_header(&model_key.model_id()).await?,
            )?;
            Self::fetch_tensors(&db, &model_repo, &header, model_key.clone(), progress).await?;
            let model_record = ModelRecord::new(model_key.clone(), model.clone(), header);
            db.put_model(&model_key, model_record).await.map_err(|e| {
                let e: JsError = e.into();
                Into::<JsValue>::into(e)
            })?;
        };

        let model_record = db.get_model(&model_key).await.unwrap().unwrap();
        let tensors = db.get_tensors(&model_key).await.unwrap();

        Ok(WebModel::from_stored(model_record, tensors).await.unwrap())
    }

    /// User-facing method to run the model.
    ///
    /// Untyped input is required unfortunately.
    pub async fn run(&mut self, input: JsValue) -> Result<JsValue, JsValue> {
        self.inner.run(input).await
    }

    async fn fetch_tensors(
        db: &RatchetDB,
        model_repo: &Api,
        header: &Header,
        model_key: ModelKey,
        progress: &js_sys::Function,
    ) -> Result<(), JsValue> {
        let model_id = model_key.model_id();
        let data_offset = header.tensor_data_offset;
        let content_len = header
            .tensor_infos
            .values()
            .fold(0, |acc, ti| acc + ti.size_in_bytes());

        let mut tensor_infos: Vec<(String, TensorInfo)> =
            header.tensor_infos.clone().into_iter().collect();
        tensor_infos.sort_by(|(_, a), (_, b)| b.size_in_bytes().cmp(&a.size_in_bytes()));

        let tensor_stream = futures::stream::iter(tensor_infos);

        let mut total_progress = 0.0;

        tensor_stream
            .map(|(name, info): (String, TensorInfo)| {
                let model_id = model_id.clone();
                let model_key = model_key.clone();
                async move {
                    let range = info.byte_range(data_offset);
                    let bytes = model_repo
                        .fetch_range(&model_id, range.start, range.end)
                        .await
                        .unwrap();
                    let length = bytes.length();
                    let record =
                        TensorRecord::new(name.clone().to_string(), model_key.clone(), bytes);
                    db.put_tensor(record).await.map_err(|e| {
                        let e: JsError = e.into();
                        Into::<JsValue>::into(e)
                    });
                    length
                }
            })
            .buffer_unordered(6)
            .map(|num_bytes| {
                let req_progress = (num_bytes as f64) / (content_len as f64) * 100.0;
                total_progress += req_progress;
                let _ = progress.call1(&JsValue::NULL, &total_progress.into());
            })
            .collect::<()>()
            .await;

        Ok(())
    }
}