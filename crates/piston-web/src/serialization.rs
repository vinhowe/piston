use std::sync::Arc;

use crate::error::IntoJsError;
use crate::tensor::JsTensor;
use futures::lock::Mutex;
use js_sys::{Object, Reflect};
use piston::Device;
use piston::OpTensor;
use piston::Tensor;
use wasm_bindgen::JsValue;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use wasm_bindgen_futures::future_to_promise;
use web_sys::Blob;
use web_sys::Url;

/// Saves tensors to a safetensors format and returns a URL for download
///
/// The input is expected to be a JavaScript object mapping tensor ID strings to
/// either JsTensor or JsParameter objects.
#[wasm_bindgen]
pub async fn save(data: JsValue) -> anyhow::Result<String, JsError> {
    // Check if the data is a JavaScript object
    if !data.is_object() {
        return Err(JsError::new(
            "Expected an object mapping tensor IDs to tensors or parameters",
        ));
    }

    // Get object keys
    let obj = Object::from(data);
    let keys = js_sys::Object::keys(&obj);
    let keys_len = keys.length();

    let mut tensors: Vec<(String, Tensor)> = Vec::with_capacity(keys_len as usize);

    for i in 0..keys_len {
        let key = keys.get(i).as_string().unwrap();
        let value = Reflect::get(&obj, &JsValue::from_str(&key)).map_err(|e| e.into_js_error())?;

        // Try to extract tensor from the value
        match JsTensor::try_from(value) {
            Ok(tensor) => tensors.push((key, tensor.inner.clone())),
            Err(e) => {
                return Err(JsError::new(&format!(
                    "Error processing key '{key}': {e:?}"
                )));
            }
        }
    }

    // Make sure we have at least one tensor
    if tensors.is_empty() {
        return Err(JsError::new(
            "No valid tensors found in the provided object",
        ));
    }

    let data = Arc::new(Mutex::new(Vec::new()));
    let mut futures = Vec::new();

    // For each Var, move the Tensor to CPU asynchronously.
    for (k, tensor) in tensors.iter() {
        let k = k.clone();
        let t = tensor.clone();
        let data_ref = Arc::clone(&data);
        futures.push(future_to_promise(async move {
            let t_cpu = t.to(&Device::CPU).await.unwrap();
            data_ref.lock().await.push((k, t_cpu));
            Ok(JsValue::undefined())
        }));
    }

    // Wait for all futures
    let promise_array = js_sys::Array::from_iter(futures.iter());
    JsFuture::from(js_sys::Promise::all(&promise_array))
        .await
        .map_err(|e| e.into_js_error())?;

    // Collect the results
    let data = Arc::try_unwrap(data).unwrap().into_inner();

    // Convert to the format expected by safetensors
    let data_ref: Vec<(&String, OpTensor)> = data
        .iter()
        .map(|(k, v)| (k, v.inner().read().clone()))
        .collect::<Vec<_>>();

    // Serialize to safetensors format
    let serialized = safetensors::tensor::serialize(data_ref.iter().map(|(k, v)| (k, v)), &None)
        .map_err(|e| JsError::new(&format!("Failed to serialize tensors: {e}")))?;

    // Create a Uint8Array from the serialized data
    let uint8_array = js_sys::Uint8Array::from(&serialized[..]);
    let blob = Blob::new_with_u8_array_sequence(&js_sys::Array::of1(&JsValue::from(uint8_array)))
        .map_err(|e| e.into_js_error())?;

    // Create a download URL
    let url = Url::create_object_url_with_blob(&blob).map_err(|e| e.into_js_error())?;

    Ok(url)
}
