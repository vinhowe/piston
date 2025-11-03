use std::sync::Arc;

use crate::device::JsDevice;
use crate::error::IntoJsError;
use crate::tensor::JsTensor;
use futures::lock::Mutex;
use js_sys::{Object, Reflect};
use piston::{DType, Device, OpTensor, Tensor, TensorOptions};
use std::collections::HashMap;
use wasm_bindgen::JsValue;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use wasm_bindgen_futures::future_to_promise;

/// Saves tensors to a safetensors format and returns an ArrayBuffer
///
/// The input is expected to be a JavaScript object mapping tensor ID strings to
/// either JsTensor or JsParameter objects.
#[wasm_bindgen]
pub async fn save(
    data: JsValue,
    #[wasm_bindgen(js_name = configSerialized)] config_serialized: Option<String>,
) -> Result<js_sys::Uint8Array, JsError> {
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
            Ok(tensor) => tensors.push((key, tensor.inner().clone())),
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
    let serialized = safetensors::tensor::serialize(
        data_ref.iter().map(|(k, v)| (k, v)),
        &config_serialized.map(|s| HashMap::from([("piston_extra".to_string(), s)])),
    )
    .map_err(|e| JsError::new(&format!("Failed to serialize tensors: {e}")))?;

    // Create an ArrayBuffer from the serialized data
    Ok(js_sys::Uint8Array::from(&serialized[..]))
}

/// Loads tensors from a safetensors byte buffer and returns a JavaScript object
/// mapping tensor names to Tensor objects.
#[wasm_bindgen(unchecked_return_type = "{ state: Record<string, Tensor>; extra?: unknown }")]
pub fn load(
    bytes: js_sys::Uint8Array,
    #[wasm_bindgen(js_name = mapDevice)] map_device: Option<JsDevice>,
) -> Result<JsValue, JsError> {
    let device = map_device.unwrap_or_else(|| JsDevice { inner: Device::CPU });

    // Copy bytes from the Uint8Array into a Rust Vec<u8>
    let mut buf = vec![0u8; bytes.length() as usize];
    bytes.copy_to(&mut buf[..]);

    // Deserialize safetensors from bytes
    let st = safetensors::SafeTensors::deserialize(&buf)
        .map_err(|e| JsError::new(&format!("Failed to deserialize safetensors: {e}")))?;

    let obj = js_sys::Object::new();

    for name in st.names() {
        let view = st
            .tensor(name)
            .map_err(|e| JsError::new(&format!("Failed to read tensor '{name}': {e}")))?;

        // Map safetensors dtype to piston dtype
        let dtype = match view.dtype() {
            safetensors::Dtype::F32 => DType::F32,
            safetensors::Dtype::F16 => DType::F16,
            safetensors::Dtype::BF16 => DType::BF16,
            safetensors::Dtype::I32 => DType::I32,
            safetensors::Dtype::U32 => DType::U32,
            other => {
                return Err(JsError::new(&format!(
                    "Unsupported dtype in safetensors: {:?}",
                    other
                )));
            }
        };

        let shape = view.shape();
        let data = view.data();

        let tensor = Tensor::from_bytes(
            data,
            shape,
            TensorOptions::new()
                .dtype(dtype)
                .device(device.inner.clone())
                .requires_grad(false),
        )
        .map_err(|e| JsError::new(&format!("Failed to construct tensor '{name}': {e}")))?;

        // Wrap as a JS-visible Tensor
        let js_tensor = JsTensor::new(tensor);
        js_sys::Reflect::set(&obj, &JsValue::from_str(name), &JsValue::from(js_tensor))
            .map_err(|e| JsError::new(&format!("Failed to set property '{name}': {e:?}")))?;
    }

    // Build return object: { state, extra }
    let out = js_sys::Object::new();

    // state
    js_sys::Reflect::set(&out, &JsValue::from_str("state"), &obj)
        .map_err(|e| JsError::new(&format!("Failed to set state on return object: {e:?}")))?;

    // extra (parse JSON if present)
    let extra_js = safetensors::SafeTensors::read_metadata(&buf)
        .ok()
        .and_then(|(_, meta)| {
            meta.metadata().as_ref().and_then(|map| {
                map.get("piston_extra")
                    .and_then(|s| js_sys::JSON::parse(s).ok())
            })
        })
        .unwrap_or(JsValue::UNDEFINED);
    js_sys::Reflect::set(&out, &JsValue::from_str("extra"), &extra_js)
        .map_err(|e| JsError::new(&format!("Failed to set extra on return object: {e:?}")))?;

    Ok(out.into())
}
