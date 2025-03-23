use piston::HashMap;
use std::sync::{Arc, Mutex};

use piston::{Device, Parameter, Shape, Tensor};

#[cfg(target_arch = "wasm32")]
use {
    wasm_bindgen::JsValue,
    wasm_bindgen_futures::future_to_promise,
    wasm_bindgen_futures::js_sys,
    wasm_bindgen_futures::JsFuture,
    web_sys::{Blob, Url},
};

/// A `VarMap` is a store that holds named variables. Variables can be retrieved from the stores
/// and new variables can be added by providing some initialization config in case they are
/// missing.
/// `VarMap` structures can be serialized in the safetensors format.
#[derive(Clone)]
pub struct VarMap {
    data: Arc<Mutex<HashMap<String, Parameter>>>,
}

impl VarMap {
    /// Create a new empty `VarMap`.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        let data = Arc::new(Mutex::new(HashMap::default()));
        Self { data }
    }

    /// Retrieve all the variables currently stored in the map.
    pub fn all_vars(&self) -> Vec<Parameter> {
        let tensor_data = self.data.lock().unwrap();
        #[allow(clippy::map_clone)]
        tensor_data.values().map(|c| c.clone()).collect::<Vec<_>>()
    }

    pub fn all_labeled_vars(&self) -> Vec<(String, Parameter)> {
        let tensor_data = self.data.lock().unwrap();
        tensor_data
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Save the map in the safetensors format.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> anyhow::Result<()> {
        let tensor_data = self.data.lock().unwrap();
        let data = tensor_data.iter().map(|(k, v)| (k, v.as_tensor()));
        safetensors::tensor::serialize_to_file(data, &None, path.as_ref())?;
        Ok(())
    }

    #[cfg(target_arch = "wasm32")]
    async fn create_safetensors_download_url(
        &self,
        data: Vec<(String, Tensor)>,
    ) -> anyhow::Result<String, JsValue> {
        // Convert (String, Tensor) -> (&String, &Tensor) for serialization
        let data_ref: Vec<(&String, &Tensor)> = data.iter().map(|(k, v)| (k, v)).collect();

        // Safetensors serialization
        let serialized = safetensors::tensor::serialize(data_ref, &None).unwrap();

        // Create a Blob from the serialized data
        let uint8_array = js_sys::Uint8Array::from(&serialized[..]);
        let blob =
            Blob::new_with_u8_array_sequence(&js_sys::Array::of1(&JsValue::from(uint8_array)))?;

        // Create a URL for the Blob
        let url = Url::create_object_url_with_blob(&blob)?;

        Ok(url)
    }

    /// Download the variables (instead of grads) as a safetensors file using web APIs.
    /// This avoids logic duplication by calling the same `_download_safetensors` helper.
    #[cfg(target_arch = "wasm32")]
    pub async fn download_url(&self) -> anyhow::Result<String, JsValue> {
        let tensor_data = self.data.lock().unwrap();
        let data = Arc::new(Mutex::new(Vec::new()));
        let mut futures = Vec::new();

        // For each Var, move the Tensor to CPU asynchronously.
        for (k, var) in tensor_data.iter() {
            let k = k.clone();
            let t = var.as_tensor().clone();
            let data_ref = Arc::clone(&data);
            futures.push(future_to_promise(async move {
                let t_cpu = t.to(&Device::CPU).await.unwrap();
                data_ref.lock().unwrap().push((k, t_cpu));
                Ok(JsValue::undefined())
            }));
        }

        // Wait for all futures
        let promise_array = js_sys::Array::from_iter(futures.iter());
        JsFuture::from(js_sys::Promise::all(&promise_array)).await?;

        // Collect the results
        let data = Arc::try_unwrap(data).unwrap().into_inner().unwrap();

        // Use the shared helper
        self.create_safetensors_download_url(data).await
    }

    /// Set a named variable to some value.
    pub fn set_one<K: AsRef<str>>(&mut self, name: K, value: Tensor) -> anyhow::Result<()> {
        let tensor_data = self.data.lock().unwrap();
        let name = name.as_ref();
        match tensor_data.get(name) {
            // None => candle::bail!("cannot find {name} in VarMap"),
            None => panic!("cannot find {name} in VarMap"),
            // Some(var) => {
            //     if let Err(err) = var.set(value.as_ref()) {
            //         candle::bail!("error setting {name}: {err}",)
            //     }
            // }
            Some(var) => {
                if let Err(err) = var.set_sync(value.clone()) {
                    panic!("error setting {name}: {err}")
                }
            }
        }
        Ok(())
    }

    /// Set some named variables to some values.
    ///
    /// If an error is returned, some of the variables might have already been set to their new
    /// values.
    pub fn set<I: Iterator<Item = (K, V)>, K: AsRef<str>, V: AsRef<Tensor>>(
        &mut self,
        iter: I,
    ) -> anyhow::Result<()> {
        let tensor_data = self.data.lock().unwrap();
        for (name, value) in iter {
            let name = name.as_ref();
            match tensor_data.get(name) {
                // None => candle::bail!("cannot find {name} in VarMap"),
                None => panic!("cannot find {name} in VarMap"),
                // Some(var) => {
                //     if let Err(err) = var.set(value.as_ref()) {
                //         candle::bail!("error setting {name}: {err}",)
                //     }
                // }
                Some(var) => {
                    if let Err(err) = var.set_sync(value.as_ref().clone()) {
                        panic!("error setting {name}: {err}")
                    }
                }
            }
        }
        Ok(())
    }

    /// Retrieve or add a new variable.
    pub fn get(
        &self,
        shape: Shape,
        path: &str,
        init: crate::Init,
        device: Device,
    ) -> anyhow::Result<Tensor> {
        let mut tensor_data = self.data.lock().unwrap();
        if let Some(tensor) = tensor_data.get(path) {
            let tensor_shape = tensor.as_tensor().shape();
            if &shape != tensor_shape {
                // candle::bail!("shape mismatch on {path}: {shape:?} <> {tensor_shape:?}")
                panic!("shape mismatch on {path}: {shape:?} <> {tensor_shape:?}")
            }
            return Ok(tensor.as_tensor().clone());
        }
        let var = init.var(&shape, device)?;
        let tensor = var.as_tensor().clone();
        tensor_data.insert(path.to_string(), var);
        Ok(tensor)
    }

    pub fn data(&self) -> &Mutex<HashMap<String, Parameter>> {
        &self.data
    }
}
