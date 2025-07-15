use std::cell::RefCell;

use piston::{Device, DeviceRequest};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = Device)]
#[derive(Clone)]
pub struct JsDevice {
    #[wasm_bindgen(skip)]
    pub(crate) inner: Device,
}

#[wasm_bindgen(js_class = Device)]
impl JsDevice {
    // This is used so we can keep a global device instance in the js lib without it being moved
    // into various methods. Probably not the best way to do this.
    #[wasm_bindgen(js_name = _clone)]
    pub fn _clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }

    #[wasm_bindgen(js_name = markStep)]
    pub async fn mark_step(&self) -> Result<(), JsValue> {
        self.inner
            .try_gpu()
            .unwrap()
            .mark_step()
            .await
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        match self.inner {
            Device::CPU => "cpu".to_string(),
            Device::GPU(_) => "webgpu".to_string(),
        }
    }
}

thread_local! {
    pub static GPU_DEVICE: RefCell<Option<JsDevice>> = const { RefCell::new(None) };
}

pub async fn gpu() -> Result<JsDevice, JsValue> {
    // First check if the device is already initialized
    let maybe_device = GPU_DEVICE.with(|refcell| refcell.borrow().clone());

    if let Some(device) = maybe_device {
        return Ok(device);
    }

    // Otherwise, initialize it
    let device = Device::request_device(DeviceRequest::GPU)
        .await
        .map_err(|e| e.to_string())?;

    let js_device = JsDevice { inner: device };
    GPU_DEVICE.with(|refcell| refcell.borrow_mut().replace(js_device.clone()));

    Ok(js_device)
}

pub fn gpu_sync() -> Result<JsDevice, JsValue> {
    GPU_DEVICE
        .with(|refcell| refcell.borrow().clone())
        .ok_or(JsValue::from_str("GPU device not initialized"))
}

pub async fn cpu() -> JsDevice {
    JsDevice { inner: Device::CPU }
}

#[wasm_bindgen(js_name = gpu)]
pub async fn gpu_wasm() -> Result<JsDevice, JsValue> {
    gpu().await
}

#[wasm_bindgen(js_name = cpu)]
pub async fn cpu_wasm() -> Result<JsDevice, JsValue> {
    Ok(cpu().await)
}

#[wasm_bindgen]
pub fn seed(seed: u64) -> Result<(), JsValue> {
    gpu_sync()?.inner.set_seed(seed);
    Ok(())
}
