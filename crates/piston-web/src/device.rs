use std::cell::RefCell;

use piston::{Device, DeviceRequest};
use serde::de::Error as DeError;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use wasm_bindgen::prelude::*;

use crate::js_util::downcast_from_ptr;

#[wasm_bindgen(js_name = Device)]
#[derive(Clone)]
pub struct JsDevice {
    #[wasm_bindgen(skip)]
    pub(crate) inner: Device,
}

#[wasm_bindgen(js_class = Device)]
impl JsDevice {
    // Marker function for downcasting from a JS object
    #[wasm_bindgen]
    pub fn __wbg_piston_device() {}

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
        crate::function::handle_mark_step(self)
            .await
            .map_err(|e| e.into())
    }

    #[wasm_bindgen(js_name = beginPass)]
    pub fn begin_pass(&self) {
        self.inner.try_gpu().unwrap().begin_pass(0);
    }

    #[wasm_bindgen(js_name = setSharedObjectAllocationEnabled)]
    pub fn set_shared_object_allocation_enabled(&self, enabled: bool) {
        self.inner
            .try_gpu()
            .unwrap()
            .set_shared_object_allocation_enabled(enabled);
    }

    #[wasm_bindgen(js_name = setCachingEnabled)]
    pub fn set_caching_enabled(&self, enabled: bool) {
        self.inner.try_gpu().unwrap().set_caching_enabled(enabled);
    }

    #[wasm_bindgen(js_name = setInplaceSupport)]
    pub fn set_inplace_support(&self, enabled: bool) {
        self.inner.try_gpu().unwrap().set_inplace_support(enabled);
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        match self.inner {
            Device::CPU => "cpu".to_string(),
            Device::GPU(_) => "webgpu".to_string(),
        }
    }

    #[wasm_bindgen(js_name = usageBytes)]
    pub fn usage_bytes(&self) -> u64 {
        self.inner.try_gpu().unwrap().usage_bytes()
    }

    #[wasm_bindgen(js_name = markUsageBytesStep)]
    pub fn mark_usage_bytes_step(&self) {
        self.inner.try_gpu().unwrap().mark_usage_bytes_step();
    }

    #[wasm_bindgen(js_name = peakUsageBytes)]
    pub fn peak_usage_bytes(&self) -> u64 {
        self.inner.try_gpu().unwrap().peak_usage_bytes_since_reset()
    }

    #[wasm_bindgen(js_name = setVRAMLimit)]
    pub fn set_vram_limit(&self, #[wasm_bindgen(js_name = vramLimit)] vram_limit: Option<u64>) {
        self.inner.try_gpu().unwrap().set_vram_limit(vram_limit);
    }

    #[wasm_bindgen(js_name = asWebGPUDevice)]
    pub fn as_webgpu_device(&self) -> Option<web_sys::GpuDevice> {
        match &self.inner {
            Device::GPU(gpu) => Some(
                gpu.as_webgpu_device()
                    .dyn_into::<web_sys::GpuDevice>()
                    .unwrap(),
            ),
            Device::CPU => None,
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

pub fn gpu_sync() -> Result<JsDevice, JsError> {
    GPU_DEVICE
        .with(|refcell| refcell.borrow().clone())
        .ok_or(JsError::new("GPU device not initialized"))
}

pub fn cpu() -> JsDevice {
    JsDevice { inner: Device::CPU }
}

#[wasm_bindgen(js_name = gpu)]
pub async fn gpu_wasm() -> Result<JsDevice, JsValue> {
    gpu().await
}

#[wasm_bindgen(js_name = cpu)]
pub async fn cpu_wasm() -> Result<JsDevice, JsValue> {
    Ok(cpu())
}

#[wasm_bindgen]
pub fn seed(seed: u64) -> Result<(), JsValue> {
    gpu_sync()?.inner.set_seed(seed);
    Ok(())
}

// Serialize and deserialize implementations for JsDevice tsify
impl Serialize for JsDevice {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Serialize by a short string identifier
        match self.inner {
            Device::CPU => serializer.serialize_str("cpu"),
            Device::GPU(_) => serializer.serialize_str("webgpu"),
        }
    }
}

impl<'de> Deserialize<'de> for JsDevice {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        let normalized = s.to_ascii_lowercase();
        match normalized.as_str() {
            "cpu" => Ok(JsDevice { inner: Device::CPU }),
            "gpu" | "webgpu" => {
                // Use the already-initialized GPU if available
                match gpu_sync() {
                    Ok(dev) => Ok(dev),
                    Err(_e) => Err(DeError::custom(
                        "GPU device not initialized. Call gpu() from JS to initialize first.",
                    )),
                }
            }
            other => Err(DeError::custom(format!("Unsupported device name: {other}"))),
        }
    }
}

impl TryFrom<JsValue> for JsDevice {
    type Error = JsError;
    fn try_from(value: JsValue) -> Result<Self, Self::Error> {
        if value.is_string() {
            let s = value.as_string().unwrap();
            match s.as_str() {
                "cpu" => Ok(cpu()),
                "gpu" | "webgpu" => gpu_sync(),
                _ => Err(JsError::new(&format!("Unsupported device name: {s}"))),
            }
        } else {
            downcast_from_ptr(&value, "__wbg_piston_device", false)
                .ok_or_else(|| JsError::new("Failed to downcast Device from JS value"))
        }
    }
}
