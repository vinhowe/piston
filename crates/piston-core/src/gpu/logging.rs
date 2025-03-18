use std::sync::Arc;

use crate::{
    wgpu_buffer_to_cpu_buffer, DType, ExportedTensorProfilingEntry, HashMap, Shape, Tensor,
    TensorId,
};
use derive_new::new;
#[cfg(target_arch = "wasm32")]
use half::f16;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::js_sys;

use super::WgpuDevice;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DebugSelection {
    All,
    Some(Vec<String>),
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(getter_with_clone))]
#[derive(Default)]
pub struct StepLogConfig {
    pub profiling: bool,
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(skip))]
    pub debug_selection: Option<DebugSelection>,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl StepLogConfig {
    #[wasm_bindgen(constructor, js_name = "new")]
    pub fn new_js(profiling: bool, debug_selection: JsValue) -> Self {
        let mut config = Self {
            profiling,
            debug_selection: None,
        };

        config.set_debug_selection_js(debug_selection);

        config
    }

    #[wasm_bindgen(getter, js_name = "debug_selection")]
    pub fn debug_selection_js(&self) -> JsValue {
        match &self.debug_selection {
            Some(DebugSelection::All) => JsValue::from_str("all"),
            Some(DebugSelection::Some(ids)) => {
                let array = js_sys::Array::new();
                for id in ids {
                    array.push(&JsValue::from(id));
                }
                array.into()
            }
            None => JsValue::from_str("none"),
        }
    }

    #[wasm_bindgen(setter, js_name = "debug_selection")]
    pub fn set_debug_selection_js(&mut self, value: JsValue) {
        self.debug_selection = if let Some(s) = value.as_string() {
            if s == "all" {
                Some(DebugSelection::All)
            } else {
                None
            }
        } else if js_sys::Array::is_array(&value) {
            let array = js_sys::Array::from(&value);
            let mut ids = Vec::new();

            for i in 0..array.length() {
                if let Some(item) = array.get(i).as_string() {
                    ids.push(item);
                }
            }

            if !ids.is_empty() {
                Some(DebugSelection::Some(ids))
            } else {
                None
            }
        } else {
            None
        };
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(getter_with_clone))]
#[derive(Debug, Clone, new)]
pub struct TensorLogStep {
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(skip))]
    pub id: TensorId,
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(skip))]
    pub srcs: Vec<TensorId>,
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(skip))]
    pub gpu_buf: Option<Arc<wgpu::Buffer>>,
    pub kernel_name: String,
    pub scope: Option<String>,
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(skip))]
    pub dtype: DType,
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(skip))]
    pub shape: Shape,
    pub profile: Option<ExportedTensorProfilingEntry>,
    /// If exported, this will be a flat array of the values
    pub values: Option<Vec<f32>>,
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(skip))]
    pub device: Option<WgpuDevice>,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl TensorLogStep {
    #[wasm_bindgen(getter, js_name = "id")]
    pub fn id(&self) -> usize {
        self.id.0
    }

    #[wasm_bindgen(getter, js_name = "srcs")]
    pub fn srcs(&self) -> Vec<usize> {
        self.srcs.iter().map(|id| id.0).collect()
    }

    #[wasm_bindgen(getter, js_name = "dtype")]
    pub fn dtype(&self) -> String {
        self.dtype.as_str().to_string()
    }

    #[wasm_bindgen(getter, js_name = "shape")]
    pub fn shape(&self) -> Vec<usize> {
        self.shape.to_vec()
    }

    #[wasm_bindgen(
        unchecked_return_type = "Uint8Array|Int8Array|Uint16Array|Int16Array|Uint32Array|Int32Array|Float32Array|Float64Array|null"
    )]
    pub async fn cpu(&self) -> JsValue {
        if let Some(gpu_buf) = self.gpu_buf.as_ref() {
            let cpu_buf = wgpu_buffer_to_cpu_buffer(
                gpu_buf,
                self.dtype.size_of(),
                None,
                self.device
                    .as_ref()
                    .expect("Device should exist if GPU buffer exists"),
            )
            .await;
            match self.dtype {
                DType::F32 => JsValue::from(js_sys::Float32Array::from(
                    cpu_buf.to_slice::<f32>(&self.shape),
                )),
                DType::F16 => JsValue::from(js_sys::Float32Array::from(
                    cpu_buf
                        .to_slice::<f16>(&self.shape)
                        .iter()
                        .map(|x| x.to_f32())
                        .collect::<Vec<_>>()
                        .as_slice(),
                )),
                DType::I32 => JsValue::from(js_sys::Int32Array::from(
                    cpu_buf.to_slice::<i32>(&self.shape),
                )),
                DType::U32 => JsValue::from(js_sys::Uint32Array::from(
                    cpu_buf.to_slice::<u32>(&self.shape),
                )),
                _ => panic!("Unsupported dtype: {:?}", self.dtype),
            }
        } else {
            JsValue::null()
        }
    }

    #[wasm_bindgen(getter, unchecked_return_type = "GPUBuffer|null")]
    pub fn gpu_buf(&self) -> JsValue {
        self.gpu_buf
            .as_ref()
            .map(|buf| buf.as_webgpu_buffer().unwrap().into())
            .unwrap_or(JsValue::null())
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(getter_with_clone))]
#[derive(Debug, Clone, new)]
pub struct StepLog {
    pub tensors: Vec<TensorLogStep>,
    pub cached: bool,
    pub hash: u64,
    pub using_shared_buffers: bool,
}

impl StepLog {
    pub fn from_post_order(
        post_order: Vec<&Tensor>,
        profiling_entries: Option<HashMap<TensorId, ExportedTensorProfilingEntry>>,
        mut gpu_bufs: Option<HashMap<TensorId, wgpu::Buffer>>,
        hash: u64,
        cached: bool,
        using_shared_buffers: bool,
        device: &WgpuDevice,
    ) -> Self {
        let tensors = post_order
            .iter()
            .map(|t| {
                let gpu_buf = gpu_bufs
                    .as_mut()
                    .and_then(|bufs| bufs.remove(&t.id()).map(Arc::new));
                TensorLogStep::new(
                    t.id(),
                    t.op().srcs().iter().map(|tensor| tensor.id()).collect(),
                    gpu_bufs
                        .as_mut()
                        .and_then(|bufs| bufs.remove(&t.id()).map(Arc::new)),
                    t.op().name().to_string(),
                    t.scope().as_ref().map(|s| s.to_string()),
                    t.dtype(),
                    t.shape().clone(),
                    profiling_entries
                        .as_ref()
                        .and_then(|entries| entries.get(&t.id()))
                        .cloned(),
                    None, // ...for now
                    gpu_buf.map(|_| device.clone()),
                )
            })
            .collect();
        Self::new(tensors, cached, hash, using_shared_buffers)
    }
}
