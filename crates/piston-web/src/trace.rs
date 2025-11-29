#![cfg(target_arch = "wasm32")]

use piston::{TraceConfig, TraceSession, trace_sink};
use serde_wasm_bindgen as swb;
use wasm_bindgen::prelude::*;

/// JS-facing trace configuration.
#[wasm_bindgen(getter_with_clone)]
pub struct JsTraceConfig {
    pub tensors: bool,
    pub allocations: bool,
    pub phases: bool,
}

#[wasm_bindgen]
impl JsTraceConfig {
    #[wasm_bindgen(constructor)]
    pub fn new(tensors: bool, allocations: bool, phases: bool) -> JsTraceConfig {
        JsTraceConfig {
            tensors,
            allocations,
            phases,
        }
    }
}

/// Start a new global tracing session.
#[wasm_bindgen(js_name = "traceStart")]
pub fn trace_start(config: &JsTraceConfig) {
    let cfg = TraceConfig {
        enabled: true,
        tensors: config.tensors,
        allocations: config.allocations,
        phases: config.phases,
    };
    trace_sink().start(cfg);
}

/// Stop the current tracing session and return the recorded trace (or null if none).
#[wasm_bindgen(js_name = "traceStop")]
pub fn trace_stop() -> Result<JsValue, JsError> {
    let session: Option<TraceSession> = trace_sink().stop();
    match session {
        Some(s) => swb::to_value(&s).map_err(|e| JsError::new(&e.to_string())),
        None => Ok(JsValue::NULL),
    }
}

/// Set the current phase tag for subsequent ops (e.g. "forward", "backward").
#[wasm_bindgen(js_name = "traceSetPhase")]
pub fn trace_set_phase(phase: Option<String>) {
    trace_sink().set_phase_tag(phase);
}

/// Convert a recorded trace object into Chrome/Perfetto-compatible JSON.
#[wasm_bindgen(js_name = "traceToChrome")]
pub fn trace_to_chrome(trace: JsValue) -> Result<JsValue, JsError> {
    let session: TraceSession = swb::from_value(trace).map_err(|e| JsError::new(&e.to_string()))?;
    let chrome = session.to_chrome_trace();
    swb::to_value(&chrome).map_err(|e| JsError::new(&e.to_string()))
}
