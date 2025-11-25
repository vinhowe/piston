//! WASM bindings for the Piston profiler.
//!
//! Provides JavaScript-accessible functions for:
//! - Starting/stopping profiling sessions
//! - Defining named scopes (like torch.profiler.record_function)
//! - Exporting profile data to Chrome Trace Event format

use piston::{
    clear_profiler, export_chrome_trace, export_events_json, is_profiling_enabled, pop_scope,
    push_scope, start_profiling, stop_profiling,
};
use wasm_bindgen::prelude::*;

/// Start profiling - clears previous events and begins collection
#[wasm_bindgen(js_name = profilerStart)]
pub fn profiler_start() {
    start_profiling();
}

/// Stop profiling - closes any open scopes and stops collection
#[wasm_bindgen(js_name = profilerStop)]
pub fn profiler_stop() {
    stop_profiling();
}

/// Check if profiling is currently enabled
#[wasm_bindgen(js_name = profilerIsEnabled)]
pub fn profiler_is_enabled() -> bool {
    is_profiling_enabled()
}

/// Push a named scope onto the stack (called at start of recordFunction)
#[wasm_bindgen(js_name = profilerPushScope)]
pub fn profiler_push_scope(name: &str) {
    push_scope(name);
}

/// Pop the current scope from the stack (called at end of recordFunction)
#[wasm_bindgen(js_name = profilerPopScope)]
pub fn profiler_pop_scope() {
    pop_scope();
}

/// Clear all collected events
#[wasm_bindgen(js_name = profilerClear)]
pub fn profiler_clear() {
    clear_profiler();
}

/// Export events to Chrome Trace Event format JSON
/// Returns a JSON string that can be loaded into chrome://tracing or Perfetto
#[wasm_bindgen(js_name = profilerExportChromeTrace)]
pub fn profiler_export_chrome_trace() -> String {
    log::debug!("profiler_export_chrome_trace called");
    let result = export_chrome_trace();
    log::debug!(
        "profiler_export_chrome_trace returning {} bytes",
        result.len()
    );
    result
}

/// Export events as a JSON array (structured format for JS consumption)
#[wasm_bindgen(js_name = profilerExportEventsJson)]
pub fn profiler_export_events_json() -> String {
    log::debug!("profiler_export_events_json called");
    export_events_json()
}
