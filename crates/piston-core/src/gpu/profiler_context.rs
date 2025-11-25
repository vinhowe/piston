//! Profiler context for torch.profiler-like functionality.
//!
//! This module provides a thread-local profiling context that collects:
//! - Named scopes (like `torch.profiler.record_function`)
//! - GPU kernel timing data
//! - Buffer allocation events
//!
//! Events can be exported to Chrome Trace Event format JSON for visualization
//! in chrome://tracing or Perfetto.

use crate::HashMap;
use serde::Serialize;
use std::cell::RefCell;

/// Category of a profile event
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfileCategory {
    Scope,
    Kernel,
    /// Buffer allocation (GPU memory pool)
    Allocation,
    /// Buffer deallocation (GPU memory pool)
    Deallocation,
    /// Tensor gets assigned a buffer
    TensorAllocation,
    /// Tensor releases its buffer
    TensorDeallocation,
}

impl ProfileCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            ProfileCategory::Scope => "scope",
            ProfileCategory::Kernel => "kernel",
            ProfileCategory::Allocation => "allocation",
            ProfileCategory::Deallocation => "deallocation",
            ProfileCategory::TensorAllocation => "tensor_allocation",
            ProfileCategory::TensorDeallocation => "tensor_deallocation",
        }
    }
}

/// A single profiling event
#[derive(Debug, Clone, Serialize)]
pub struct ProfileEvent {
    /// Name of the operation or scope
    pub name: String,
    /// Category of the event
    pub category: ProfileCategory,
    /// Start timestamp in microseconds since profiling started
    pub start_us: f64,
    /// Duration in microseconds (0 for instant events)
    pub duration_us: f64,
    /// Additional metadata (shape, dtype, buffer size, workgroups, etc.)
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, String>,
    /// Scope stack at the time of the event (ancestry of recordFunction calls)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub stack: Vec<String>,
}

/// Chrome Trace Event format (subset we use)
#[derive(Debug, Clone, Serialize)]
struct ChromeTraceEvent {
    name: String,
    cat: String,
    ph: String, // "X" for complete, "B" for begin, "E" for end
    ts: f64,    // timestamp in microseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    dur: Option<f64>, // duration in microseconds (only for "X" events)
    pid: u32,
    tid: u32,
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    args: HashMap<String, String>,
}

#[derive(Debug, Serialize)]
struct ChromeTraceOutput {
    #[serde(rename = "traceEvents")]
    trace_events: Vec<ChromeTraceEvent>,
}

/// Thread-local profiler context
pub struct ProfilerContext {
    /// Whether profiling is currently enabled
    enabled: bool,
    /// Stack of active scopes (from recordFunction)
    scope_stack: Vec<ScopeEntry>,
    /// Collected events
    events: Vec<ProfileEvent>,
    /// Start time of profiling session (microseconds, platform time)
    start_time_us: Option<f64>,
}

/// Entry in the scope stack
#[derive(Debug, Clone)]
struct ScopeEntry {
    name: String,
    start_us: f64,
}

impl Default for ProfilerContext {
    fn default() -> Self {
        Self::new()
    }
}

impl ProfilerContext {
    pub fn new() -> Self {
        Self {
            enabled: false,
            scope_stack: Vec::new(),
            events: Vec::new(),
            start_time_us: None,
        }
    }

    /// Start profiling - clears previous events and begins collection
    pub fn start(&mut self) {
        self.events.clear();
        self.scope_stack.clear();
        self.start_time_us = Some(now_us());
        self.enabled = true;
    }

    /// Stop profiling - closes any open scopes and stops collection
    pub fn stop(&mut self) {
        // Close any remaining open scopes
        while !self.scope_stack.is_empty() {
            self.pop_scope();
        }
        self.enabled = false;
    }

    /// Check if profiling is currently enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get the current timestamp relative to profiling start
    fn relative_time_us(&self) -> f64 {
        match self.start_time_us {
            Some(start) => now_us() - start,
            None => 0.0,
        }
    }

    /// Get the current scope stack as a vec of names
    fn current_stack(&self) -> Vec<String> {
        self.scope_stack.iter().map(|s| s.name.clone()).collect()
    }

    /// Push a new scope onto the stack (called at start of recordFunction)
    pub fn push_scope(&mut self, name: &str) {
        if !self.enabled {
            return;
        }
        let start_us = self.relative_time_us();
        self.scope_stack.push(ScopeEntry {
            name: name.to_string(),
            start_us,
        });
    }

    /// Pop the current scope from the stack (called at end of recordFunction)
    pub fn pop_scope(&mut self) {
        if !self.enabled {
            return;
        }
        if let Some(entry) = self.scope_stack.pop() {
            let end_us = self.relative_time_us();
            let duration_us = end_us - entry.start_us;

            // Record scope event
            self.events.push(ProfileEvent {
                name: entry.name,
                category: ProfileCategory::Scope,
                start_us: entry.start_us,
                duration_us,
                metadata: HashMap::default(),
                stack: self.current_stack(), // parent scopes only
            });
        }
    }

    /// Record a buffer allocation event
    pub fn record_allocation(&mut self, size_bytes: u64, usage: &str, buffer_id: u64) {
        if !self.enabled {
            return;
        }
        let mut metadata = HashMap::default();
        metadata.insert("size_bytes".to_string(), size_bytes.to_string());
        metadata.insert("usage".to_string(), usage.to_string());
        metadata.insert("buffer_id".to_string(), buffer_id.to_string());

        self.events.push(ProfileEvent {
            name: "buffer_alloc".to_string(),
            category: ProfileCategory::Allocation,
            start_us: self.relative_time_us(),
            duration_us: 0.0, // instant event
            metadata,
            stack: self.current_stack(),
        });
    }

    /// Record a buffer deallocation/reclaim event
    pub fn record_deallocation(&mut self, size_bytes: u64, buffer_id: u64) {
        if !self.enabled {
            return;
        }
        let mut metadata = HashMap::default();
        metadata.insert("size_bytes".to_string(), size_bytes.to_string());
        metadata.insert("buffer_id".to_string(), buffer_id.to_string());

        self.events.push(ProfileEvent {
            name: "buffer_free".to_string(),
            category: ProfileCategory::Deallocation,
            start_us: self.relative_time_us(),
            duration_us: 0.0,
            metadata,
            stack: self.current_stack(),
        });
    }

    /// Record a tensor allocation event (tensor gets assigned a buffer)
    pub fn record_tensor_allocation(
        &mut self,
        tensor_id: u64,
        name: &str,
        shape: &str,
        dtype: &str,
        size_bytes: u64,
        buffer_id: u64,
    ) {
        if !self.enabled {
            return;
        }
        let mut metadata = HashMap::default();
        metadata.insert("tensor_id".to_string(), tensor_id.to_string());
        metadata.insert("shape".to_string(), shape.to_string());
        metadata.insert("dtype".to_string(), dtype.to_string());
        metadata.insert("size_bytes".to_string(), size_bytes.to_string());
        metadata.insert("buffer_id".to_string(), buffer_id.to_string());

        self.events.push(ProfileEvent {
            name: name.to_string(),
            category: ProfileCategory::TensorAllocation,
            start_us: self.relative_time_us(),
            duration_us: 0.0, // instant event, duration determined by deallocation
            metadata,
            stack: self.current_stack(),
        });
    }

    /// Record a tensor deallocation event (tensor releases its buffer)
    pub fn record_tensor_deallocation(&mut self, tensor_id: u64, buffer_id: u64) {
        if !self.enabled {
            return;
        }
        let mut metadata = HashMap::default();
        metadata.insert("tensor_id".to_string(), tensor_id.to_string());
        metadata.insert("buffer_id".to_string(), buffer_id.to_string());

        self.events.push(ProfileEvent {
            name: "tensor_free".to_string(),
            category: ProfileCategory::TensorDeallocation,
            start_us: self.relative_time_us(),
            duration_us: 0.0,
            metadata,
            stack: self.current_stack(),
        });
    }

    /// Record a GPU kernel execution event
    /// Called after GPU timestamps are read back
    pub fn record_kernel(
        &mut self,
        name: &str,
        start_us: f64,
        duration_us: f64,
        metadata: HashMap<String, String>,
    ) {
        if !self.enabled {
            return;
        }
        self.events.push(ProfileEvent {
            name: name.to_string(),
            category: ProfileCategory::Kernel,
            start_us,
            duration_us,
            metadata,
            stack: self.current_stack(),
        });
    }

    /// Record kernel events from GPU profiler data after execution
    /// The offset_us adjusts GPU timestamps to align with our timeline
    pub fn record_kernel_batch(
        &mut self,
        kernels: Vec<(String, f64, HashMap<String, String>)>, // (name, duration_ns, metadata)
        base_time_us: f64,
    ) {
        if !self.enabled {
            return;
        }

        let stack = self.current_stack();
        let mut current_time = base_time_us;

        for (name, duration_ns, metadata) in kernels {
            let duration_us = duration_ns / 1000.0;
            self.events.push(ProfileEvent {
                name,
                category: ProfileCategory::Kernel,
                start_us: current_time,
                duration_us,
                metadata,
                stack: stack.clone(),
            });
            current_time += duration_us;
        }
    }

    /// Clear all collected events
    pub fn clear(&mut self) {
        self.events.clear();
        self.scope_stack.clear();
    }

    /// Get all collected events
    pub fn events(&self) -> &[ProfileEvent] {
        &self.events
    }

    /// Export events to Chrome Trace Event format JSON
    pub fn export_chrome_trace(&self) -> String {
        log::debug!(
            "export_chrome_trace: exporting {} events",
            self.events.len()
        );

        // Limit events to prevent memory issues
        const MAX_EVENTS: usize = 100_000;
        let events_to_export = if self.events.len() > MAX_EVENTS {
            log::warn!(
                "Profiler has {} events, truncating to {}",
                self.events.len(),
                MAX_EVENTS
            );
            &self.events[..MAX_EVENTS]
        } else {
            &self.events[..]
        };

        let mut trace_events = Vec::with_capacity(events_to_export.len());

        for event in events_to_export {
            let chrome_event = ChromeTraceEvent {
                name: event.name.clone(),
                cat: event.category.as_str().to_string(),
                ph: if event.duration_us > 0.0 { "X" } else { "i" }.to_string(),
                ts: event.start_us,
                dur: if event.duration_us > 0.0 {
                    Some(event.duration_us)
                } else {
                    None
                },
                pid: 1,
                tid: 1,
                args: event.metadata.clone(),
            };
            trace_events.push(chrome_event);
        }

        log::debug!(
            "export_chrome_trace: serializing {} trace events",
            trace_events.len()
        );

        let output = ChromeTraceOutput { trace_events };
        match serde_json::to_string(&output) {
            Ok(json) => {
                log::debug!("export_chrome_trace: serialized {} bytes", json.len());
                json
            }
            Err(e) => {
                log::error!("Failed to serialize Chrome trace: {}", e);
                r#"{"traceEvents":[]}"#.to_string()
            }
        }
    }

    /// Export events as a JSON array (for JS consumption)
    pub fn export_events_json(&self) -> String {
        match serde_json::to_string(&self.events) {
            Ok(json) => json,
            Err(e) => {
                log::error!("Failed to serialize profiler events: {}", e);
                "[]".to_string()
            }
        }
    }
}

// Thread-local profiler context
thread_local! {
    static PROFILER_CONTEXT: RefCell<ProfilerContext> = RefCell::new(ProfilerContext::new());
}

/// Access the thread-local profiler context
pub fn with_profiler<F, R>(f: F) -> R
where
    F: FnOnce(&mut ProfilerContext) -> R,
{
    PROFILER_CONTEXT.with(|ctx| f(&mut ctx.borrow_mut()))
}

/// Check if profiling is currently enabled (fast path for hot code)
pub fn is_profiling_enabled() -> bool {
    PROFILER_CONTEXT.with(|ctx| ctx.borrow().is_enabled())
}

/// Start profiling
pub fn start_profiling() {
    with_profiler(|ctx| ctx.start());
}

/// Stop profiling
pub fn stop_profiling() {
    with_profiler(|ctx| ctx.stop());
}

/// Push a named scope
pub fn push_scope(name: &str) {
    with_profiler(|ctx| ctx.push_scope(name));
}

/// Pop the current scope
pub fn pop_scope() {
    with_profiler(|ctx| ctx.pop_scope());
}

/// Record a buffer allocation event
pub fn record_allocation(size_bytes: u64, usage: &str, buffer_id: u64) {
    with_profiler(|ctx| ctx.record_allocation(size_bytes, usage, buffer_id));
}

/// Record a buffer deallocation event
pub fn record_deallocation(size_bytes: u64, buffer_id: u64) {
    with_profiler(|ctx| ctx.record_deallocation(size_bytes, buffer_id));
}

/// Record a tensor allocation event
pub fn record_tensor_allocation(
    tensor_id: u64,
    name: &str,
    shape: &str,
    dtype: &str,
    size_bytes: u64,
    buffer_id: u64,
) {
    with_profiler(|ctx| {
        ctx.record_tensor_allocation(tensor_id, name, shape, dtype, size_bytes, buffer_id)
    });
}

/// Record a tensor deallocation event
pub fn record_tensor_deallocation(tensor_id: u64, buffer_id: u64) {
    with_profiler(|ctx| ctx.record_tensor_deallocation(tensor_id, buffer_id));
}

/// Clear all events
pub fn clear_profiler() {
    with_profiler(|ctx| ctx.clear());
}

/// Export to Chrome Trace format
pub fn export_chrome_trace() -> String {
    with_profiler(|ctx| ctx.export_chrome_trace())
}

/// Export events as JSON
pub fn export_events_json() -> String {
    with_profiler(|ctx| ctx.export_events_json())
}

/// Get current relative time in microseconds (for syncing GPU timestamps)
pub fn current_profiler_time_us() -> f64 {
    with_profiler(|ctx| ctx.relative_time_us())
}

// Platform-specific high-resolution timing
#[cfg(target_arch = "wasm32")]
fn now_us() -> f64 {
    // Use performance.now() in WASM which returns milliseconds
    // In a Web Worker context, window() returns None, so we need to use
    // js_sys::global() to access the WorkerGlobalScope's performance object
    use wasm_bindgen::JsCast;

    // Try window first (main thread)
    if let Some(perf) = web_sys::window().and_then(|w| w.performance()) {
        return perf.now() * 1000.0; // Convert ms to μs
    }

    // Fall back to WorkerGlobalScope (Web Worker)
    let global = js_sys::global();
    if let Ok(worker_scope) = global.dyn_into::<web_sys::WorkerGlobalScope>() {
        if let Some(perf) = worker_scope.performance() {
            return perf.now() * 1000.0;
        }
    }

    // Last resort: try accessing performance directly from global
    let global = js_sys::global();
    if let Ok(perf) = js_sys::Reflect::get(&global, &"performance".into()) {
        if let Ok(perf_obj) = perf.dyn_into::<web_sys::Performance>() {
            return perf_obj.now() * 1000.0;
        }
    }

    0.0
}

#[cfg(not(target_arch = "wasm32"))]
fn now_us() -> f64 {
    use std::time::Instant;
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    let start = START.get_or_init(Instant::now);
    start.elapsed().as_micros() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_profiling() {
        start_profiling();

        push_scope("outer");
        push_scope("inner");
        pop_scope();
        pop_scope();

        stop_profiling();

        let events = with_profiler(|ctx| ctx.events().to_vec());
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].name, "inner");
        assert_eq!(events[1].name, "outer");
    }

    #[test]
    fn test_disabled_profiling() {
        // Make sure nothing is recorded when disabled
        clear_profiler();
        push_scope("should_not_record");
        pop_scope();

        let events = with_profiler(|ctx| ctx.events().to_vec());
        assert!(events.is_empty());
    }

    #[test]
    fn test_chrome_trace_export() {
        start_profiling();
        push_scope("test_scope");
        pop_scope();
        stop_profiling();

        let json = export_chrome_trace();
        assert!(json.contains("traceEvents"));
        assert!(json.contains("test_scope"));
    }
}
