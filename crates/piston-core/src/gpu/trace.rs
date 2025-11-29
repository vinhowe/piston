use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// Global access point for tracing.
///
/// Use `trace_sink()` rather than accessing this directly.
static TRACE_SINK: OnceLock<TraceSink> = OnceLock::new();

/// Get the global trace sink, initializing it on first use.
pub fn trace_sink() -> &'static TraceSink {
    TRACE_SINK.get_or_init(TraceSink::new)
}

/// Configuration for a tracing session.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TraceConfig {
    /// Master on/off switch for tracing.
    pub enabled: bool,
    /// Record tensor graph and per-op metadata.
    pub tensors: bool,
    /// Record low-level GPU allocation / reuse events.
    pub allocations: bool,
    /// If true, record the current JS-provided phase tag on tensor ops.
    pub phases: bool,
}

/// Description of a single graph node / op in the execution order.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorOpDesc {
    /// Index in the post-order execution list.
    pub op_index: u32,
    /// Logical tensor id this op produces.
    pub tensor_id: usize,
    /// Operation name (e.g. "matmul", "softmax").
    pub op_name: String,
    /// Source tensor ids for this op.
    pub src_tensor_ids: Vec<usize>,
    /// DType as human-readable string.
    pub dtype: String,
    /// Shape as plain dimensions.
    pub shape: Vec<usize>,
    /// Optional phase tag (set from JS, e.g. "forward" / "backward").
    pub phase: Option<String>,
    /// Optional logical scope for grouping / filtering in UIs.
    pub scope: Option<String>,
}

/// Lifetime of a logical tensor and its mapping onto a physical buffer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorLifetime {
    /// Logical tensor id.
    pub tensor_id: usize,
    /// First op index that produces this tensor.
    pub producer_index: u32,
    /// Last op index that consumes this tensor.
    pub last_consumer_index: u32,
    /// Whether this tensor participates in gradient computation.
    pub requires_grad: bool,
    /// True if this tensor is a graph output.
    pub is_output: bool,
    /// True if this tensor is assigned to a shared object buffer.
    pub is_shared_object: bool,
    /// Size in bytes requested for this tensor.
    pub size_bytes: u64,
    /// Identifier of the underlying physical buffer.
    ///
    /// We keep this as a string so we don't depend on internal handle layout.
    pub buffer_id: String,
    /// Optional pass index (if provided by the caller / executor).
    pub pass_index: Option<u64>,
}

/// Low-level GPU allocation / reuse events from the buffer pool.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum PhysicalAllocEvent {
    /// A brand-new GPU resource was created.
    Alloc {
        buffer_id: String,
        size_bytes: u64,
        pass_index: u64,
        desc: String,
    },
    /// An existing resource was reclaimed for reuse.
    Reuse {
        buffer_id: String,
        size_bytes: u64,
        pass_index: u64,
        desc: String,
    },
    /// A resource was moved into the "retired" set and may be destroyed next pass.
    Retire {
        buffer_id: String,
        size_bytes: u64,
        pass_index: u64,
        desc: String,
    },
    /// A resource was fully destroyed and VRAM usage decreased.
    Destroy {
        buffer_id: String,
        size_bytes: u64,
        pass_index: u64,
        desc: String,
    },
}

/// Optional per-step metadata that can be attached to a session.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StepMeta {
    pub step_index: u64,
    pub label: Option<String>,
}

/// One complete tracing session.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TraceSession {
    pub config: TraceConfig,
    pub tensor_ops: Vec<TensorOpDesc>,
    pub tensor_lifetimes: Vec<TensorLifetime>,
    pub alloc_events: Vec<PhysicalAllocEvent>,
    pub steps: Vec<StepMeta>,
}

impl TraceSession {
    pub fn new(config: TraceConfig) -> Self {
        Self {
            config,
            tensor_ops: Vec::new(),
            tensor_lifetimes: Vec::new(),
            alloc_events: Vec::new(),
            steps: Vec::new(),
        }
    }

    /// Convert this trace into a Chrome/Perfetto-style trace.
    ///
    /// Returns a `ChromeTrace` struct that is serializable with `serde`.
    /// Call `serde_wasm_bindgen::to_value(&session.to_chrome_trace())` on the JS side.
    pub fn to_chrome_trace(&self) -> ChromeTrace {
        use std::collections::HashMap;

        let events: Vec<ChromeTraceEvent> = self
            .tensor_ops
            .iter()
            .map(|op| {
                let mut args: HashMap<String, ChromeTraceArg> = HashMap::new();
                args.insert("tensor_id".into(), ChromeTraceArg::U64(op.tensor_id as u64));
                args.insert("dtype".into(), ChromeTraceArg::String(op.dtype.clone()));
                args.insert(
                    "shape".into(),
                    ChromeTraceArg::VecU64(op.shape.iter().map(|d| *d as u64).collect()),
                );
                if let Some(phase) = &op.phase {
                    args.insert("phase".into(), ChromeTraceArg::String(phase.clone()));
                }
                if let Some(scope) = &op.scope {
                    args.insert("scope".into(), ChromeTraceArg::String(scope.clone()));
                }

                ChromeTraceEvent {
                    name: op.op_name.clone(),
                    cat: "tensor_op".into(),
                    ph: "X".into(),
                    ts: op.op_index as u64,
                    dur: Some(1),
                    pid: 0,
                    tid: 0,
                    args: Some(args),
                }
            })
            .collect();

        ChromeTrace {
            trace_events: events,
        }
    }
}

/// A single Chrome/Perfetto trace event.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChromeTraceEvent {
    pub name: String,
    pub cat: String,
    pub ph: String,
    pub ts: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dur: Option<u64>,
    pub pid: u64,
    pub tid: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub args: Option<std::collections::HashMap<String, ChromeTraceArg>>,
}

/// Argument value in a Chrome trace event.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChromeTraceArg {
    String(String),
    U64(u64),
    VecU64(Vec<u64>),
}

/// Top-level Chrome/Perfetto trace format.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChromeTrace {
    pub trace_events: Vec<ChromeTraceEvent>,
}

/// In-memory sink for trace events, guarded by configuration flags.
pub struct TraceSink {
    inner: RwLock<Option<TraceSession>>,
    /// JS-provided phase tag (e.g. \"forward\" / \"backward\" / \"optimizer\").
    current_phase: RwLock<Option<String>>,
    /// Optional pass index set by the executor.
    current_pass_index: RwLock<Option<u64>>,
}

impl TraceSink {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(None),
            current_phase: RwLock::new(None),
            current_pass_index: RwLock::new(None),
        }
    }

    /// Start a new trace session, replacing any existing one.
    pub fn start(&self, config: TraceConfig) {
        if !config.enabled {
            let mut guard = self.inner.write();
            *guard = None;
            return;
        }
        let mut guard = self.inner.write();
        *guard = Some(TraceSession::new(config));
    }

    /// Stop the current trace session and return it, if any.
    pub fn stop(&self) -> Option<TraceSession> {
        let mut guard = self.inner.write();
        guard.take()
    }

    pub fn is_enabled(&self) -> bool {
        self.inner
            .read()
            .as_ref()
            .map_or(false, |s| s.config.enabled)
    }

    pub fn tensors_enabled(&self) -> bool {
        self.inner
            .read()
            .as_ref()
            .map_or(false, |s| s.config.enabled && s.config.tensors)
    }

    pub fn allocations_enabled(&self) -> bool {
        self.inner
            .read()
            .as_ref()
            .map_or(false, |s| s.config.enabled && s.config.allocations)
    }

    pub fn phases_enabled(&self) -> bool {
        self.inner
            .read()
            .as_ref()
            .map_or(false, |s| s.config.enabled && s.config.phases)
    }

    /// Set the current phase tag from JS. A `None` value clears the tag.
    pub fn set_phase_tag(&self, phase: Option<String>) {
        let mut guard = self.current_phase.write();
        *guard = phase;
    }

    /// Get the current phase tag (if any).
    pub fn current_phase(&self) -> Option<String> {
        self.current_phase.read().clone()
    }

    /// Set the current pass index from the executor.
    pub fn set_pass_index(&self, pass_index: Option<u64>) {
        let mut guard = self.current_pass_index.write();
        *guard = pass_index;
    }

    /// Get the current pass index (if any).
    pub fn current_pass_index(&self) -> Option<u64> {
        *self.current_pass_index.read()
    }

    pub fn push_tensor_op(&self, mut desc: TensorOpDesc) {
        if !self.tensors_enabled() {
            return;
        }
        if self.phases_enabled() && desc.phase.is_none() {
            desc.phase = self.current_phase();
        }

        let mut guard = self.inner.write();
        if let Some(session) = guard.as_mut() {
            session.tensor_ops.push(desc);
        }
    }

    pub fn push_tensor_lifetime(&self, mut lifetime: TensorLifetime) {
        if !self.tensors_enabled() {
            return;
        }
        if lifetime.pass_index.is_none() {
            lifetime.pass_index = self.current_pass_index();
        }

        let mut guard = self.inner.write();
        if let Some(session) = guard.as_mut() {
            session.tensor_lifetimes.push(lifetime);
        }
    }

    pub fn push_alloc_event(&self, event: PhysicalAllocEvent) {
        if !self.allocations_enabled() {
            return;
        }
        let mut guard = self.inner.write();
        if let Some(session) = guard.as_mut() {
            session.alloc_events.push(event);
        }
    }
}
