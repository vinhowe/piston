use crate::{
    Compiled, CpuUniform, DebugSelection, Executable, ExecutionError, ExecutionResult, GPUBuffer,
    HashMap, HashSet, Hasher as HasherType, Inner, LazyOp, StepLog, StepLogConfig, Storage,
    TensorError, WgpuDevice, reset_scope_context,
};
#[cfg(feature = "debug")]
use crate::{DebugTensor, Device, DeviceStorage};
use crate::{OpTensor, TensorId};
use maybe_async::maybe_async;
use parking_lot::RwLock;
use std::collections::BTreeMap;
use std::hash::Hasher;
use std::sync::{Arc, Weak};

#[derive(Debug, thiserror::Error)]
pub enum LazyGraphExecutorError {
    #[error(
        "one of the variables needed for gradient computation has been modified by an inplace operation."
    )]
    InplaceError,
    #[error(transparent)]
    TensorError(#[from] crate::TensorError),
    #[error(transparent)]
    DeviceError(#[from] crate::DeviceError),
    #[error(transparent)]
    OperationError(#[from] crate::OperationError),
}

enum EmitStatus {
    Emitting,
    Emitted,
}

type EmissionMap = HashMap<TensorId, EmitStatus>;
type PostOrderData<'a> = Vec<&'a OpTensor>;

struct CachedExecutable {
    executable: Arc<Executable>,
    is_shared_realloc: bool,
}

pub struct LazyGraphExecutor {
    tensors: Arc<RwLock<BTreeMap<TensorId, Weak<Inner>>>>,
    cache: HashMap<u64, CachedExecutable>,
    step_log_config: Option<StepLogConfig>,
    pass_index: u64,
    inplace_support: bool,
    caching_enabled: bool,
    shared_object_allocation_enabled: bool,
}

fn panic_cycle(id: TensorId) {
    panic!(
        "Cycle detected whilst computing topological order: {id:?}. Try plotting with feature `plotting`."
    );
}

#[cfg(feature = "debug")]
macro_rules! mut_in_debug {
    ($ident:ident) => { mut $ident };
}

#[cfg(not(feature = "debug"))]
macro_rules! mut_in_debug {
    ($ident:ident) => {
        $ident
    };
}

fn compute_post_order(tensor: &OpTensor) -> Vec<&OpTensor> {
    let mut post_order = Vec::new();
    let mut node_stack = vec![tensor];
    let mut emission_map = EmissionMap::default();
    while let Some(node) = node_stack.last().cloned() {
        match emission_map.get(&node.id()) {
            None => {
                emission_map.insert(node.id(), EmitStatus::Emitting);
                for src in node.op().srcs() {
                    if let Some(EmitStatus::Emitting) = emission_map.get(&src.id()) {
                        panic_cycle(src.id());
                    }

                    node_stack.push(src);
                }
            }
            Some(EmitStatus::Emitting) => {
                for src in node.op().srcs() {
                    if let Some(EmitStatus::Emitting) = emission_map.get(&src.id()) {
                        panic_cycle(src.id());
                    }
                }
                emission_map.insert(node.id(), EmitStatus::Emitted);
                post_order.push(node);
                node_stack.pop();
            }
            Some(EmitStatus::Emitted) => {
                node_stack.pop();
            }
        }
    }
    post_order
}

fn compute_post_order_from_nodes(roots: Vec<&OpTensor>) -> PostOrderData<'_> {
    let mut post_order = Vec::new();
    for root in roots {
        post_order.extend(compute_post_order(root));
    }
    post_order
}

impl LazyGraphExecutor {
    pub fn new(
        inplace_support: bool,
        caching_enabled: bool,
        shared_object_allocation_enabled: bool,
    ) -> Self {
        Self {
            tensors: Arc::new(RwLock::new(BTreeMap::default())),
            cache: HashMap::default(),
            pass_index: Default::default(),
            inplace_support,
            step_log_config: None,
            caching_enabled,
            shared_object_allocation_enabled,
        }
    }

    pub fn register_tensor(&self, tensor: &OpTensor) {
        log::trace!("Registering tensor {:?}", tensor.id());
        self.tensors
            .write()
            .insert(tensor.id(), Arc::downgrade(&tensor.inner));
    }

    /// Unregisters a tensor by its `TensorId`.
    pub fn unregister_tensor(&self, id: TensorId) {
        log::trace!("Unregistering tensor {id:?}");
        self.tensors.write().remove(&id);
    }

    fn get_live_tensors(&self) -> BTreeMap<TensorId, OpTensor> {
        self.tensors
            .read()
            .iter()
            // Attempt to upgrade from Weak<Inner> → Arc<Inner>.
            // If it succeeds, wrap Arc<Inner> in Tensor.
            .filter_map(|(id, weak_inner)| {
                weak_inner
                    .upgrade()
                    .map(|arc_inner| (*id, OpTensor { inner: arc_inner }))
                    .filter(|(_, t)| !t.resolved())
            })
            .collect()
    }

    fn run_post_order<'a>(&self, tensors: Vec<&'a OpTensor>) -> PostOrderData<'a> {
        compute_post_order_from_nodes(tensors)
    }

    #[maybe_async]
    pub async fn sync_live_tensors_graph(
        &mut self,
        gpu_device: &WgpuDevice,
    ) -> anyhow::Result<(), LazyGraphExecutorError> {
        reset_scope_context();
        log::trace!("Syncing live tensors graph");
        let tensors = self.get_live_tensors();
        log::debug!("All registered IDs: {:?}", self.tensors.read().keys());
        let owned_tensors = tensors.keys().cloned().collect();
        self.sync_tensors_graph_impl(tensors, Some(owned_tensors), gpu_device)
            .await
    }

    #[maybe_async]
    pub async fn sync_tensors_graph(
        &mut self,
        tensors: Vec<&OpTensor>,
        gpu_device: &WgpuDevice,
    ) -> anyhow::Result<(), LazyGraphExecutorError> {
        self.sync_tensors_graph_impl(
            tensors.into_iter().map(|t| (t.id(), t.clone())).collect(),
            None,
            gpu_device,
        )
        .await
    }

    #[maybe_async]
    async fn run_executable(
        &mut self,
        executable: &mut Executable,
        gpu_device: &WgpuDevice,
        immediate: bool,
    ) -> anyhow::Result<ExecutionResult, ExecutionError> {
        log::debug!("Running executable");
        #[cfg(feature = "debug")]
        let index = executable.dispatch_debugging(gpu_device)?;

        #[cfg(not(feature = "debug"))]
        let (index, result) = executable
            .dispatch(gpu_device, self.step_log_config.as_ref())
            .await?;

        if immediate {
            gpu_device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(index),
                    timeout: None,
                })
                .unwrap();
        }
        Ok(result)
    }

    #[maybe_async]
    async fn sync_tensors_graph_impl(
        &mut self,
        tensors: BTreeMap<TensorId, OpTensor>,
        owned_tensors: Option<HashSet<TensorId>>,
        gpu_device: &WgpuDevice,
    ) -> Result<(), LazyGraphExecutorError> {
        // First check if the tensors are already resolved
        log::debug!("Syncing tensors graph");
        if tensors.values().all(|t| t.resolved()) {
            return Ok(());
        }

        let use_cache = self.caching_enabled;

        // Notably, we compute post order first because we want to hash the tensors in post order,
        // since each hash depends on the hashes of its sources. It's not clear to me that this
        // violates some important unspoken assumption on the part of the LazyTensor authors.
        // We also flip the hash order—post order first, then insertion order—because it's more
        // convenient to treat it as one big hash pass.
        // let tensors = tensors.clone();
        let combined_post_orders = self.run_post_order(tensors.values().collect());

        let mut indices = Vec::with_capacity(tensors.len());
        let mut tensor_ids = HashSet::with_capacity_and_hasher(tensors.len(), Default::default());

        let mut hasher = HasherType::default();
        let mut tensor_hashes = BTreeMap::default();

        // Keep track of tensors that have been used as an src by another tensor
        let mut used_as_src = HashSet::with_capacity_and_hasher(tensors.len(), Default::default());

        let mut uniform = CpuUniform::new();
        let mut compile_keys = HashMap::default();
        #[cfg(feature = "plotting")]
        let mut strong_counts_inplace = HashMap::default();

        // Keep track of the real source of each tensor; important to help resolve handle those
        // annoying views correctly.
        let mut tensor_sources = HashMap::default();

        let mut seen_nodes = HashSet::default();
        let mut post_order = Vec::new();

        // let mut can_inplace_count = 0;
        // let mut no_op_support_for_inplacing_count = 0;
        // let mut no_inplacing_because_source_requires_grad_count = 0;
        // let post_order_len = combined_post_orders.len();

        // First we loop over the post order to hash the tensors in the right order
        for tensor in combined_post_orders.into_iter() {
            if seen_nodes.insert(tensor.id()) {
                post_order.push(tensor);
                // Scope to drop tensor_hashes before inserting
                let srcs = tensor.op().srcs();
                // log::trace!(
                //     "{:?}: Srcs: {:?}",
                //     tensor.id(),
                //     srcs.iter().map(|s| s.id()).collect::<Vec<_>>()
                // );
                let first_src = srcs.first().cloned();

                let mut to_modify = None;
                if !matches!(tensor.op(), LazyOp::View(_)) {
                    tensor_sources.insert(tensor.id(), tensor);
                    to_modify = first_src.map(|src| {
                        tensor_sources
                            .get(&src.id())
                            .cloned()
                            .expect("Source missing entry in tensor_sources")
                    });
                } else if let Some(src) = tensor_sources
                    .get(&first_src.expect("All views should have one src").id())
                    .cloned()
                {
                    tensor_sources.insert(tensor.id(), src);
                    to_modify = Some(src);
                }

                let can_inplace = match to_modify {
                    Some(to_modify_src) => {
                        log::trace!(
                            "{:?}: Supports inplace: {:?}, is parameter: {:?}",
                            tensor.id(),
                            tensor.op().supports_inplace(),
                            to_modify_src.requires_grad()
                        );

                        if tensor.is_inplace() {
                            true
                        } else if !self.inplace_support {
                            // TODO(vinhowe): This really is horrible; we should be able to just
                            // check if the op supports inplace.
                            match tensor.op() {
                                LazyOp::Softmax(_)
                                | LazyOp::ScatterAdd(_)
                                | LazyOp::IndexAdd(_) => true,
                                LazyOp::Detach(d) => {
                                    matches!(
                                        d.as_ref(),
                                        LazyOp::Softmax(_)
                                            | LazyOp::ScatterAdd(_)
                                            | LazyOp::IndexAdd(_)
                                    )
                                }
                                _ => false,
                            }
                        } else if !tensor.op().supports_inplace()
                            // vinhowe: we need to check if the src is a parameter, because we can't
                            // inplace parameters unless we've disabled gradient tracking.
                            || to_modify_src.requires_grad()
                        {
                            false
                        } else {
                            // Typical references:
                            // 1. Its original consumer. Whatever scope it was created in.
                            // 2. `tensors`, as passed into this method, if it wasn't resolved and we
                            //    upgraded its weak reference. This happens when we do a sync of live
                            //    tensors, say, in an optimizer step, but a one-off sync won't do this.
                            //    This is why we have the optional `owned_tensors`.
                            // If these two are the only references, then we can inplace. Usually,
                            // additional references include, not in any particular order:
                            // 3. The optimizer, if it is a parameter. We'll also check if the src is a
                            //    parameter.
                            // 4+ Any other Tensor consumers in the post-order. If it's not a parameter,
                            //    these are the references we're concerned about messing with.
                            //
                            // If we own a copy, 2, otherwise 1.
                            let expected_strong = owned_tensors
                                .as_ref()
                                .and_then(|ot| ot.contains(&to_modify_src.id()).then_some(2))
                                .unwrap_or(1);

                            to_modify_src.strong_count() == expected_strong
                        }
                    }
                    None => false,
                };

                #[cfg(feature = "plotting")]
                strong_counts_inplace.insert(tensor.id(), (tensor.strong_count(), can_inplace));
                log::trace!(
                    "Can inplace: {:?}, op: {:?} ({:?}), strong: {:?}",
                    can_inplace,
                    tensor.op().name(),
                    tensor.id(),
                    to_modify.as_ref().map(|t| t.strong_count())
                );
                let compile_key = tensor.gpu_compile_key(can_inplace, &mut uniform);
                let ir = tensor.op().ir();
                ir.shape_hash(&mut hasher, &tensor_hashes, &compile_key);
                if let Some(compile_key) = compile_key {
                    compile_keys.insert(tensor.id(), compile_key);
                }
                let hash = hasher.finish();
                tensor_hashes.insert(tensor.id(), hash);
                log::debug!("IR: {ir:?}");
                log::debug!("Tensor hash: {hash:#x} (op: {:?})", tensor.op().name());
                for src in tensor.op().srcs() {
                    used_as_src.insert(src.id());
                }
            } else {
                // If we've already seen this node, just add its hash to the hasher
                hasher.write_u64(
                    *tensor_hashes
                        .get(&tensor.id())
                        .expect("Missing shape hash for tensor"),
                );
            }
        }

        log::debug!("Post-order hash: {:?}", hasher.finish());

        let output_tensors = tensors
            .iter()
            .filter(|(id, _)| !used_as_src.contains(id))
            .map(|(id, tensor)| (*id, tensor))
            .collect::<BTreeMap<_, _>>();

        #[cfg(feature = "plotting")]
        crate::plot::render_to_file(
            &post_order,
            &output_tensors,
            &strong_counts_inplace,
            None,
            construct_plot_filename("post_order", self.pass_index, self.inplace_support),
        )
        .unwrap();

        for (i, (id, tensor)) in tensors.iter().enumerate() {
            if !tensor_ids.insert(id) || tensor.resolved() {
                continue;
            }

            #[cfg(feature = "debug")]
            if !tensor_hashes.contains_key(id) {
                log::warn!("Missing shape hash for tensor {:?}", id);
                continue;
            }
            hasher.write_u64(
                *tensor_hashes
                    .get(id)
                    .expect("Missing shape hash for tensor"),
            );
            indices.push(i);
        }
        let hash = hasher.finish();

        log::debug!("Shape hash: {hash:?}");

        #[cfg(feature = "debug")]
        let mut cpu_bufs = HashMap::default();

        #[cfg(feature = "debug")]
        // Get CPU buffers from existing allocations
        for tensor in &post_order {
            let storage_guard = tensor.storage();
            match storage_guard.as_ref() {
                Some(Storage::GPU(gpu_buf)) => {
                    log::trace!("Getting CPU buffer for {tensor.id():?}");
                    cpu_bufs.insert(
                        tensor.id(),
                        gpu_buf.to_cpu(&Device::GPU(gpu_device.clone()))?,
                    );
                }
                Some(Storage::CPU(cpu_buf)) => {
                    log::trace!("Using existing CPU buffer for {:?}", tensor.id());
                    cpu_bufs.insert(tensor.id(), cpu_buf.clone());
                }
                None => {}
            }
        }

        let (mut cached_exec, do_shared_realloc, is_shared_realloc) = if use_cache {
            self.cache
                .remove(&hash)
                .map(|cached_exec| {
                    if cached_exec.is_shared_realloc {
                        // Cache hit, no need to realloc, shared realloc
                        (Arc::try_unwrap(cached_exec.executable).ok(), false, true)
                    } else {
                        // Cache hit, not shared realloc and needs shared realloc, not yet shared
                        // realloc
                        (None, true, false)
                    }
                })
                // Cache miss, no need to realloc, can't be shared realloc
                .unwrap_or((None, false, false))
        } else {
            // Not using cache, no need to realloc, we don't allow shared realloc
            (None, false, false)
        };

        let mut compiled_ops = Vec::with_capacity(post_order.len());

        gpu_device.begin_pass(self.pass_index);

        let mut allocations = if cached_exec.is_none() || do_shared_realloc {
            Some(gpu_device.allocate_cfg(
                &post_order,
                &output_tensors,
                &compile_keys,
                self.shared_object_allocation_enabled,
                gpu_device,
            )?)
        } else {
            None
        };

        #[cfg(debug_assertions)]
        {
            let resolved_tensors = post_order.iter().filter(|t| t.resolved()).count();
            let resolved_tensors_len = post_order.len();
            log::trace!(
                "Post order: {:?}",
                post_order.iter().map(|t| t.id()).collect::<Vec<_>>()
            );
            log::trace!(
                "Resolved tensors: {:?}",
                post_order
                    .iter()
                    .filter(|t| t.resolved())
                    .map(|t| t.id())
                    .collect::<Vec<_>>()
            );
            log::debug!(
                "Length of resolved tensors in post order: {resolved_tensors} / {resolved_tensors_len}"
            );
        }

        #[cfg(feature = "debug")]
        let mut compute_dsts = Vec::new();

        #[cfg(feature = "plotting")]
        crate::plot::render_to_file(
            &post_order,
            &output_tensors,
            &strong_counts_inplace,
            None,
            construct_plot_filename("prealloc", self.pass_index, self.inplace_support),
        )
        .unwrap();

        #[cfg(not(feature = "debug"))]
        let mut debug_list = BTreeMap::new();

        let mut i = 0;
        for t in &post_order {
            if t.op().is_const() || t.resolved() {
                continue;
            }

            if let Some(allocations) = &mut allocations {
                let id = t.id();
                let inner = allocations.remove(&id).ok_or(TensorError::NoStorage(id))?;
                t.update_storage(Storage::GPU(GPUBuffer {
                    inner,
                    alignment: t.dtype().size_of(),
                    cpu_size: Some(t.num_bytes()),
                }));
            }

            if let Some(compile_key) = compile_keys.get(&t.id()) {
                let selected_for_step_log = self
                    .step_log_config
                    .as_ref()
                    .map(|c| c.debug_selection.as_ref())
                    .and_then(|s| {
                        s.as_ref().map(|s| match s {
                            DebugSelection::All => true,
                            DebugSelection::Some(scopes) => {
                                if let Some(scope) = t.scope() {
                                    scopes.contains(scope)
                                } else {
                                    false
                                }
                            }
                        })
                    })
                    .unwrap_or(false);

                #[cfg(not(feature = "debug"))]
                let debug_list_ref = &mut debug_list;

                // TODO(vinhowe): Rethink this whole thing and don't use a function here.
                #[cfg(not(feature = "debug"))]
                let mut set_debug_buffer =
                    move |compiled_op: &mut Compiled| -> Result<(), TensorError> {
                        let tensor_debug_buffer = t.debug_tensor();
                        if selected_for_step_log || tensor_debug_buffer.is_some() {
                            // We ignore any requests to debug copy items
                            if let Compiled::Compute(op) = compiled_op {
                                let debug_tensor = if let Some(tensor) = tensor_debug_buffer {
                                    tensor
                                } else {
                                    t.get_or_create_debug_tensor()?
                                };
                                let storage_guard = debug_tensor.storage();
                                let debug_buffer = storage_guard
                                    .as_ref()
                                    .expect("Debug tensor should have a storage")
                                    .try_gpu()?;
                                op.debug_buffer = Some(debug_buffer.inner.clone());
                                debug_list_ref.insert(t.id(), (*t).clone());
                            };
                        };
                        Ok(())
                    };

                if let Some(exec) = cached_exec.as_mut() {
                    let compiled_op = &mut exec.steps[i];
                    #[cfg(not(feature = "debug"))]
                    set_debug_buffer(compiled_op)?;
                } else if let Some(mut compiled_op) =
                    t.compile_gpu(compile_key, gpu_device, selected_for_step_log)
                {
                    #[cfg(not(feature = "debug"))]
                    set_debug_buffer(&mut compiled_op)?;
                    compiled_ops.push(Some(compiled_op));
                } else {
                    log::warn!("Compilation failed for operation: {:?}", t.op().name());
                    compiled_ops.push(None);
                };
                i += 1;

                #[cfg(feature = "debug")]
                compute_dsts.push((*t).clone());
            }
        }

        // At this point we have a cached executable that we want to ignore.
        cached_exec = cached_exec.map(|exec| exec.with_tensors(&post_order).unwrap());

        #[cfg(feature = "debug")]
        let debug_list = compute_dsts
            .into_iter()
            .map(|t| {
                DebugTensor::new(
                    t.storage().clone(),
                    t.dtype(),
                    t.op()
                        .srcs()
                        .iter()
                        .map(|s| {
                            DebugTensor::new(s.storage().clone(), s.dtype(), vec![], s.num_bytes())
                        })
                        .collect(),
                    t.num_bytes(),
                )
            })
            .collect::<Vec<_>>();

        let is_cached = cached_exec.is_some();

        let mut executable;
        if let Some(mut_in_debug!(cached_exec)) = cached_exec {
            log::debug!("Using cached executable");

            #[cfg(feature = "debug")]
            let mut cpu_bufs = HashMap::default();

            #[cfg(feature = "debug")]
            // Get CPU buffers from existing allocations
            for tensor in &post_order {
                let storage_guard = tensor.storage();
                match storage_guard.as_ref() {
                    Some(Storage::GPU(gpu_buf)) => {
                        log::trace!("Getting CPU buffer for {tensor.id():?}");
                        cpu_bufs.insert(
                            tensor.id(),
                            gpu_buf.to_cpu(&Device::GPU(gpu_device.clone()))?,
                        );
                    }
                    Some(Storage::CPU(cpu_buf)) => {
                        log::trace!("Using existing CPU buffer for {tensor.id():?}");
                        cpu_bufs.insert(tensor.id(), cpu_buf.clone());
                    }
                    None => {}
                }
            }

            #[cfg(feature = "debug")]
            {
                cached_exec.debug_list = Some(debug_list);
                cached_exec.cpu_bufs = Some(Arc::new(RwLock::new(cpu_bufs)));
            }

            executable = cached_exec;
        } else {
            if use_cache {
                // On a cache miss: Clear cache because currently I don't know how to make sure
                // allocations are compatible between runs.
                self.cache.clear();
            }

            #[cfg(feature = "plotting")]
            crate::plot::render_to_file(
                &post_order,
                &output_tensors,
                &strong_counts_inplace,
                None,
                construct_plot_filename("alloc", self.pass_index, self.inplace_support),
            )
            .unwrap();

            // Only keep the ops that successfully compiled.
            let filtered_compiled_ops: Vec<_> = compiled_ops.into_iter().flatten().collect();

            executable = Executable::new(
                None,
                filtered_compiled_ops,
                uniform.into_gpu(gpu_device)?,
                #[cfg(not(feature = "debug"))]
                if debug_list.is_empty() {
                    None
                } else {
                    Some(debug_list)
                },
                #[cfg(feature = "debug")]
                Some(Arc::new(RwLock::new(cpu_bufs))),
            );
        }

        let result = self
            .run_executable(&mut executable, gpu_device, false)
            .await
            .unwrap();

        #[cfg(all(feature = "debug", feature = "plotting"))]
        {
            let cpu_bufs_guard = executable.cpu_bufs.as_ref().map(|arc| arc.read());

            crate::plot::render_to_file(
                &post_order,
                &output_tensors,
                &strong_counts_inplace,
                cpu_bufs_guard.as_deref(),
                construct_plot_filename("post_exec", self.pass_index, self.inplace_support),
            )
            .unwrap();
        }

        if !is_cached && use_cache {
            // We save the storage of the executable to be used in the next pass
            executable.set_storage(post_order.iter().map(|t| t.storage().clone()).collect());
        }

        if self.step_log_config.is_some() {
            let step_log = StepLog::from_post_order(
                post_order,
                result.profiling_entries,
                result.gpu_bufs,
                hash,
                is_cached,
                is_shared_realloc,
                gpu_device,
            );
            gpu_device.set_last_step_log(step_log);
        }

        if use_cache {
            // After creating/running the executable, we cache it
            self.cache.insert(
                hash,
                CachedExecutable {
                    executable: Arc::new(executable),
                    // If we already did a shared realloc, we don't need to do it again
                    is_shared_realloc: is_shared_realloc || do_shared_realloc,
                },
            );
        }

        self.pass_index += 1;
        Ok(())
    }

    pub fn step_log_config(&self) -> Option<&StepLogConfig> {
        self.step_log_config.as_ref()
    }

    pub fn set_step_log_config(&mut self, config: StepLogConfig) {
        let old_config_debug_selection = self
            .step_log_config
            .as_ref()
            .map(|c| c.debug_selection.clone());
        // If the debug selection has changed, clear the cache; we'll need to recompile all the ops
        let new_config_debug_selection = self
            .step_log_config
            .as_ref()
            .map(|c| c.debug_selection.clone());
        if old_config_debug_selection != new_config_debug_selection {
            log::debug!("Debug selection changed, clearing cache");
            self.cache.clear();
        }
        self.step_log_config = Some(config);
    }

    pub fn set_caching_enabled(&mut self, enabled: bool) {
        self.caching_enabled = enabled;
    }

    pub fn caching_enabled(&self) -> bool {
        self.caching_enabled
    }

    pub fn set_shared_object_allocation_enabled(&mut self, enabled: bool) {
        self.shared_object_allocation_enabled = enabled;
    }

    pub fn shared_object_allocation_enabled(&self) -> bool {
        self.shared_object_allocation_enabled
    }

    pub fn set_inplace_support(&mut self, enabled: bool) {
        self.inplace_support = enabled;
    }

    pub fn inplace_support(&self) -> bool {
        self.inplace_support
    }
}

impl Default for LazyGraphExecutor {
    fn default() -> Self {
        Self::new(false, false, false)
    }
}

/// Constructs the plot filename with an optional "_inplace" segment.
///
/// The resulting filename is in the format:
///   "<name>[_inplace]_<pass_index>.svg"
///
/// # Arguments
/// * `name` - The base part of the file name (e.g., "post_order").
/// * `pass_index` - The pass index used in the file name.
/// * `inplace_support` - Flag indicating whether to add "_inplace" before the pass number.
#[cfg(feature = "plotting")]
fn construct_plot_filename(name: &str, pass_index: u64, inplace_support: bool) -> String {
    if inplace_support {
        format!("{}_inplace_{}", name, pass_index)
    } else {
        format!("{}_{}", name, pass_index)
    }
}
