use crate::{
    CpuUniform, Executable, ExecutionError, GPUBuffer, HashMap, HashSet, Hasher as HasherType,
    Inner, LazyOp, Storage, TensorError, WgpuDevice,
};
#[cfg(feature = "debug")]
use crate::{DebugTensor, Device, DeviceStorage};
use crate::{Tensor, TensorId};
use parking_lot::RwLock;
use std::collections::BTreeMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::{Arc, Weak};

enum EmitStatus {
    Emitting,
    Emitted,
}

type EmissionMap = HashMap<TensorId, EmitStatus>;
type PostOrderData<'a> = Vec<&'a Tensor>;

struct CachedExecutable {
    executable: Arc<Executable>,
    shared_realloc: bool,
}

pub struct LazyGraphExecutor {
    tensors: Arc<RwLock<BTreeMap<TensorId, Weak<Inner>>>>,
    cache: HashMap<u64, CachedExecutable>,
    pass_index: u64,
    inplace_support: bool,
}

fn panic_cycle(id: TensorId) {
    panic!(
        "Cycle detected whilst computing topological order: {:?}. Try plotting with feature `plotting`.",
        id
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

fn compute_post_order(tensor: &Tensor) -> Vec<&Tensor> {
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

fn compute_post_order_from_nodes(roots: Vec<&Tensor>) -> PostOrderData {
    let mut post_order = Vec::new();
    for root in roots {
        post_order.extend(compute_post_order(root));
    }
    post_order
}

impl LazyGraphExecutor {
    pub fn new(inplace_support: bool) -> Self {
        Self {
            tensors: Arc::new(RwLock::new(BTreeMap::default())),
            cache: HashMap::default(),
            pass_index: Default::default(),
            inplace_support,
        }
    }

    pub fn register_tensor(&self, tensor: &Tensor) {
        log::trace!("Registering tensor {:?}", tensor.id());
        self.tensors
            .write()
            .insert(tensor.id(), Arc::downgrade(&tensor.inner));
    }

    /// Unregisters a tensor by its `TensorId`.
    pub fn unregister_tensor(&self, id: TensorId) {
        log::trace!("Unregistering tensor {:?}", id);
        self.tensors.write().remove(&id);
    }

    fn get_live_tensors(&self) -> BTreeMap<TensorId, Tensor> {
        self.tensors
            .read()
            .iter()
            // Attempt to upgrade from Weak<Inner> → Arc<Inner>.
            // If it succeeds, wrap Arc<Inner> in Tensor.
            .filter_map(|(id, weak_inner)| {
                weak_inner
                    .upgrade()
                    .map(|arc_inner| (*id, Tensor { inner: arc_inner }))
                    .filter(|(_, t)| !t.resolved())
            })
            .collect()
    }

    fn run_post_order<'a>(&self, tensors: Vec<&'a Tensor>) -> PostOrderData<'a> {
        compute_post_order_from_nodes(tensors)
    }

    pub fn sync_live_tensors_graph(
        &mut self,
        gpu_device: &WgpuDevice,
    ) -> anyhow::Result<(), TensorError> {
        log::trace!("Syncing live tensors graph");
        let tensors = self.get_live_tensors();
        log::debug!("All registered IDs: {:?}", self.tensors.read().keys());
        let owned_tensors = tensors.keys().cloned().collect();
        // self.sync_tensors_graph_impl(tensors, Some(owned_tensors), gpu_device, true)
        self.sync_tensors_graph_impl(tensors, Some(owned_tensors), gpu_device, false)
    }

    pub fn sync_tensors_graph(
        &mut self,
        tensors: Vec<&Tensor>,
        gpu_device: &WgpuDevice,
    ) -> anyhow::Result<(), TensorError> {
        self.sync_tensors_graph_impl(
            tensors.into_iter().map(|t| (t.id(), t.clone())).collect(),
            None,
            gpu_device,
            // true,
            false,
        )
    }

    fn run_executable(
        &mut self,
        executable: &Executable,
        gpu_device: &WgpuDevice,
        immediate: bool,
    ) -> anyhow::Result<(), ExecutionError> {
        log::debug!("Running executable");
        #[cfg(feature = "debug")]
        let index = executable.dispatch_debugging(gpu_device)?;

        #[cfg(not(feature = "debug"))]
        let index = executable.dispatch(gpu_device)?;

        if immediate {
            gpu_device.poll(wgpu::MaintainBase::WaitForSubmissionIndex(index));
        }
        Ok(())
    }

    fn sync_tensors_graph_impl(
        &mut self,
        tensors: BTreeMap<TensorId, Tensor>,
        owned_tensors: Option<HashSet<TensorId>>,
        gpu_device: &WgpuDevice,
        use_cache: bool,
    ) -> Result<(), TensorError> {
        // First check if the tensors are already resolved
        log::debug!("Syncing tensors graph");
        if tensors.values().all(|t| t.resolved()) {
            return Ok(());
        }

        // Notably, we compute post order first because we want to hash the tensors in post order,
        // since each hash depends on the hashes of its sources. It's not clear to me that this
        // violates some important unspoken assumption on the part of the LazyTensor authors.
        // We also flip the hash order—post order first, then insertion order—because it's more
        // convenient to treat it as one big hash pass.
        // let tensors = tensors.clone();
        let post_order = self.run_post_order(tensors.values().collect());

        let mut indices = Vec::with_capacity(tensors.len());
        let mut tensor_ids = HashSet::with_capacity_and_hasher(
            tensors.len(),
            BuildHasherDefault::<HasherType>::default(),
        );

        let mut hasher = HasherType::default();
        let mut tensor_hashes = BTreeMap::default();

        let mut consumed_tensors = HashSet::with_capacity_and_hasher(
            tensors.len(),
            BuildHasherDefault::<HasherType>::default(),
        );

        let mut uniform = CpuUniform::new();
        let mut compile_keys = HashMap::default();
        #[cfg(feature = "plotting")]
        let mut strong_counts_inplace = HashMap::default();

        // Keep track of the real source of each tensor; important to help resolve handle those
        // annoying views correctly.
        let mut tensor_sources = HashMap::default();

        // First we loop over the post order to hash the tensors in the right order
        for tensor in &post_order {
            // Scope to drop tensor_hashes before inserting
            let srcs = tensor.op().srcs();
            log::trace!(
                "{:?}: Srcs: {:?}",
                tensor.id(),
                srcs.iter().map(|s| s.id()).collect::<Vec<_>>()
            );
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
                        "{:?}: Supports inplace: {:?}, is variable: {:?}",
                        tensor.id(),
                        tensor.op().supports_inplace(),
                        to_modify_src.is_variable()
                    );

                    if !self.inplace_support {
                        match tensor.op() {
                            LazyOp::Softmax(_) | LazyOp::ScatterAdd(_) | LazyOp::IndexAdd(_) => {
                                true
                            }
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
                    // vinhowe: we need to check if the src is a variable, because we can't
                    // inplace variables unless we've disabled gradient tracking.
                    || to_modify_src.is_variable()
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
                        // 3. The optimizer, if it is a variable. We'll also check if the src is a
                        //    variable.
                        // 4+ Any other Tensor consumers in the post-order. If it's not a variable,
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
            log::debug!("IR: {:?}", ir);
            log::debug!("Tensor hash: {:#x} (op: {:?})", hash, tensor.op().name());
            for src in tensor.op().srcs() {
                consumed_tensors.insert(src.id());
            }
        }

        log::debug!("Post-order hash: {:?}", hasher.finish());

        let output_tensors = tensors
            .iter()
            .filter(|(id, _)| !consumed_tensors.contains(id))
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

        log::debug!("Shape hash: {:?}", hash);

        #[cfg(feature = "debug")]
        let mut cpu_bufs = HashMap::default();

        #[cfg(feature = "debug")]
        // Get CPU buffers from existing allocations
        for tensor in &post_order {
            let storage_guard = tensor.storage();
            match storage_guard.as_ref() {
                Some(Storage::GPU(gpu_buf)) => {
                    log::trace!("Getting CPU buffer for {:?}", tensor.id());
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

        let (mut cached_exec, do_shared_realloc) = if use_cache {
            self.cache
                .remove(&hash)
                .map(|cached_exec| {
                    if cached_exec.shared_realloc {
                        (Arc::try_unwrap(cached_exec.executable).ok(), false)
                    } else {
                        (None, true)
                    }
                })
                .unwrap_or((None, false))
        } else {
            (None, false)
        };

        let mut compiled_ops = Vec::with_capacity(post_order.len());

        gpu_device.begin_pass(self.pass_index);

        let mut allocations = if cached_exec.is_none() || do_shared_realloc {
            Some(gpu_device.allocate_cfg(
                &post_order,
                &output_tensors,
                &compile_keys,
                do_shared_realloc,
                gpu_device,
            )?)
        } else {
            None
        };

        #[cfg(debug_assertions)]
        log::debug!(
            "Resolved tensors in post order: {:?}",
            post_order
                .iter()
                .filter(|t| t.resolved())
                .map(|t| t.id())
                .collect::<Vec<_>>()
        );

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

        for t in &post_order {
            if t.op().is_const() || t.resolved() {
                continue;
            }

            if let Some(allocations) = &mut allocations {
                let id = t.id();
                let inner = allocations.remove(&id).ok_or(TensorError::NoStorage(id))?;
                t.update_storage(Storage::GPU(GPUBuffer {
                    inner,
                    alignment: t.dt().size_of(),
                    cpu_size: Some(t.num_bytes()),
                }));
            }

            if let Some(compile_key) = compile_keys.get(&t.id()) {
                if cached_exec.is_some() {
                    // TODO: Update debug things if needed here, otherwise, delete this branch
                } else if let Some(compiled_op) =
                    t.compile_gpu(compile_key, gpu_device, cfg!(feature = "debug"))
                {
                    compiled_ops.push(Some(compiled_op));
                } else {
                    log::warn!("Compilation failed for operation: {:?}", t.op().name());
                    compiled_ops.push(None);
                }

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
                    t.dt(),
                    t.op()
                        .srcs()
                        .iter()
                        .map(|s| {
                            DebugTensor::new(s.storage().clone(), s.dt(), vec![], s.num_bytes())
                        })
                        .collect(),
                    t.num_bytes(),
                )
            })
            .collect::<Vec<_>>();

        if use_cache {
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
                            log::trace!("Getting CPU buffer for {:?}", tensor.id());
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

                #[cfg(feature = "debug")]
                {
                    cached_exec.debug_list = Some(debug_list);
                    cached_exec.cpu_bufs = Some(Arc::new(RwLock::new(cpu_bufs)));
                }

                self.run_executable(&cached_exec, gpu_device, false)
                    .unwrap();

                #[cfg(all(feature = "debug", feature = "plotting"))]
                {
                    let cpu_bufs_guard = cached_exec.cpu_bufs.as_ref().map(|arc| arc.read());

                    crate::plot::render_to_file(
                        &post_order,
                        &output_tensors,
                        &strong_counts_inplace,
                        cpu_bufs_guard.as_deref(),
                        construct_plot_filename("post_exec", self.pass_index, self.inplace_support),
                    )
                    .unwrap();
                }

                self.cache.insert(
                    hash,
                    CachedExecutable {
                        executable: Arc::new(cached_exec),
                        shared_realloc: true,
                    },
                );
                self.pass_index += 1;
                return Ok(());
            }

            // // On a cache miss: Clear cache because currently I don't know how to make sure
            // // allocations are compatible between runs.
            // self.cache.clear();
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

        let mut executable = Executable::new(
            None,
            filtered_compiled_ops,
            uniform.into_gpu(gpu_device)?,
            #[cfg(feature = "debug")]
            Some(debug_list),
            #[cfg(feature = "debug")]
            Some(Arc::new(RwLock::new(cpu_bufs))),
        );

        self.run_executable(&executable, gpu_device, false).unwrap();

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

        executable.set_storage(post_order.iter().map(|t| t.storage().clone()).collect());

        if use_cache {
            // After creating/running the executable, we cache it
            self.cache.insert(
                hash,
                CachedExecutable {
                    executable: Arc::new(executable),
                    shared_realloc: do_shared_realloc,
                },
            );
        }

        self.pass_index += 1;
        Ok(())
    }
}

impl Default for LazyGraphExecutor {
    fn default() -> Self {
        Self::new(false)
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
