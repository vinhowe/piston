use crate::gpu::{GpuUniform, PoolError, StaticResourcePoolAccessor, WgpuDevice};
use crate::{
    Compiled, CompiledCopy, CompiledOp, ExportedTensorProfilingEntry, HashMap, PooledGPUBuffer,
    StepLogConfig, Storage, TensorId,
};

// #[cfg(feature = "debug")] (for tensor?)
use crate::OpTensor;
use crate::gpu::Profiler;
#[cfg(feature = "debug")]
use crate::{DType, HashMap, TensorId};
#[cfg(feature = "debug")]
use crate::{DeviceStorage, KernelKey, RVec};
use derive_new::new;
use maybe_async::maybe_async;
#[cfg(feature = "debug")]
use parking_lot::RwLock;
#[cfg(feature = "debug")]
use slotmap::Key;
#[cfg(not(feature = "debug"))]
use std::collections::BTreeMap;
use std::iter::Peekable;
#[cfg(feature = "debug")]
use std::sync::Arc;
use wgpu::SubmissionIndex;

#[cfg(feature = "debug")]
#[derive(new, Debug, Clone)]
pub struct DebugTensor {
    pub(crate) storage: Option<Storage>,
    pub(crate) dtype: DType,
    pub(crate) srcs: Vec<DebugTensor>,
    pub(crate) size: usize,
}

/// # Executable
///
/// A linear sequence of compiled operations, with a single uniform buffer
/// containing metadata for all operations.
#[derive(new, Debug)]
pub struct Executable {
    storage: Option<Vec<Option<Storage>>>,
    pub(crate) steps: Vec<Compiled>,
    gpu_uniform: GpuUniform,
    #[cfg(not(feature = "debug"))]
    pub(crate) debug_list: Option<BTreeMap<TensorId, OpTensor>>,
    #[cfg(feature = "debug")]
    pub(crate) debug_list: Option<Vec<DebugTensor>>,
    #[cfg(feature = "debug")]
    pub(crate) cpu_bufs: Option<Arc<RwLock<HashMap<TensorId, CPUBuffer>>>>,
}

//this error ExecutionError
#[derive(Debug, thiserror::Error)]
pub enum ExecutionError {
    #[error(transparent)]
    PipelineNotFound(#[from] PoolError),
    #[error("Failed during debugging: {0}")]
    DebuggingError(&'static str),
}

pub struct ExecutionResult {
    pub profiling_entries: Option<HashMap<TensorId, ExportedTensorProfilingEntry>>,
    pub gpu_bufs: Option<HashMap<TensorId, PooledGPUBuffer>>,
}

impl Executable {
    // Helper function to set up pipeline and dispatch workgroups
    #[inline]
    fn compute_pass_inner(
        compute_pass: &mut wgpu::ComputePass<'_>,
        op: &CompiledOp,
        pipeline_resources: &crate::StaticResourcePoolReadLockAccessor<
            '_,
            crate::ComputePipelineHandle,
            wgpu::ComputePipeline,
        >,
        gpu_uniform: &GpuUniform,
    ) -> Result<(), ExecutionError> {
        compute_pass.set_pipeline(pipeline_resources.get(op.pipeline_handle())?);
        for (group_index, bind_group) in op.storage_groups().iter().enumerate() {
            compute_pass.set_bind_group(group_index as u32, bind_group, &[]);
        }
        let uniform_group_index = op.storage_groups().len() as u32;
        let uniform_group = gpu_uniform.bind_group();
        compute_pass.set_bind_group(uniform_group_index, uniform_group, &[op.offset()]);
        let [x_count, y_count, z_count] = op.workgroup_count().as_slice();
        compute_pass.dispatch_workgroups(x_count, y_count, z_count);
        Ok(())
    }

    #[cfg(all(not(feature = "gpu-profiling"), not(feature = "debug")))]
    #[maybe_async]
    pub async fn dispatch(
        &mut self,
        device: &WgpuDevice,
        step_log_config: Option<&StepLogConfig>,
    ) -> Result<(SubmissionIndex, ExecutionResult), ExecutionError> {
        let pipeline_resources: crate::StaticResourcePoolReadLockAccessor<
            '_,
            crate::ComputePipelineHandle,
            wgpu::ComputePipeline,
        > = device.pipeline_resources();

        let mut encoder =
            Some(device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None }));

        #[inline]
        fn create_timestamp_writes<'a>(
            profiler: &'a mut Option<Profiler>,
            op: &CompiledOp,
        ) -> Option<wgpu::ComputePassTimestampWrites<'a>> {
            if let Some(profiler) = profiler {
                let label = format!("{}_{}", op.kernel_key, op.workgroup_count());
                op.tensor_id
                    .map(|id| profiler.create_timestamp_queries(id, label.as_str()))
            } else {
                None
            }
        }

        let compute_step_count = self
            .steps
            .iter()
            .filter(|step| matches!(step, Compiled::Compute(_)))
            .count();

        let grouped_iter = GroupedIter::new(self.steps.iter_mut());

        let mut profiler = step_log_config.as_ref().and_then(|config| {
            config
                .profiling
                .then(|| Profiler::new(device.clone(), compute_step_count as _))
        });

        // let mut debug_steps = vec![];
        let mut gpu_bufs: Option<HashMap<TensorId, PooledGPUBuffer>> = None;

        for group in grouped_iter {
            match group {
                GroupedOp::Compute(mut ops) => {
                    let mut cpass = None;

                    let ops_iter = ops.iter_mut().peekable();
                    for op in ops_iter {
                        let timestamp_writes = create_timestamp_writes(&mut profiler, op);

                        // Existing compute pass; shared
                        if let Some(compute_pass) = cpass.as_mut() {
                            Self::compute_pass_inner(
                                compute_pass,
                                op,
                                &pipeline_resources,
                                &self.gpu_uniform,
                            )?;
                        } else {
                            // Create a new encoder if one doesn't exist already
                            if encoder.is_none() {
                                encoder = Some(device.create_command_encoder(
                                    &wgpu::CommandEncoderDescriptor { label: None },
                                ));
                            }

                            let mut new_cpass = encoder
                                .as_mut()
                                .unwrap()
                                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                                    label: None,
                                    timestamp_writes,
                                })
                                .forget_lifetime();

                            Self::compute_pass_inner(
                                &mut new_cpass,
                                op,
                                &pipeline_resources,
                                &self.gpu_uniform,
                            )?;

                            cpass = Some(new_cpass);
                        }

                        if profiler.is_some() {
                            // In profile mode, we need to create a new cpass for each op
                            cpass = None;
                        }

                        if let Some(debug_buffer) = op.debug_buffer.take() {
                            // Cpass needs to be dropped before we submit the encoder
                            cpass = None;

                            let result_t = self
                                .debug_list
                                .as_ref()
                                .expect("Debug list is not set")
                                .get(&op.tensor_id.expect("Tensor id is not set"))
                                .expect("Tensor not found in debug list");

                            let gpu_storage = result_t.storage();
                            let result_buf = &gpu_storage
                                .as_ref()
                                .ok_or(ExecutionError::DebuggingError("Failed to get result buf."))?
                                .try_gpu()
                                .map_err(|_| {
                                    ExecutionError::DebuggingError("Result buf is not on GPU.")
                                })?
                                .inner;

                            encoder.as_mut().unwrap().copy_buffer_to_buffer(
                                result_buf,
                                0,
                                &debug_buffer,
                                0,
                                result_t.num_bytes() as _,
                            );

                            gpu_bufs
                                .get_or_insert_default()
                                .insert(op.tensor_id.expect("Tensor id is not set"), debug_buffer);
                        }
                    }
                }
                GroupedOp::Copy(op) => {
                    // Create a new encoder if one doesn't exist already
                    if encoder.is_none() {
                        encoder = Some(device.create_command_encoder(
                            &wgpu::CommandEncoderDescriptor { label: None },
                        ));
                    }

                    // Process the copy operation outside of a compute pass.
                    let src = op.src().as_ref();
                    let dst = op.dst().as_ref();
                    let size = op.size();
                    encoder
                        .as_mut()
                        .unwrap()
                        .copy_buffer_to_buffer(src, 0, dst, 0, size);
                }
            }
        }

        if let Some(profiler) = profiler.as_mut() {
            profiler.resolve(encoder.as_mut().unwrap());
        }

        let index = device
            .queue()
            .submit(Some(encoder.take().unwrap().finish()));

        let profiling_entries = if let Some(profiler) = profiler {
            Some(profiler.read_timestamps().await)
        } else {
            None
        };

        Ok((
            index,
            ExecutionResult {
                profiling_entries,
                gpu_bufs,
            },
        ))
    }

    #[cfg(feature = "debug")]
    pub(crate) fn dispatch_debugging(
        &self,
        device: &WgpuDevice,
    ) -> Result<SubmissionIndex, ExecutionError> {
        let pipeline_resources = device.pipeline_resources();
        log::debug!(
            "n steps: {}, n debug_list: {}",
            self.steps.len(),
            self.debug_list.as_ref().map(|d| d.len()).unwrap_or(0)
        );
        if let Some(debug_list) = &self.debug_list {
            assert!(debug_list.len() == self.steps.len());
        }

        let mut last_index = None;
        // Create a peekable enumerated iterator over the steps.
        let mut steps_iter = self.steps.iter().enumerate().peekable();

        while let Some((step_index, step)) = steps_iter.next() {
            match step {
                Compiled::Compute(op) => {
                    let mut compute_group = vec![(step_index, op)];
                    while let Some(&(next_index, next_step)) = steps_iter.peek() {
                        if let Compiled::Compute(next_op) = next_step {
                            compute_group.push((next_index, next_op));
                            steps_iter.next();
                        } else {
                            break;
                        }
                    }

                    for &(step_index, op) in &compute_group {
                        let mut encoder =
                            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: None,
                            });

                        {
                            let mut cpass =
                                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                    label: Some("piston inference pass"),
                                    timestamp_writes: None,
                                });
                            log::debug!(
                                "Bind group layout entries: {} {:?} {:?}",
                                op.kernel_key,
                                op.storage_bind_group_layout_entries
                                    .iter()
                                    .map(|e| e.read_only)
                                    .collect::<Vec<_>>(),
                                &op.storage_groups()[0]
                                    .descriptor()
                                    .entries
                                    .iter()
                                    .map(|e| e.handle.data())
                                    .collect::<Vec<_>>(),
                            );
                            Self::compute_pass_inner(
                                &mut cpass,
                                op,
                                &pipeline_resources,
                                &self.gpu_uniform,
                            )?;
                        }

                        // Process the debugging copy operations associated with each compute op.
                        if let Some(debug_list) = &self.debug_list {
                            let result_t = debug_list[step_index].clone();
                            let size = result_t.size;
                            let gpu_storage = result_t.storage;
                            let result_buf = &gpu_storage
                                .as_ref()
                                .ok_or(ExecutionError::DebuggingError("Failed to get result buf."))?
                                .try_gpu()
                                .map_err(|_| {
                                    ExecutionError::DebuggingError("Result buf is not on GPU.")
                                })?
                                .inner;

                            let input_storage_list = result_t
                                .srcs
                                .iter()
                                .map(|s| {
                                    (
                                        s.storage
                                            .as_ref()
                                            .unwrap()
                                            .try_gpu()
                                            .unwrap()
                                            .inner
                                            .clone(),
                                        s.size,
                                    )
                                })
                                .collect::<Vec<_>>();
                            let debug_input_buffers = op.debug_input_buffers.as_ref().ok_or(
                                ExecutionError::DebuggingError("Failed to get input buffers."),
                            )?;
                            for (i, (buf, size)) in input_storage_list.iter().enumerate() {
                                let debug_input_buffer = debug_input_buffers[i].clone().1;
                                encoder.copy_buffer_to_buffer(
                                    buf,
                                    0,
                                    &debug_input_buffer,
                                    0,
                                    *size as _,
                                );
                            }

                            let (_, debug_buffer) =
                                op.debug_buffer
                                    .as_ref()
                                    .ok_or(ExecutionError::DebuggingError(
                                        "Failed to get debug buffer.",
                                    ))?;
                            encoder.copy_buffer_to_buffer(
                                result_buf,
                                0,
                                debug_buffer,
                                0,
                                size as _,
                            );
                        }

                        let index = device.queue().submit(Some(encoder.finish()));
                        last_index = Some(index);
                    }
                }
                Compiled::Copy(op) => {
                    // Process copy operations individually.
                    let mut encoder = device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                    let src = op.src().as_ref();
                    let dst = op.dst().as_ref();
                    let size = op.size();
                    encoder.copy_buffer_to_buffer(src, 0, dst, 0, size);
                    last_index = Some(device.queue().submit(Some(encoder.finish())));
                }
            }
        }

        if let Some(debug_list) = &self.debug_list {
            // Dump all of our debug results
            for (si, step) in self.steps.iter().enumerate().filter_map(|(si, step)| {
                if let Compiled::Compute(op) = step {
                    Some((si, op))
                } else {
                    None
                }
            }) {
                let d = device.clone();
                let dtype = debug_list[si].dtype;
                let (id, debug_buffer) = step.debug_buffer.clone().unwrap();
                let debug_input_buffers = step.debug_input_buffers.clone().unwrap();
                let alignment = dtype.size_of();
                let kernel_key = step.kernel_key.clone();

                #[allow(clippy::too_many_arguments)]
                #[maybe_async]
                async fn log_fn(
                    si: usize,
                    d: WgpuDevice,
                    debug_buffer: &wgpu::Buffer,
                    debug_input_buffers: RVec<(TensorId, Arc<wgpu::Buffer>)>,
                    kernel_key: KernelKey,
                    alignment: usize,
                    id: TensorId,
                    dtype: DType,
                    output_cpu_bufs: &Arc<RwLock<HashMap<TensorId, CPUBuffer>>>,
                ) {
                    let cpu_buf =
                        wgpu_buffer_to_cpu_buffer(debug_buffer, alignment, None, &d).await;
                    let mut input_bufs = vec![];
                    for (id, buf) in debug_input_buffers.iter() {
                        let cpu_buf = wgpu_buffer_to_cpu_buffer(buf, alignment, None, &d).await;
                        input_bufs.push((id, cpu_buf));
                    }
                    let mut debug_str = format!(
                        "\x1b[1m{} ({:?}): {}\x1b[0m\n {:?}\n\n",
                        si,
                        id,
                        kernel_key,
                        cpu_buf.dump(dtype, (cpu_buf.n_bytes() / 4) <= 8)
                    );

                    for (i, (id, cpu_buf)) in input_bufs.iter().enumerate() {
                        debug_str.push_str(&format!(
                            "\x1b[32;1minput {} ({:?})\x1b[0m: {:?}\n\n",
                            i,
                            id,
                            cpu_buf.dump(dtype, (cpu_buf.n_bytes() / 4) <= 8)
                        ));
                    }

                    output_cpu_bufs.write().insert(id, cpu_buf);

                    log::debug!("{}", debug_str);
                }

                #[cfg(target_arch = "wasm32")]
                {
                    wasm_bindgen_futures::spawn_local(log_fn(
                        si,
                        d,
                        debug_buffer.as_ref(),
                        debug_input_buffers,
                        kernel_key,
                        alignment,
                        id,
                        dtype,
                        self.cpu_bufs.as_ref().unwrap(),
                    ));
                }
                #[cfg(not(target_arch = "wasm32"))]
                {
                    log_fn(
                        si,
                        d,
                        debug_buffer.as_ref(),
                        debug_input_buffers,
                        kernel_key,
                        alignment,
                        id,
                        dtype,
                        self.cpu_bufs.as_ref().unwrap(),
                    );
                }
            }
        }

        Ok(last_index.unwrap())
    }

    #[cfg(feature = "gpu-profiling")]
    pub fn dispatch_operations(
        &self,
        device: &WgpuDevice,
    ) -> Result<SubmissionIndex, ExecutionError> {
        use crate::gpu::Profiler;

        let pipeline_resources = device.pipeline_resources();
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let mut profiler = Profiler::new(device.clone(), self.steps.len() as _);

        // Create a peekable iterator over steps.
        let mut steps_iter = self.steps.iter().peekable();

        while let Some(step) = steps_iter.next() {
            match step {
                Compiled::Compute(op) => {
                    let mut compute_group = vec![op];
                    while let Some(next_step) = steps_iter.peek() {
                        if let Compiled::Compute(next_op) = next_step {
                            compute_group.push(next_op);
                            // consume the op
                            steps_iter.next();
                        } else {
                            break;
                        }
                    }

                    for op in compute_group {
                        let label =
                            format!("{}_{}", op.kernel_key, op.workgroup_count().to_string());
                        let timestamp_writes =
                            Some(profiler.create_timestamp_queries(0, label.as_str()));
                        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: None,
                            timestamp_writes,
                        });
                        cpass.set_pipeline(pipeline_resources.get(op.pipeline_handle())?);

                        for (group_index, bind_group) in op.storage_groups().iter().enumerate() {
                            cpass.set_bind_group(group_index as u32, bind_group, &[]);
                        }

                        let uniform_group_index = op.storage_groups().len() as u32;
                        let uniform_group = self.gpu_uniform.bind_group();
                        cpass.set_bind_group(uniform_group_index, uniform_group, &[op.offset()]);

                        let [x_count, y_count, z_count] = op.workgroup_count().as_slice();
                        cpass.dispatch_workgroups(x_count, y_count, z_count);
                    }
                }
                Compiled::Copy(op) => {
                    // TODO(vinhowe): Decide on the best way to profile these?
                    let mut encoder_copy = device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                    let src = op.src().as_ref();
                    let dst = op.dst().as_ref();
                    let size = op.size();
                    encoder_copy.copy_buffer_to_buffer(src, 0, dst, 0, size);
                    let _ = device.queue().submit(Some(encoder_copy.finish()));
                }
            }
        }

        profiler.resolve(&mut encoder);
        let index = device.queue().submit(Some(encoder.finish()));
        profiler.print_timestamps(true);
        Ok(index)
    }

    pub fn set_storage(&mut self, storage: Vec<Option<Storage>>) {
        self.storage = Some(storage);
    }

    pub fn with_tensors(self, tensors: &Vec<&OpTensor>) -> anyhow::Result<Self> {
        assert_eq!(
            tensors.len(),
            self.storage.as_ref().unwrap().len(),
            "Number of tensors must match number of storage slots"
        );

        // This doesn't work;
        // - We need to update uniforms too
        // - This logic doesn't totally make sense; I'm taking the storage from the existing run and
        //   assigning it to the new tensors... is that fine?
        for (storage, tensor) in self.storage.as_ref().unwrap().iter().zip(tensors) {
            log::debug!("Setting storage for tensor: {:?}", tensor.id());
            #[cfg(feature = "plotting")]
            log::debug!(
                "Storage: {:?}",
                storage.as_ref().map(|s| s.plot_fmt().to_string())
            );
            if let Some(storage) = storage
                && !tensor.resolved()
            {
                tensor.update_storage(storage.clone());
            }
        }

        Ok(Self {
            storage: self.storage,
            steps: self.steps,
            gpu_uniform: self.gpu_uniform,
            debug_list: None,
            #[cfg(feature = "debug")]
            debug_list: None,
            #[cfg(feature = "debug")]
            cpu_bufs: None,
        })
    }
}

enum GroupedOp<'a> {
    Compute(Vec<&'a mut CompiledOp>),
    Copy(&'a mut CompiledCopy),
}

struct GroupedIter<'a, I>
where
    I: Iterator<Item = &'a mut Compiled>,
{
    iter: Peekable<I>,
}

impl<'a, I> GroupedIter<'a, I>
where
    I: Iterator<Item = &'a mut Compiled>,
{
    fn new(iter: I) -> Self {
        Self {
            iter: iter.peekable(),
        }
    }
}

impl<'a, I> Iterator for GroupedIter<'a, I>
where
    I: Iterator<Item = &'a mut Compiled>,
{
    type Item = GroupedOp<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        // Get the first item of the next group.
        let first = self.iter.next()?;
        match first {
            Compiled::Compute(op) => {
                let mut ops = vec![op];
                while let Some(Compiled::Compute(_)) = self.iter.peek() {
                    if let Some(Compiled::Compute(next_op)) = self.iter.next() {
                        ops.push(next_op);
                    }
                }
                Some(GroupedOp::Compute(ops))
            }
            Compiled::Copy(op) => Some(GroupedOp::Copy(op)),
        }
    }
}
