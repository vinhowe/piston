use super::TensorUsageRecord;
use crate::{
    gpu::{
        BufferDescriptor, BufferPool, BufferUsagesExt, CpuUniform, GpuBufferHandle,
        PooledGPUBuffer, TensorUsageRecords, WgpuDevice, UNIFORM_ALIGN,
    },
    DeviceError, GpuCompileKey, Tensor, TensorId,
};
use crate::{HashMap, LazyOp};
use parking_lot::RwLock;
use std::num::NonZero;
use std::{borrow::Cow, collections::BTreeMap};
use wgpu::BufferUsages;

#[derive(Clone, Debug, thiserror::Error)]
pub enum AllocatorError {
    #[error("Buffer not found")]
    BufferNotFound,
}

pub struct BufferAllocator {
    pool: RwLock<BufferPool>,
}

impl Default for BufferAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl BufferAllocator {
    pub fn new() -> Self {
        Self {
            pool: BufferPool::new().into(),
        }
    }

    pub fn begin_pass(&self, pass_index: u64) {
        self.pool.write().begin_pass(pass_index);
    }

    pub fn get(&self, handle: GpuBufferHandle) -> PooledGPUBuffer {
        self.pool.read().get(handle).unwrap()
    }

    pub fn create_buffer(
        &self,
        desc: &BufferDescriptor,
        device: &WgpuDevice,
        immediate: bool,
    ) -> PooledGPUBuffer {
        self.pool.write().get_or_create(desc, device, immediate)
    }

    pub fn create_buffer_init(
        &self,
        desc: &BufferDescriptor,
        contents: Cow<'_, [u8]>,
        device: &WgpuDevice,
    ) -> PooledGPUBuffer {
        //cannot write content to a buffer if it is less than 4 bytes
        let contents = if contents.len() < wgpu::COPY_BUFFER_ALIGNMENT as _ {
            let mut min_contents = vec![0u8; wgpu::COPY_BUFFER_ALIGNMENT as _];
            min_contents[..contents.len()].copy_from_slice(contents.as_ref());
            Cow::Owned(min_contents)
        } else {
            contents
        };

        let buf = self.pool.write().get_or_create(desc, device, false);
        let mut buffer_view = device.queue().write_buffer_with(
            &buf.inner,
            0,
            NonZero::try_from(contents.len() as u64).unwrap(),
        );
        buffer_view
            .as_mut()
            .unwrap()
            .copy_from_slice(contents.as_ref());
        drop(buffer_view);
        buf
    }

    pub fn create_uniform_init(&self, uniform: CpuUniform, device: &WgpuDevice) -> PooledGPUBuffer {
        let mut uniform = uniform.into_inner();
        uniform.resize(
            uniform.len() + UNIFORM_ALIGN - uniform.len() % UNIFORM_ALIGN,
            0u8,
        );
        let desc = BufferDescriptor::new(
            uniform.len() as _,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            false,
        );

        let resource = self.pool.write().get_or_create(&desc, device, false);
        let mut buffer_view = device.queue().write_buffer_with(
            &resource.inner,
            0,
            NonZero::try_from(uniform.len() as u64).unwrap(),
        );
        buffer_view
            .as_mut()
            .unwrap()
            .copy_from_slice(uniform.as_slice());
        drop(buffer_view);
        resource
    }

    /// # Graph memory allocation
    ///
    /// Greedy algorithm, that takes the first buffer larger than the request
    /// In future, since we know the entire graph and sizes, we can
    /// do better.
    fn graph_allocate(
        &self,
        descriptor: BufferDescriptor,
        free: &mut Vec<PooledGPUBuffer>,
        device: &WgpuDevice,
    ) -> PooledGPUBuffer {
        let required_size = descriptor.size as _;
        let mut closest_index = None;
        let mut closest_size_diff: Option<usize> = None;
        for (idx, buffer) in free.iter().enumerate() {
            let current_size = buffer.descriptor.size as _;
            if current_size >= required_size {
                let size_diff = usize::abs_diff(current_size, required_size);

                if closest_size_diff.is_none_or(|diff| size_diff < diff) {
                    closest_index = Some(idx);
                    closest_size_diff = Some(size_diff);
                }
            }
        }

        if std::env::var("RATCHET_DEBUG").is_ok() {
            return self.create_buffer(&descriptor, device, true);
        }

        match closest_index {
            Some(idx) => free.remove(idx),
            None => self.create_buffer(&descriptor, device, false),
        }
    }

    /// # Inplace operations
    ///
    /// If an operation supports inplace, we need to "lease" the buffer
    /// from the actual source (i.e the first non-inplace operation).
    ///
    /// We traverse upwards checking how far the inplace chain goes. This is done by checking:
    /// 1. If the operation has any sources (i.e it's not a constant)
    /// 2. If the operation has an inplace kernel available
    /// 3. If our PARENT (i.e the buffer we are about to apply an operation to) has multiple consumers
    ///    if it has multiple consumers, you can't inplace
    fn determine_tensor_source<'a>(
        source: &'a Tensor,
        gpu_compile_keys: &HashMap<TensorId, GpuCompileKey>,
    ) -> &'a Tensor {
        let old_source_id = source.id();
        let mut candidate = source;
        log::trace!("Determining source for {:?}", old_source_id);

        // If we start on a view, we need to iterate unconditionally
        // until we find a non-view operation
        while let LazyOp::View(_) = candidate.op() {
            log::trace!("Stepping through view: {:?}", candidate.id());
            candidate = candidate.op().srcs().first().unwrap();
        }

        let mut true_source = candidate;

        loop {
            if candidate.op().srcs().is_empty()
                || !true_source.op().supports_inplace()
                || candidate.is_variable()
            {
                //If no sources, we are at the root
                //Or if the operation doesn't support inplace
                break;
            }
            //TODO: operations should define their "inplace" source
            //doesn't necessarily have to be the zeroth
            let to_modify = candidate.op().srcs()[0];
            log::trace!("To modify source: {:?}", to_modify.id());

            if gpu_compile_keys
                .get(&candidate.id())
                .map(|key| !key.can_inplace())
                .unwrap_or_else(|| !matches!(candidate.op(), LazyOp::View(_)))
            {
                break;
            }
            if !matches!(to_modify.op(), LazyOp::View(_)) {
                log::trace!("Found non-view source: {:?}", to_modify.id());
                true_source = to_modify;
            }
            candidate = to_modify;
            log::trace!("Candidate: {:?}", candidate.id());
        }

        // If true source is a view, panic
        if let LazyOp::View(_) = true_source.op() {
            log::warn!("True source is a view: {:?}", true_source.id());
        }

        log::trace!(
            "Traversed {:?} to true source: {:?}",
            old_source_id,
            true_source.id()
        );
        true_source
    }

    //To calculate the tensor usage records, we do the following:
    //1. Traverse topologically sorted graph in reverse order
    //2. When we encounter the last consumer of a tensor, we start recording the interval.
    //3. When we encounter the producer of a tensor, we stop recording the interval.
    fn calculate_usage_records(
        execution_order: &[&Tensor],
        gpu_compile_keys: &HashMap<TensorId, GpuCompileKey>,
    ) -> HashMap<TensorId, TensorUsageRecord> {
        let mut records =
            HashMap::with_capacity_and_hasher(execution_order.len(), Default::default());
        let topo_len = execution_order.len() - 1;
        for (iter, t) in execution_order.iter().rev().enumerate() {
            if t.resolved() {
                continue;
            }

            for source in t.op().srcs() {
                if source.resolved() {
                    continue;
                }
                let true_source = Self::determine_tensor_source(source, gpu_compile_keys);
                records
                    .entry(true_source.id())
                    .or_insert_with(|| TensorUsageRecord {
                        id: None,
                        producer: None,
                        is_variable: None,
                        last_consumer: topo_len - iter,
                        last_consumer_id: t.id(),
                        size: true_source.num_bytes(),
                    });
            }

            if let Some(record) = records.get_mut(&t.id()) {
                record.id = Some(t.id());
                record.producer = Some(topo_len - iter);
                record.is_variable = Some(t.is_variable());
            }
        }

        //filter records with no producer
        //TODO: Warning: could be a bug here
        records.retain(|_, v| v.producer.is_some());
        records
    }

    //https://arxiv.org/pdf/2001.03288.pdf + inplace support
    //Takes in const assignments as inplace may be performed on constants
    pub fn greedy_by_size(
        &self,
        execution_order: &[&Tensor],
        output_tensors: &BTreeMap<TensorId, &Tensor>,
        assignments: &mut HashMap<TensorId, PooledGPUBuffer>,
        gpu_compile_keys: &HashMap<TensorId, GpuCompileKey>,
        use_shared_buffers: bool,
        device: &WgpuDevice,
    ) -> Result<(), DeviceError> {
        let record_map = Self::calculate_usage_records(execution_order, gpu_compile_keys);
        let records = TensorUsageRecords::from(record_map);
        let mut shared_objects: Vec<PooledGPUBuffer> = Vec::with_capacity(records.0.len());

        for record in records.0.iter() {
            let should_be_shared = use_shared_buffers
                && !(record.is_variable.unwrap_or(false)
                    || output_tensors.get(&record.last_consumer_id).is_some());

            let mut best_obj = None;

            if should_be_shared {
                let record_producer = record.producer.unwrap();
                for obj in shared_objects.iter() {
                    let mut suitable = true;
                    for inner_r in records.0.iter() {
                        let max_first = std::cmp::max(record_producer, inner_r.producer.unwrap());
                        let min_last = std::cmp::min(record.last_consumer, inner_r.last_consumer);
                        if max_first <= min_last
                            && assignments.get(&inner_r.id.unwrap()) == Some(obj)
                        {
                            suitable = false;
                            break;
                        }
                    }
                    if suitable {
                        // log::debug!("Suitable for {:?}: {:?}", record.id.unwrap(), obj);
                        best_obj = Some(obj);
                    }
                }
            }

            if let Some(obj) = best_obj {
                assignments.insert(record.id.unwrap(), (*obj).clone());
            } else {
                //let rounded_size = (record.size - 1).next_power_of_two();
                let rounded_size = record.size;
                let buf = self.create_buffer(
                    &BufferDescriptor::new(rounded_size as _, BufferUsages::standard(), false),
                    device,
                    false,
                );
                if should_be_shared {
                    shared_objects.push(buf.clone());
                }
                assignments.insert(record.id.unwrap(), buf);
            }
        }

        //Loop through and add inplace assignments
        for t in execution_order.iter() {
            if t.resolved() {
                continue;
            }
            for source in t.op().srcs() {
                let true_source = Self::determine_tensor_source(source, gpu_compile_keys);
                if true_source.id() != source.id() {
                    if let Some(buf) = assignments.get(&true_source.id()) {
                        assignments.insert(source.id(), buf.clone());
                    }
                }
            }
        }

        //We use `immediate` = false here in create_buffer
        //and submit the queue after all allocations are done.
        // device.queue().submit(None);
        // device.poll(wgpu::Maintain::Wait);
        Ok(())
    }

    /// # Graph memory allocation
    ///
    /// Simple greedy algorithm
    /// 1. Iterate over all tensors in reverse order (leaf -> root)
    /// 2. For each tensor, loop through it's input values.
    ///     a. Assign a buffer for each input value, if it is not already assigned
    ///     b. If the input value is an inplace operation, traverse upwards until we find
    ///        the "true" buffer source (i.e the first non-inplace operation).
    /// 3. We release our **output** buffer, because the value is no longer needed,
    ///    and earlier tensors can use it.
    pub fn allocate_cfg(
        &self,
        execution_order: &[&Tensor],
        output_tensors: &BTreeMap<TensorId, &Tensor>,
        gpu_compile_keys: &HashMap<TensorId, GpuCompileKey>,
        use_shared_buffers: bool,
        device: &WgpuDevice,
    ) -> Result<HashMap<TensorId, PooledGPUBuffer>, DeviceError> {
        let mut free = Vec::with_capacity(execution_order.len()); //TODO: switch to BTreeMap
        let mut assignments =
            HashMap::with_capacity_and_hasher(execution_order.len(), Default::default());
        //Assignments already needs all of the constants in it.
        for t in execution_order.iter().rev().filter(|t| t.resolved()) {
            //Consts are immediately resolved
            let storage_guard = t.storage();
            let pooled = storage_guard
                .as_ref()
                .ok_or(AllocatorError::BufferNotFound)?
                .try_gpu()?
                .inner
                .clone();
            assignments.insert(t.id(), pooled);
        }

        //Allocate intermediates
        self.greedy_by_size(
            execution_order,
            output_tensors,
            &mut assignments,
            gpu_compile_keys,
            use_shared_buffers,
            device,
        )?;

        //The output tensors are a special case.
        //We know we need an allocation for the output.
        //We traverse upwards until we find the first non-inplace operation, and use it's buffer.
        //It's also handy to treat output as different, as we can handle getting data back to CPU
        //more efficiently in future.
        for output in output_tensors.values() {
            log::debug!("Allocating output: {:?}", output.id());
            let output_source = Self::determine_tensor_source(output, gpu_compile_keys);
            let output_buffer = assignments
                .get(&output_source.id())
                .cloned()
                .unwrap_or_else(|| {
                    self.graph_allocate(
                        BufferDescriptor::new(
                            output_source.num_bytes() as _,
                            BufferUsages::standard(),
                            false,
                        ),
                        &mut free,
                        device,
                    )
                });
            assignments.insert(output.id(), output_buffer);
        }

        #[cfg(debug_assertions)]
        {
            let mut output_allocations = BTreeMap::new();
            for t in execution_order.iter() {
                if let Some(allocation) = assignments.get(&t.id()) {
                    if output_tensors.contains_key(&t.id()) {
                        output_allocations.insert(allocation.global_id(), t.id());
                    } else if let Some(output_id) = output_allocations.get(&allocation.global_id())
                    {
                        panic!(
                            "Allocation {:?} used by output tensor {:?} was reused by tensor {:?}",
                            allocation.global_id(),
                            output_id,
                            t.id()
                        );
                    }
                }
            }
        }

        log::debug!(
            "Total bytes allocated: {}kb",
            self.pool.read().total_gpu_size_in_bytes() / 1024,
        );
        log::debug!(
            "Total buffers allocated: {}",
            self.pool.read().num_resources()
        );

        Ok(assignments)
    }

    pub fn usage_bytes(&self) -> u64 {
        self.pool.read().total_gpu_size_in_bytes()
    }
}
