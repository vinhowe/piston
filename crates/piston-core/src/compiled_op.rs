#[cfg(feature = "debug")]
use crate::gpu::BindGroupLayoutEntryDescriptor;
#[cfg(feature = "debug")]
use crate::TensorId;
use crate::{drvec, rvec, KernelKey, OperationError, PooledGPUBuffer, RVec, OpTensor};
use crate::{
    gpu::{
        BindGroupDescriptor, BindGroupLayoutHandle, ComputePipelineHandle, GpuBindGroup,
        WgpuDevice, WorkgroupCount,
    },
    TensorId,
};
use derive_new::new;
use std::sync::Arc;
use wgpu::DynamicOffset;

#[derive(Debug, new)]
pub struct CompiledCopy {
    src: Arc<PooledGPUBuffer>,
    dst: Arc<PooledGPUBuffer>,
    size: u64,
}

impl CompiledCopy {
    pub fn src(&self) -> &Arc<PooledGPUBuffer> {
        &self.src
    }

    pub fn dst(&self) -> &Arc<PooledGPUBuffer> {
        &self.dst
    }

    pub fn size(&self) -> u64 {
        self.size
    }
}

//Compiled op represents a single kernel invocation
#[derive(Debug, new)]
pub struct CompiledOp {
    pub(crate) pipeline_handle: ComputePipelineHandle,
    pub(crate) workgroup_count: WorkgroupCount,
    pub(crate) storage_groups: RVec<GpuBindGroup>,
    pub(crate) offset: DynamicOffset, //offset into the metadata uniform buffer
    pub kernel_key: KernelKey,
    // Mapping between tensor and compiled op is not necessarily 1:1—for example: things like AMP
    // likely insert casts between tensor operations.
    pub tensor_id: Option<TensorId>,
    #[cfg(not(feature = "debug"))]
    pub debug_buffer: Option<PooledGPUBuffer>,
    #[cfg(feature = "debug")]
    pub debug_buffer: Option<(TensorId, Arc<wgpu::Buffer>)>,
    #[cfg(feature = "debug")]
    pub debug_input_buffers: Option<RVec<(TensorId, Arc<wgpu::Buffer>)>>,
    #[cfg(feature = "debug")]
    pub storage_bind_group_layout_entries: RVec<BindGroupLayoutEntryDescriptor>,
}

#[derive(Debug)]
pub enum Compiled {
    Copy(CompiledCopy),
    Compute(CompiledOp),
}

impl CompiledOp {
    const MAX_BINDINGS_PER_GROUP: usize = 8;

    pub fn create_storage_bind_groups(
        srcs: &[&OpTensor],
        dst: &OpTensor,
        bind_group_layouts: RVec<BindGroupLayoutHandle>,
        device: &WgpuDevice,
        inplace: bool,
    ) -> Result<RVec<GpuBindGroup>, OperationError> {
        let mut bind_group_entries = drvec![];

        for tensor in srcs.iter() {
            bind_group_entries.append(&mut tensor.bind_group_entries());
        }

        if !inplace {
            bind_group_entries.append(&mut dst.bind_group_entries());
        }

        let mut storage_groups = rvec![];
        for (group_index, bind_group_layout) in bind_group_layouts.iter().enumerate() {
            let group_range = Self::group_range(group_index, bind_group_entries.len());
            let entries = bind_group_entries[group_range].into();
            let layout = *bind_group_layout;

            let bg = device.get_or_create_bind_group(&BindGroupDescriptor { entries, layout })?;
            storage_groups.push(bg);
        }
        Ok(storage_groups)
    }

    /// Determines which bindings belong to which bind group
    fn group_range(group_index: usize, binding_counter: usize) -> std::ops::Range<usize> {
        let group_end = usize::min(
            (group_index + 1) * Self::MAX_BINDINGS_PER_GROUP,
            binding_counter,
        );
        group_index * Self::MAX_BINDINGS_PER_GROUP..group_end
    }

    pub fn workgroup_count(&self) -> &WorkgroupCount {
        &self.workgroup_count
    }

    pub fn offset(&self) -> DynamicOffset {
        self.offset
    }

    pub fn storage_groups(&self) -> &RVec<GpuBindGroup> {
        &self.storage_groups
    }

    pub fn pipeline_handle(&self) -> ComputePipelineHandle {
        self.pipeline_handle
    }
}
