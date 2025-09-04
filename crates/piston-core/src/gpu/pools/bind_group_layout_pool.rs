use std::hash::Hash;
#[cfg(feature = "debug")]
use std::hash::Hasher;

use crate::{RVec, gpu::WgpuDevice, rvec};

use super::{StaticResourcePoolReadLockAccessor, static_resource_pool::StaticResourcePool};

pub trait BindGroupLayoutEntryExt {
    fn compute_storage_buffer(binding: u32, read_only: bool) -> Self;
    fn dynamic_uniform_buffer() -> Self;
}

impl BindGroupLayoutEntryExt for wgpu::BindGroupLayoutEntry {
    fn compute_storage_buffer(binding: u32, read_only: bool) -> Self {
        Self {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                min_binding_size: None,
                has_dynamic_offset: false,
            },
            count: None,
        }
    }

    fn dynamic_uniform_buffer() -> Self {
        Self {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                min_binding_size: None,
                has_dynamic_offset: true,
            },
            count: None,
        }
    }
}

slotmap::new_key_type! { pub struct BindGroupLayoutHandle; }

#[cfg(feature = "debug")]
#[derive(Debug, Clone)]
pub struct BindGroupLayoutEntryDescriptor {
    pub entry: wgpu::BindGroupLayoutEntry,
    pub read_only: bool,
}

#[cfg(feature = "debug")]
impl Hash for BindGroupLayoutEntryDescriptor {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.entry.hash(state);
    }
}

#[cfg(feature = "debug")]
impl PartialEq for BindGroupLayoutEntryDescriptor {
    fn eq(&self, other: &Self) -> bool {
        self.entry == other.entry
    }
}

#[cfg(feature = "debug")]
impl Eq for BindGroupLayoutEntryDescriptor {}

#[cfg(feature = "debug")]
impl BindGroupLayoutEntryDescriptor {
    pub fn compute_storage_buffer(binding: u32, read_only: bool) -> Self {
        Self {
            entry: wgpu::BindGroupLayoutEntry::compute_storage_buffer(binding, read_only),
            read_only,
        }
    }

    pub fn dynamic_uniform_buffer() -> Self {
        Self {
            entry: wgpu::BindGroupLayoutEntry::dynamic_uniform_buffer(),
            read_only: true, // Uniform buffers are always read-only from shader perspective
        }
    }
}

#[cfg(not(feature = "debug"))]
pub type BindGroupLayoutEntryDescriptor = wgpu::BindGroupLayoutEntry;

#[derive(Debug, Clone, Hash, PartialEq, Eq, Default)]
pub struct BindGroupLayoutDescriptor {
    pub entries: RVec<BindGroupLayoutEntryDescriptor>,
}

impl BindGroupLayoutDescriptor {
    //Used for unary, binary, ternary (NOT INPLACE)
    fn entries(ro_length: usize) -> RVec<BindGroupLayoutEntryDescriptor> {
        let mut read_only: RVec<BindGroupLayoutEntryDescriptor> = (0..ro_length)
            .map(|idx| BindGroupLayoutEntryDescriptor::compute_storage_buffer(idx as u32, true))
            .collect();
        read_only.push(BindGroupLayoutEntryDescriptor::compute_storage_buffer(
            ro_length as u32,
            false,
        ));
        read_only
    }

    pub fn unary() -> Self {
        Self {
            entries: Self::entries(1),
        }
    }

    pub fn unary_inplace() -> Self {
        Self {
            entries: rvec![BindGroupLayoutEntryDescriptor::compute_storage_buffer(
                0, false
            )],
        }
    }

    pub fn binary() -> Self {
        Self {
            entries: Self::entries(2),
        }
    }

    pub fn binary_inplace() -> Self {
        Self {
            entries: rvec![
                BindGroupLayoutEntryDescriptor::compute_storage_buffer(0, false),
                BindGroupLayoutEntryDescriptor::compute_storage_buffer(1, true)
            ],
        }
    }

    pub fn ternary() -> Self {
        Self {
            entries: Self::entries(3),
        }
    }

    pub fn ternary_inplace() -> Self {
        Self {
            entries: rvec![
                BindGroupLayoutEntryDescriptor::compute_storage_buffer(0, false),
                BindGroupLayoutEntryDescriptor::compute_storage_buffer(1, true),
                BindGroupLayoutEntryDescriptor::compute_storage_buffer(2, true)
            ],
        }
    }

    pub fn nthary(ro: usize) -> Self {
        Self {
            entries: Self::entries(ro),
        }
    }

    pub fn uniform() -> Self {
        Self {
            entries: rvec![BindGroupLayoutEntryDescriptor::dynamic_uniform_buffer()],
        }
    }
}

pub struct BindGroupLayoutPool {
    inner:
        StaticResourcePool<BindGroupLayoutHandle, BindGroupLayoutDescriptor, wgpu::BindGroupLayout>,
}

impl Default for BindGroupLayoutPool {
    fn default() -> Self {
        Self::new()
    }
}

impl BindGroupLayoutPool {
    pub fn new() -> Self {
        Self {
            inner: StaticResourcePool::default(),
        }
    }
}

impl BindGroupLayoutPool {
    pub fn get_or_create(
        &self,
        descriptor: &BindGroupLayoutDescriptor,
        device: &WgpuDevice,
    ) -> BindGroupLayoutHandle {
        self.inner.get_or_create(descriptor, |desc| {
            #[cfg(feature = "debug")]
            let entries: Vec<_> = desc.entries.iter().map(|e| e.entry).collect();
            #[cfg(not(feature = "debug"))]
            let entries: Vec<_> = desc.entries.to_vec();
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &entries,
            })
        })
    }

    /// Locks the resource pool for resolving handles.
    ///
    /// While it is locked, no new resources can be added.
    pub fn resources(
        &self,
    ) -> StaticResourcePoolReadLockAccessor<'_, BindGroupLayoutHandle, wgpu::BindGroupLayout> {
        self.inner.resources()
    }
}
