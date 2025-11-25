mod align;
mod buffer_allocator;
mod device;
mod logging;
mod pools;
mod profiler;
mod profiler_context;
mod uniform;
mod wgsl;
mod workload;

pub use align::*;
pub use buffer_allocator::*;
pub use device::*;
pub use logging::*;
pub use pools::*;
pub use profiler::*;
pub use profiler_context::*;
pub use uniform::*;
pub use wgsl::*;
pub use workload::*;
pub const MIN_STORAGE_BUFFER_SIZE: usize = 16;
pub const STORAGE_BUFFER_ALIGN: usize = 256; //TODO: should be a device limit

/// Usages we use everywhere
pub trait BufferUsagesExt {
    fn standard() -> Self;
}

impl BufferUsagesExt for wgpu::BufferUsages {
    fn standard() -> Self {
        Self::COPY_DST | Self::COPY_SRC | Self::STORAGE
    }
}
