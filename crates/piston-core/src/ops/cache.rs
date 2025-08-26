use derive_new::new;
use encase::ShaderType;
use glam::UVec4;
use half::f16;
use inline_wgsl::wgsl;
use piston_macros::{IrFields, WgslMetadata};

use crate::{
    Array, BindGroupLayoutEntryDescriptor, BindGroupLayoutEntryExt, BindingMode, BuiltIn, DType,
    GPUOperation, Kernel, KernelElement, KernelRenderable, KernelSource, OpGuards, OpTensor,
    Operation, OperationError, RVec, Scalar, Shape, StorageView, Stride, Vec2, Vec4,
    WgslKernelBuilder, WgslPrimitive, WorkgroupSize, Workload, gpu::BindGroupLayoutDescriptor,
    rvec,
};

/// # Cache
///
/// Custom operator used for KV caching. Custom operator to support quantized KV caching.
///
/// Takes in 3 arguments:
/// 1. Cache, large partially filled tensors. E.g [1, 512, 1024], with [1, 5, 1024] filled.
/// 2. Source, new K or V tensor, e.g [1, 1, 1024]
/// 3. offset, where to start the write in the cache tensor, e.g [1, 5, 1024], [1, 1, 1024], offset = 5 -> [1, 6, 1024]
#[derive(new, Debug, Clone, IrFields)]
pub struct Cache {
    cache: OpTensor,
    source: OpTensor,
    dim: usize,
    offset: usize,
}

impl KernelRenderable for CacheKernels {
    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        inplace: bool,
    ) -> Result<(), OperationError> {
        if inplace {
            return Err(OperationError::InplaceError(self.kernel_name().to_string()));
        }

        builder.register_storage("C", BindingMode::ReadWrite, Array::<P>::default());
        builder.register_storage("S", BindingMode::ReadOnly, Array::<P>::default());
        builder.register_storage("D", BindingMode::ReadWrite, Array::<P>::default());

        builder.register_uniform();
        Ok(())
    }

    fn render<P: WgslPrimitive>(
        &self,
        inplace: bool,
        dst: &OpTensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let device = dst.device().try_gpu()?;
        let mut kernel_builder = WgslKernelBuilder::new(
            workgroup_size.clone(),
            rvec![
                BuiltIn::WorkgroupId,
                BuiltIn::LocalInvocationIndex,
                BuiltIn::NumWorkgroups,
            ],
            device.compute_features().clone(),
        );
        self.register_bindings::<P>(&mut kernel_builder, inplace)?;

        kernel_builder.render_metadata(&self.metadata(dst, &self.kernel_element(dst))?);
        kernel_builder.write_offset_to_index();
        kernel_builder.write_index_to_offset();

        kernel_builder.write_main(wgsl! {
            //Dispatch 1 thread per output element
            //dst_offset is index into the output buffer (1D)
            let x_offset = workgroup_id.x * 64u;
            let dst_offset = (workgroup_id.y * num_workgroups.x * 64u) + x_offset + local_invocation_index;
            if (dst_offset >= metadata.dst_numel) {
                return;
            }
            //Convert 1D offset into 4D index
            var dst_index = offsetToNdIndex(dst_offset, metadata.dst_stride);

            let dim = metadata.dim;
            if (dst_index[dim] < metadata.cum0) {
                //Inside cache, just copy from cache to DST
                let src_offset = ndIndexToOffset(dst_index, metadata.cache_stride);
                D[dst_offset] = C[src_offset];
                return;
            }

            if (dst_index[dim] < metadata.cum1) {
                //Inside src, copy from src to cache and then to DST
                let cache_offset = ndIndexToOffset(dst_index, metadata.cache_stride);
                dst_index[dim] -= metadata.cum0;
                let src_offset = ndIndexToOffset(dst_index, metadata.src_stride);
                let val = S[src_offset];
                C[cache_offset] = val;
                D[dst_offset] = val;
                return;
            }
        });

        Ok(kernel_builder.build()?)
    }
}

#[derive(Debug, derive_new::new, ShaderType, WgslMetadata)]
pub struct CacheMeta {
    cache_stride: glam::UVec4,
    src_stride: glam::UVec4,
    dst_stride: glam::UVec4,
    dst_numel: u32,
    cum0: u32,
    cum1: u32,
    dim: u32,
}

impl OpGuards for Cache {
    fn check_shapes(&self) {
        assert!(self.cache.dim() >= 3);
        assert!(self.offset <= self.cache.shape()[self.dim]);
    }

    fn check_dtypes(&self) {
        assert_eq!(self.cache.dtype(), self.source.dtype());
    }
}

impl Operation for Cache {
    fn name(&self) -> &'static str {
        "Cache"
    }

    #[inline]
    fn srcs(&self) -> RVec<&OpTensor> {
        rvec![&self.cache, &self.source]
    }

    fn compute_view(&self) -> Result<StorageView, OperationError> {
        let mut result_shape = self.cache.shape().clone();
        result_shape[self.dim] = self.offset + self.source.shape()[self.dim];
        let result_stride = Stride::from(&result_shape);
        Ok(StorageView::new(
            result_shape,
            self.cache.dtype(),
            result_stride,
        ))
    }

    fn supports_inplace(&self) -> bool {
        false
    }
}

impl GPUOperation for Cache {
    type KernelEnum = CacheKernels;

    fn select_kernel(&self) -> Self::KernelEnum {
        CacheKernels::Standard(self.clone())
    }
}

pub enum CacheKernels {
    Standard(Cache),
}

impl Kernel for CacheKernels {
    type Metadata = CacheMeta;

    fn storage_bind_group_layout(
        &self,
        _: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        // Custom layout because of funky mutability requirements
        Ok(BindGroupLayoutDescriptor {
            entries: rvec![
                BindGroupLayoutEntryDescriptor::compute_storage_buffer(0, false),
                BindGroupLayoutEntryDescriptor::compute_storage_buffer(1, true),
                BindGroupLayoutEntryDescriptor::compute_storage_buffer(2, false)
            ],
        })
    }

    fn kernel_name(&self) -> String {
        match self {
            CacheKernels::Standard(_) => "cache".to_string(),
        }
    }

    fn metadata(
        &self,
        dst: &OpTensor,
        _: &KernelElement,
    ) -> Result<Self::Metadata, OperationError> {
        let CacheKernels::Standard(inner) = self;

        let original_rank = inner.cache.dim();
        let promotion = 4 - original_rank;
        let promoted_dim = inner.dim + promotion;

        let cache_shape = Shape::promote(inner.cache.shape().clone(), 4);
        let cache_stride = Stride::from(&cache_shape);

        let source_shape = Shape::promote(inner.source.shape().clone(), 4);
        let source_stride = Stride::from(&source_shape);

        let dst_shape = Shape::promote(dst.shape().clone(), 4);
        let dst_stride = Stride::from(&dst_shape);

        let cum0 = inner.offset as u32;
        let cum1 = cum0 + source_shape[promoted_dim] as u32;

        Ok(CacheMeta {
            cache_stride: UVec4::from(&cache_stride),
            src_stride: UVec4::from(&source_stride),
            dst_stride: UVec4::from(&dst_stride),
            dst_numel: dst_shape.numel() as u32,
            cum0,
            cum1,
            dim: promoted_dim as u32,
        })
    }

    fn kernel_element(&self, _dst: &OpTensor) -> KernelElement {
        KernelElement::Scalar
    }

    fn calculate_dispatch(&self, dst: &OpTensor) -> Result<Workload, OperationError> {
        Ok(Workload::std(dst.shape().numel(), self.kernel_element(dst)))
    }

    fn build_kernel(
        &self,
        inplace: bool,
        dst: &OpTensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let kernel_element = self.kernel_element(dst);
        match (dst.dtype(), &kernel_element) {
            (DType::F32, KernelElement::Scalar) => {
                self.render::<Scalar<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F32, KernelElement::Vec2) => {
                self.render::<Vec2<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F32, KernelElement::Vec4) => {
                self.render::<Vec4<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Scalar) => {
                self.render::<Scalar<f16>>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Vec2) => {
                self.render::<Vec2<f16>>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Vec4) => {
                self.render::<Vec4<f16>>(inplace, dst, workgroup_size)
            }
            _ => Err(OperationError::CompileError(format!(
                "Unsupported dtype {:?} or kernel element {:?}",
                dst.dtype(),
                kernel_element
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Device, DeviceRequest, cat, randn, rvec, zeros};

    #[test]
    fn test_cache() -> anyhow::Result<()> {
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        let populated = 2;
        //Create cache with 2 populated entries, and 14 blank entries
        let mut dst0 = randn((1, 2, populated, 16), None, None, Default::default())?;
        println!(
            "PREVIOUS CACHE\n {:?}\n",
            dst0.inner().read().to_ndarray_view::<f32>()
        );
        dst0 = dst0.to(&device)?;
        let dst1 = zeros((1, 2, 4, 16), Default::default())?;
        let cur_cache = cat(rvec![dst0.clone(), dst1], 2)?;

        //This is the k or v vector we write
        let mut src = randn((1, 2, 1, 16), None, None, Default::default())?;
        println!("SRC \n {:?}\n", src.inner().read().to_ndarray_view::<f32>());
        src = src.to(&device)?;

        //The result should be the concatenation of the cache and the source
        let ground_truth = cat(rvec![dst0.clone(), src.clone()], 2)?.to(&Device::CPU)?;

        let dim = 2;
        let b = cur_cache.clone().cache(src, dim, populated)?;

        let cur_cache_cpu = cur_cache.to(&Device::CPU)?;
        println!(
            "CACHE RESULT \n{:?}\n",
            cur_cache_cpu.inner().read().to_ndarray_view::<f32>()
        );

        let result = b.to(&Device::CPU)?;
        println!(
            "RESULT \n{:?}",
            result.inner().read().to_ndarray_view::<f32>()
        );

        result.all_close(&ground_truth, 1e-5, 1e-5).unwrap();
        Ok(())
    }
}
