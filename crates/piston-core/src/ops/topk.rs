use derive_new::new;
use half::f16;
use inline_wgsl::wgsl;
use piston_macros::IrFields;

use crate::{
    Array, BindingMode, BuiltIn, DType, GPUOperation, Kernel, KernelElement, KernelRenderable,
    KernelSource, OpGuards, OpTensor, Operation, OperationError, RVec, Scalar, Shape, StorageView,
    Stride, WgslKernelBuilder, WgslPrimitive, WorkgroupSize, Workload,
    gpu::BindGroupLayoutDescriptor,
};

const MAX_TOPK: usize = 256;

#[derive(new, Debug, Clone, IrFields)]
pub struct TopK {
    pub input: OpTensor,
    pub k: usize,
    pub dim: usize,
    pub largest: bool,
    pub sorted: bool,
}

impl OpGuards for TopK {
    fn check_shapes(&self) {
        let rank = self.input.dim();
        assert!(rank > 0, "topk: input must have at least 1 dimension");
        assert!(self.dim < rank, "topk: dim out of range");
        assert!(self.k > 0, "topk: k must be > 0");
        assert!(
            self.k <= self.input.shape()[self.dim],
            "topk: k must be <= dim size"
        );
    }

    fn check_dtypes(&self) {
        let dt = self.input.dtype();
        assert!(
            dt.is_float() || matches!(dt, DType::I32),
            "topk: only F32/F16/I32 supported"
        );
        assert!(
            self.k <= MAX_TOPK,
            "topk: k={} exceeds MAX_TOPK={} (temporary limitation)",
            self.k,
            MAX_TOPK
        );
        assert!(!self.sorted, "topk: sorted=true not implemented");
    }
}

impl Operation for TopK {
    fn name(&self) -> &'static str {
        "TopK"
    }

    fn compute_view(&self) -> Result<StorageView, OperationError> {
        // Output are indices along dim: replace dim size with k, dtype I32
        let mut out_shape = self.input.shape().to_vec();
        out_shape[self.dim] = self.k;
        let out_shape = Shape::from(out_shape);
        let stride = Stride::from(&out_shape);
        Ok(StorageView::new(out_shape, DType::I32, stride))
    }

    #[inline]
    fn srcs(&self) -> RVec<&OpTensor> {
        crate::rvec![&self.input]
    }

    fn supports_inplace(&self) -> bool {
        false
    }
}

pub enum TopKKernels {
    Standard(TopK),
}

impl GPUOperation for TopK {
    type KernelEnum = TopKKernels;

    fn select_kernel(&self) -> Self::KernelEnum {
        TopKKernels::Standard(self.clone())
    }
}

#[derive(Debug, derive_new::new, encase::ShaderType, piston_macros::WgslMetadata)]
pub struct TopKMeta {
    rank: u32,
    dim: u32,
    k: u32,
    num_slices: u32,
    shape: glam::UVec4,
    stride: glam::UVec4,
    out_stride: glam::UVec4,
}

impl Kernel for TopKKernels {
    type Metadata = TopKMeta;

    fn kernel_name(&self) -> String {
        match self {
            TopKKernels::Standard(inner) => if inner.largest {
                "topk_largest"
            } else {
                "topk_smallest"
            }
            .to_string(),
        }
    }

    fn kernel_element(&self, _dst: &OpTensor) -> KernelElement {
        KernelElement::Scalar
    }

    fn calculate_dispatch(&self, dst: &OpTensor) -> Result<Workload, OperationError> {
        // One invocation per slice (all dims except target)
        let TopKKernels::Standard(inner) = self;
        let total_slices = dst.shape().numel() / inner.k;
        Ok(Workload::std(total_slices, KernelElement::Scalar))
    }

    fn storage_bind_group_layout(
        &self,
        inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        if inplace {
            return Err(OperationError::InplaceError(
                "TopK cannot be done in place".to_string(),
            ));
        }
        Ok(BindGroupLayoutDescriptor::unary())
    }

    fn metadata(
        &self,
        dst: &OpTensor,
        _: &KernelElement,
    ) -> Result<Self::Metadata, OperationError> {
        let TopKKernels::Standard(inner) = self;
        let rank = inner.input.dim() as u32;
        let mut shape = [1u32; 4];
        let mut stride = [0u32; 4];
        for (i, &d) in inner.input.shape().iter().enumerate() {
            shape[i] = d as u32;
            stride[i] = inner.input.stride()[i] as u32;
        }
        let mut out_stride = [0u32; 4];
        for (i, &s) in dst.stride().iter().enumerate() {
            out_stride[i] = s as u32;
        }
        let num_slices = (inner.input.shape().numel() / inner.input.shape()[inner.dim]) as u32;
        Ok(TopKMeta {
            rank,
            dim: inner.dim as u32,
            k: inner.k as u32,
            num_slices,
            shape: shape.into(),
            stride: stride.into(),
            out_stride: out_stride.into(),
        })
    }

    fn build_kernel(
        &self,
        inplace: bool,
        dst: &OpTensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let TopKKernels::Standard(inner) = self;
        match inner.input.dtype() {
            DType::F32 => self.render::<Scalar<f32>>(inplace, dst, workgroup_size),
            DType::F16 => self.render::<Scalar<f16>>(inplace, dst, workgroup_size),
            DType::I32 => self.render::<Scalar<i32>>(inplace, dst, workgroup_size),
            _ => Err(OperationError::CompileError(
                "TopK only supports F32/F16/I32".to_string(),
            )),
        }
    }
}

impl KernelRenderable for TopKKernels {
    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        _: bool,
    ) -> Result<(), OperationError> {
        builder.register_storage("X", BindingMode::ReadOnly, Array::<P>::default());
        builder.register_storage("Y", BindingMode::ReadWrite, Array::<Scalar<i32>>::default());
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
            crate::rvec![
                BuiltIn::WorkgroupId,
                BuiltIn::LocalInvocationIndex,
                BuiltIn::NumWorkgroups
            ],
            device.compute_features().clone(),
        );

        let TopKKernels::Standard(inner) = self;

        self.register_bindings::<P>(&mut kernel_builder, inplace)?;
        kernel_builder.render_metadata(&self.metadata(dst, &self.kernel_element(dst))?);

        let dtype = P::render_type();
        // Helpers
        kernel_builder.write_global(wgsl! {
            const MAX_TOPK: u32 = 'MAX_TOPK;

            fn coords_from_slice_id(
                idx_in: u32,
                rank: u32,
                shape: vec4<u32>,
                exclude_dim: u32,
            ) -> vec4<u32> {
                var coords: vec4<u32> = vec4<u32>(0u);
                var idx: u32 = idx_in;
                for (var d: u32 = 0u; d < rank; d++) {
                    let dim_idx = rank - 1u - d;
                    if (dim_idx == exclude_dim) { continue; }
                    let base = shape[dim_idx];
                    coords[dim_idx] = idx % base;
                    idx = idx / base;
                }
                return coords;
            }

            fn linear_from_coords(coords: vec4<u32>, stride: vec4<u32>, rank: u32) -> u32 {
                var off: u32 = 0u;
                for (var d: u32 = 0u; d < rank; d++) {
                    off += coords[d] * stride[d];
                }
                return off;
            }
        });

        let loop_body = if inner.largest {
            wgsl! {
                while (pos < limit && top_vals[pos] > v) { pos += 1u; }
            }
        } else {
            wgsl! {
                while (pos < limit && top_vals[pos] < v) { pos += 1u; }
            }
        };

        // Main
        kernel_builder.write_main(wgsl! {
            // One invocation per slice
            let x_offset = workgroup_id.x * 64u;
            let slice_id = (workgroup_id.y * num_workgroups.x * 64u) + x_offset + local_invocation_index;
            if (slice_id >= metadata.num_slices) { return; }

            // Guard Max K
            if (metadata.k > MAX_TOPK) { return; }

            let rank = metadata.rank;
            let dim = metadata.dim;
            let n_dim = metadata.shape[dim];

            var coords = coords_from_slice_id(slice_id, rank, metadata.shape, dim);
            // Ensure target dim index 0 for base
            coords[dim] = 0u;
            let base = linear_from_coords(coords, metadata.stride, rank);
            let step = metadata.stride[dim];

            // Local buffers
            var top_vals: array<'dtype, MAX_TOPK>;
            var top_idxs: array<i32, MAX_TOPK>;
            var count: u32 = 0u;

            // Iterate across the reduction dimension
            for (var j: u32 = 0u; j < n_dim; j++) {
                let idx = base + j * step;
                let v = X[idx];

                // Determine insertion position
                var pos: u32 = 0u;
                let limit = count;
                'loop_body

                if (count < metadata.k) {
                    // Make room
                    var i: i32 = i32(count);
                    while (i > i32(pos)) {
                        top_vals[u32(i)] = top_vals[u32(i - 1)];
                        top_idxs[u32(i)] = top_idxs[u32(i - 1)];
                        i -= 1;
                    }
                    top_vals[pos] = v;
                    top_idxs[pos] = i32(j);
                    count += 1u;
                } else if (pos < metadata.k) {
                    // Insert and drop the last element
                    var i: i32 = i32(metadata.k - 1u);
                    while (i > i32(pos)) {
                        top_vals[u32(i)] = top_vals[u32(i - 1)];
                        top_idxs[u32(i)] = top_idxs[u32(i - 1)];
                        i -= 1;
                    }
                    top_vals[pos] = v;
                    top_idxs[pos] = i32(j);
                }
            }

            // Write out indices for this slice across k outputs
            // coords currently has target dim = 0; update per r
            for (var r: u32 = 0u; r < metadata.k; r++) {
                coords[dim] = r;
                let out_off = linear_from_coords(coords, metadata.out_stride, rank);
                Y[out_off] = top_idxs[r];
            }
        });

        Ok(kernel_builder.build()?)
    }
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use crate::randint;
    use crate::{DType, Device, DeviceRequest, Tensor, randn, test_util::run_py_prg};
    use test_strategy::{Arbitrary, proptest};

    #[derive(Arbitrary, Debug)]
    struct TopKProblem {
        #[strategy(1..=3usize)]
        B: usize,
        #[strategy(1..=64usize)]
        M: usize,
        #[strategy(1..=64usize)]
        N: usize,
        #[strategy(0..=2usize)]
        dim: usize,
        #[strategy(1..=4usize)]
        k: usize,
        largest: bool,
    }

    fn ground_truth_values(
        a: &Tensor,
        k: usize,
        dim: usize,
        largest: bool,
    ) -> anyhow::Result<Tensor> {
        let prg = r#"
import torch
def topk_vals(a, k, dim, largest):
    return torch.topk(torch.from_numpy(a), k, dim=dim, largest=largest)[0].float().numpy()
"#
        .to_string();
        run_py_prg(
            prg,
            &[a],
            &[&(k as i32), &(dim as i32), &largest],
            DType::F32,
        )
    }

    fn ground_truth_indices(
        a: &Tensor,
        k: usize,
        dim: usize,
        largest: bool,
    ) -> anyhow::Result<Tensor> {
        let prg = r#"
import torch
import numpy as np
def topk_idx(a, k, dim, largest):
    return torch.topk(torch.from_numpy(a), k, dim=dim, largest=largest)[1].to(torch.int32).numpy().astype(np.int32)
"#.to_string();
        run_py_prg(
            prg,
            &[a],
            &[&(k as i32), &(dim as i32), &largest],
            DType::I32,
        )
    }

    #[proptest(cases = 8)]
    fn test_topk(prob: TopKProblem) {
        let TopKProblem {
            B,
            M,
            N,
            dim,
            k,
            largest,
        } = prob;
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        let a = randn((B, M, N), None, None, Default::default()).unwrap();
        let (_dim_len, k_use) = {
            let dims = a.shape().to_vec();
            let dlen = dims[dim];
            (dlen, k.min(dims[dim].max(1)))
        };
        let a_vals = ground_truth_values(&a, k_use, dim, largest).unwrap();
        let a_idx = ground_truth_indices(&a, k_use, dim, largest).unwrap();

        let a_gpu = a.to(&device).unwrap();
        let ours = crate::topk(a_gpu, k_use, Some(dim), Some(largest), Some(false)).unwrap();
        let values = ours[0].clone().to(&Device::CPU).unwrap();
        let indices = ours[1].clone().to(&Device::CPU).unwrap();
        let a_idx_f32 = a_idx
            .clone()
            .cast(DType::F32)
            .unwrap()
            .to(&Device::CPU)
            .unwrap();
        let indices_f32 = indices
            .clone()
            .cast(DType::F32)
            .unwrap()
            .to(&Device::CPU)
            .unwrap();

        a_vals.all_close(&values, 3e-5f32, 1e-5f32).unwrap();
        assert_eq!(a_idx.dtype(), DType::I32);
        assert_eq!(indices.dtype(), DType::I32);
        a_idx_f32.all_close(&indices_f32, 0.0f32, 0.0f32).unwrap();
    }

    #[derive(Arbitrary, Debug)]
    struct TopKIntProblem {
        #[strategy(1..=3usize)]
        B: usize,
        #[strategy(1..=32usize)]
        M: usize,
        #[strategy(1..=32usize)]
        N: usize,
        #[strategy(0..=2usize)]
        dim: usize,
        #[strategy(1..=4usize)]
        k: usize,
        largest: bool,
    }

    fn ground_truth_values_i32(
        a: &Tensor,
        k: usize,
        dim: usize,
        largest: bool,
    ) -> anyhow::Result<Tensor> {
        let prg = r#"
import torch
def topk_vals_i32(a, k, dim, largest):
    return torch.topk(torch.from_numpy(a).to(torch.int32), k, dim=dim, largest=largest)[0].to(torch.int32).numpy()
"#.to_string();
        run_py_prg(
            prg,
            &[a],
            &[&(k as i32), &(dim as i32), &largest],
            DType::I32,
        )
    }

    fn ground_truth_indices_i32(
        a: &Tensor,
        k: usize,
        dim: usize,
        largest: bool,
    ) -> anyhow::Result<Tensor> {
        let prg = r#"
import torch
import numpy as np
def topk_idx_i32(a, k, dim, largest):
    return torch.topk(torch.from_numpy(a).to(torch.int32), k, dim=dim, largest=largest)[1].to(torch.int32).numpy().astype(np.int32)
"#.to_string();
        run_py_prg(
            prg,
            &[a],
            &[&(k as i32), &(dim as i32), &largest],
            DType::I32,
        )
    }

    #[proptest(cases = 6)]
    fn test_topk_i32(prob: TopKIntProblem) {
        let TopKIntProblem {
            B,
            M,
            N,
            dim,
            k,
            largest,
        } = prob;
        let device = Device::request_device(DeviceRequest::GPU).unwrap();

        // Generate integer tensor in a safe range [0, 1000)
        let a = randint(0, 1000, (B, M, N), Default::default()).unwrap();
        let (_dim_len, k_use) = {
            let dims = a.shape().to_vec();
            let dlen = dims[dim];
            (dlen, k.min(dims[dim].max(1)))
        };

        let a_vals = ground_truth_values_i32(&a, k_use, dim, largest).unwrap();
        let a_idx = ground_truth_indices_i32(&a, k_use, dim, largest).unwrap();

        let a_gpu = a.to(&device).unwrap();
        let ours = crate::topk(a_gpu, k_use, Some(dim), Some(largest), Some(false)).unwrap();
        let values = ours[0].clone().to(&Device::CPU).unwrap();
        let indices = ours[1].clone().to(&Device::CPU).unwrap();

        assert_eq!(values.dtype(), DType::I32);
        assert_eq!(indices.dtype(), DType::I32);

        // Compare values and indices against PyTorch ground truth
        // Convert to F32 for all_close API that expects floats
        let a_vals_f32 = a_vals.clone().cast(DType::F32).unwrap();
        let values_f32 = values.clone().cast(DType::F32).unwrap();
        a_vals_f32.all_close(&values_f32, 0.0f32, 0.0f32).unwrap();

        let a_idx_f32 = a_idx.clone().cast(DType::F32).unwrap();
        let indices_f32 = indices.clone().cast(DType::F32).unwrap();
        a_idx_f32.all_close(&indices_f32, 0.0f32, 0.0f32).unwrap();
    }
}
