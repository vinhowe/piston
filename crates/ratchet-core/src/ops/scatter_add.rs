use derive_new::new;
use encase::ShaderType;
use half::f16;
use inline_wgsl::wgsl;
use ratchet_macros::{IrFields, WgslMetadata};

use crate::{
    gpu::BindGroupLayoutDescriptor, rvec, Array, BindingMode, BuiltIn, DType, GPUOperation, Kernel,
    KernelElement, KernelRenderable, KernelSource, OpGuards, Operation, OperationError, RVec,
    Scalar, StorageView, Tensor, Vec2, Vec4, WgslKernelBuilder, WgslPrimitive, WorkgroupSize,
    Workload,
};

#[derive(new, Debug, Clone, IrFields)]
pub struct ScatterAdd {
    pub dst: Tensor,
    pub src: Tensor,
    pub ids: Tensor,
    pub dim: usize,
}

#[derive(Debug, derive_new::new, ShaderType, WgslMetadata)]
pub struct ScatterAddMeta {
    left_size: u32,
    right_size: u32,
    dst_dim_size: u32,
    src_dim_size: u32,
}

impl OpGuards for ScatterAdd {
    fn check_shapes(&self) {
        assert!(self.src.rank() >= 1);
        assert_eq!(self.src.shape().len(), self.ids.shape().len());
        assert_eq!(self.src.shape().len(), self.dst.shape().len());
    }

    fn check_dtypes(&self) {
        assert!(self.ids.dt() == crate::DType::I32);
        assert!(self.dst.dt() == crate::DType::F32);
        assert!(self.src.dt() == crate::DType::F32);
    }
}

impl Operation for ScatterAdd {
    fn name(&self) -> &'static str {
        "ScatterAdd"
    }

    fn compute_view(&self) -> Result<StorageView, OperationError> {
        Ok(self.dst.storage_view().clone())
    }

    #[inline]
    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.dst, &self.src, &self.ids]
    }

    fn supports_inplace(&self) -> bool {
        true
    }
}

pub enum ScatterAddKernels {
    Standard(ScatterAdd),
}

impl GPUOperation for ScatterAdd {
    type KernelEnum = ScatterAddKernels;

    fn select_kernel(&self) -> Self::KernelEnum {
        ScatterAddKernels::Standard(self.clone())
    }
}

impl KernelRenderable for ScatterAddKernels {
    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        _: bool,
    ) -> Result<(), OperationError> {
        builder.register_storage("X", BindingMode::ReadWrite, Array::<P>::default());
        builder.register_storage("S", BindingMode::ReadOnly, Array::<P>::default());
        builder.register_storage("I", BindingMode::ReadOnly, Array::<Scalar<i32>>::default());
        builder.register_uniform();
        Ok(())
    }

    fn render<P: WgslPrimitive>(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let device = dst.device().try_gpu()?;
        let mut kernel_builder = WgslKernelBuilder::new(
            workgroup_size.clone(),
            rvec![
                BuiltIn::WorkgroupId,
                BuiltIn::LocalInvocationIndex,
                BuiltIn::NumWorkgroups
            ],
            device.compute_features().clone(),
        );

        self.register_bindings::<P>(&mut kernel_builder, inplace)?;
        kernel_builder.render_metadata(&self.metadata(dst, &self.kernel_element(dst))?);

        kernel_builder.write_main(wgsl! {
            let numel = metadata.left_size * metadata.right_size;

            for (var i = workgroup_id.x * 64u + local_invocation_index; i < numel; i += 64u * num_workgroups.x) {
                let pre = i / metadata.right_size;
                let post = i % metadata.right_size;
                for (var j = 0u; j < metadata.src_dim_size; j++) {
                    let src_i = (pre * metadata.src_dim_size + j) * metadata.right_size + post;
                    let idx = u32(I[src_i]);
                    let dst_i = (pre * metadata.dst_dim_size + idx) * metadata.right_size + post;
                    X[dst_i] += S[src_i];
                }
            }
        });

        Ok(kernel_builder.build()?)
    }
}

impl Kernel for ScatterAddKernels {
    type Metadata = ScatterAddMeta;

    fn kernel_name(&self) -> String {
        match self {
            ScatterAddKernels::Standard(_inner) => "scatter_add".to_string(),
        }
    }

    fn kernel_element(&self, _dst: &Tensor) -> KernelElement {
        KernelElement::Scalar
    }

    fn calculate_dispatch(&self, dst: &Tensor) -> Result<Workload, OperationError> {
        let ScatterAddKernels::Standard(inner) = self;
        let dim = inner.dim;
        let src_shape_vec = inner.src.shape().to_vec();
        let left_size = src_shape_vec[..dim].iter().product::<usize>() as u32;
        let right_size = src_shape_vec[dim + 1..].iter().product::<usize>() as u32;
        let numel = left_size * right_size;
        Ok(Workload::std(numel as _, self.kernel_element(dst)))
    }

    fn storage_bind_group_layout(
        &self,
        inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        if !inplace {
            panic!("ScatterAdd only supports inplace operation");
        }
        Ok(BindGroupLayoutDescriptor::ternary_inplace())
    }

    fn metadata(&self, _: &Tensor, _: &KernelElement) -> Result<Self::Metadata, OperationError> {
        let ScatterAddKernels::Standard(inner) = self;
        let dim = inner.dim;
        let src_shape_vec = inner.src.shape().to_vec();
        let dst_shape_vec = inner.dst.shape().to_vec();
        let left_size = src_shape_vec[..dim].iter().product::<usize>() as u32;
        let right_size = src_shape_vec[dim + 1..].iter().product::<usize>() as u32;
        let dst_dim_size = dst_shape_vec[dim] as u32;
        let src_dim_size = src_shape_vec[dim] as u32;

        Ok(ScatterAddMeta {
            left_size,
            right_size,
            dst_dim_size,
            src_dim_size,
        })
    }

    fn build_kernel(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let ScatterAddKernels::Standard(inner) = self;
        let kernel_element = self.kernel_element(dst);
        match (inner.src.dt(), &kernel_element) {
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
                inner.src.dt(),
                kernel_element
            ))),
        }
    }
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use std::cmp::max;

    use test_strategy::{proptest, Arbitrary};

    use crate::test_util::run_py_prg;
    use crate::{shape, Device, DeviceRequest, Tensor};

    fn ground_truth(
        dst: &Tensor,
        src: &Tensor,
        ids: &Tensor,
        dim: usize,
    ) -> anyhow::Result<Tensor> {
        let prg = format!(
            r#"
import torch
def scatter_add(dst, src, ids):
    dst_tensor = torch.from_numpy(dst)
    src_tensor = torch.from_numpy(src)
    ids_tensor = torch.from_numpy(ids).to(torch.long)
    result = dst_tensor.scatter_add({}, ids_tensor, src_tensor)
    return result.numpy()
"#,
            dim
        );
        run_py_prg(prg.to_string(), &[dst, src, ids], &[], src.dt())
    }

    fn run_scatter_add_trial(problem: ScatterAddProblem, device: Device) {
        let ScatterAddProblem { B, M, N, dim } = problem;

        let dst_shape = shape![B, M, N];
        let mut src_shape = vec![B, M, N];
        src_shape[dim] = max(M.min(N) / 2, 1); // Make src dimension smaller than dst

        let dst = Tensor::zeros::<f32>(&dst_shape, &Device::CPU);
        let src = Tensor::ones::<f32>(&src_shape.into(), &Device::CPU);

        // Create ids tensor with the same shape as src, but with values in range [0, dst_shape[dim])
        let ids = Tensor::randint(0, dst_shape[dim] as i32, src.shape().clone(), Device::CPU);

        let ground = ground_truth(&dst, &src, &ids, dim).unwrap();

        let dst_gpu = dst.to(&device).unwrap();
        let src_gpu = src.to(&device).unwrap();
        let ids_gpu = ids.to(&device).unwrap();

        let result = dst_gpu
            .scatter_add(ids_gpu.clone(), src_gpu.clone(), dim)
            .unwrap();

        let ours = result.to(&Device::CPU).unwrap();

        ground.all_close(&ours, 1e-5, 1e-5).unwrap();
    }

    #[derive(Arbitrary, Debug)]
    struct ScatterAddProblem {
        #[strategy(1..=256usize)]
        B: usize,
        #[strategy(1..=3usize)]
        M: usize,
        #[strategy(1..=3usize)]
        N: usize,
        #[strategy(0..=2usize)]
        dim: usize,
    }

    #[proptest(cases = 16)]
    fn test_scatter_add(prob: ScatterAddProblem) {
        log::info!(
            "B = {}, M = {}, N = {}, dim = {}",
            prob.B,
            prob.M,
            prob.N,
            prob.dim
        );
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        run_scatter_add_trial(prob, device);
    }
}
