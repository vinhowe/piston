use crate::{
    gpu::{dtype::WgslDType, BindGroupLayoutDescriptor},
    rvec, Array, BindingMode, BuiltIn, DType, GPUOperation, Kernel, KernelElement,
    KernelRenderable, KernelSource, OpGuards, Operation, OperationError, RVec, Scalar, StorageView,
    Tensor, Vec2, Vec4, WgslKernelBuilder, WgslPrimitive, WorkgroupSize, Workload,
};
use derive_new::new;
use encase::ShaderType;
use half::f16;
use inline_wgsl::wgsl;
use ratchet_macros::{IrFields, WgslMetadata};

#[derive(new, Debug, Clone, IrFields)]
pub struct Gather {
    pub src: Tensor,
    pub ids: Tensor,
    pub dim: usize,
}

#[derive(Debug, derive_new::new, ShaderType, WgslMetadata)]
pub struct GatherMeta {
    ids_numel: u32,
    left_size: u32,
    right_size: u32,
    src_dim_size: u32,
    ids_dim_size: u32,
}

impl OpGuards for Gather {
    fn check_shapes(&self) {
        assert!(self.src.rank() >= 1);
    }

    fn check_dtypes(&self) {
        assert!(self.ids.dtype() == crate::DType::I32);
    }
}

impl Operation for Gather {
    fn name(&self) -> &'static str {
        "Gather"
    }

    fn compute_view(&self) -> Result<StorageView, OperationError> {
        Ok(StorageView::new(
            self.ids.shape().clone(),
            self.src.dtype(),
            self.ids.stride().clone(),
        ))
    }

    #[inline]
    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.src, &self.ids]
    }

    fn supports_inplace(&self) -> bool {
        false
    }
}

pub enum GatherKernels {
    Standard(Gather),
}

impl GPUOperation for Gather {
    type KernelEnum = GatherKernels;

    fn select_kernel(&self) -> Self::KernelEnum {
        GatherKernels::Standard(self.clone())
    }
}

impl KernelRenderable for GatherKernels {
    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        _: bool,
    ) -> Result<(), OperationError> {
        let arr = Array::<P>::default();
        builder.register_storage("X", BindingMode::ReadOnly, arr);
        builder.register_storage("I", BindingMode::ReadOnly, Array::<Scalar<i32>>::default());
        builder.register_storage("Y", BindingMode::ReadWrite, arr);
        builder.register_uniform();
        Ok(())
    }

    fn render<P: WgslPrimitive>(
        &self,
        _: bool,
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

        self.register_bindings::<P>(&mut kernel_builder, false)?;
        kernel_builder.render_metadata(&self.metadata(dst, &self.kernel_element(dst))?);

        // This is always 1 because we currently only support scalar gather
        let N = (P::W as u32).render();

        kernel_builder.write_main(wgsl! {
            let x_offset = workgroup_id.x * 64u;
            let index = (workgroup_id.y * num_workgroups.x * 64u) + x_offset + local_invocation_index;
            if (index >= metadata.ids_numel / 'N) {
                return;
            }

            for (var i: u32 = index; i < metadata.ids_numel; i += 64u * num_workgroups.x) {
                let post = i % metadata.right_size;
                let idx = u32(I[i]);
                let pre = u32(i / (metadata.right_size * metadata.ids_dim_size));
                let src_i = u32((pre * metadata.src_dim_size + idx) * metadata.right_size + post);
                Y[i] = X[src_i];
            }
        });

        Ok(kernel_builder.build()?)
    }
}

impl Kernel for GatherKernels {
    type Metadata = GatherMeta;

    fn storage_bind_group_layout(
        &self,
        inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        if inplace {
            panic!("Only non-inplace gather is supported");
        }
        Ok(BindGroupLayoutDescriptor::binary())
    }

    fn calculate_dispatch(&self, _dst: &Tensor) -> Result<Workload, OperationError> {
        let GatherKernels::Standard(inner) = self;
        Ok(Workload::std(
            inner.ids.shape().numel(),
            KernelElement::Scalar,
        ))
    }

    fn kernel_name(&self) -> String {
        match self {
            GatherKernels::Standard(_) => "gather".to_string(),
        }
    }

    fn kernel_element(&self, _dst: &Tensor) -> KernelElement {
        KernelElement::Scalar
    }

    fn metadata(&self, _: &Tensor, _: &KernelElement) -> Result<Self::Metadata, OperationError> {
        let GatherKernels::Standard(inner) = self;
        let src_dims = inner.src.shape().to_vec();

        let left_sz: usize = src_dims[..inner.dim].iter().product();
        let right_sz: usize = src_dims[inner.dim + 1..].iter().product();
        let src_dim_sz = src_dims[inner.dim];
        let ids_dim_sz = inner.ids.shape()[inner.dim];

        Ok(GatherMeta {
            ids_numel: inner.ids.shape().numel() as u32,
            left_size: left_sz as u32,
            right_size: right_sz as u32,
            src_dim_size: src_dim_sz as u32,
            ids_dim_size: ids_dim_sz as u32,
        })
    }

    fn build_kernel(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let kernel_element = self.kernel_element(dst);
        let GatherKernels::Standard(inner) = self;
        match (inner.src.dtype(), &kernel_element) {
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
                inner.src.dtype(),
                kernel_element
            ))),
        }
    }
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use test_strategy::{proptest, Arbitrary};

    use crate::test_util::run_py_prg;
    use crate::{shape, DType, Device, DeviceRequest, Tensor, Var};

    fn ground_truth(src: &Tensor, ids: &Tensor, dim: usize) -> anyhow::Result<Tensor> {
        let prg = format!(
            r#"
import torch
def gather(src, ids, dim):
    return torch.gather(torch.from_numpy(src), {}, torch.from_numpy(ids).long()).numpy()
"#,
            dim
        );
        run_py_prg(prg.to_string(), &[src, ids], &[&dim], src.dtype())
    }

    fn run_gather_trial(problem: GatherProblem, device: Device) {
        let GatherProblem { B, M, N, dim } = problem;

        let src = Tensor::randn::<f32>(0., 1., shape![B, M, N], Device::CPU);

        // Create the shape for ids tensor
        let mut ids_shape = vec![B, M, N];
        ids_shape[dim] = 1;
        let ids = Tensor::randint::<i32>(0, src.shape()[dim] as i32, ids_shape.into(), Device::CPU);

        let ground = ground_truth(&src, &ids, dim).unwrap();

        let src_gpu = src.to(&device).unwrap();
        let ids_gpu = ids.to(&device).unwrap();

        let result = src_gpu.gather(ids_gpu, dim).unwrap();

        let ours = result.to(&Device::CPU).unwrap();
        log::debug!("src = {:?}", src);
        log::debug!("ids = {:?}", ids);
        log::debug!("ours = {:?}", ours);
        log::debug!("ground = {:?}", ground);
        ground.all_close(&ours, 1e-5, 1e-5).unwrap();
    }

    #[derive(Arbitrary, Debug)]
    struct GatherProblem {
        #[strategy(1..=3usize)]
        B: usize,
        #[strategy(1..=64usize)]
        M: usize,
        #[strategy(1..=64usize)]
        N: usize,
        #[strategy(0..=2usize)]
        dim: usize,
    }

    #[proptest(cases = 8)]
    fn test_gather(prob: GatherProblem) {
        let _ = env_logger::builder().is_test(true).try_init();
        let GatherProblem { B, M, N, dim } = prob;
        log::info!("B = {}, M = {}, N = {}, dim = {}", B, M, N, dim);
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        run_gather_trial(prob, device);
    }

    #[derive(Arbitrary, Debug)]
    struct GatherBackwardProblem {
        #[strategy(1..=3usize)]
        B: usize,
        #[strategy(1..=64usize)]
        M: usize,
        #[strategy(1..=64usize)]
        N: usize,
        #[strategy(0..=2usize)]
        dim: usize,
    }

    fn ground_truth_backward(src: &Tensor, ids: &Tensor, dim: usize) -> anyhow::Result<Tensor> {
        let prg = format!(
            r#"
import torch
def gather_backward(src, ids):
    src_tensor = torch.tensor(torch.from_numpy(src), requires_grad=True)
    ids_tensor = torch.from_numpy(ids).long()
    result = torch.gather(src_tensor, {}, ids_tensor)
    result.backward(torch.ones_like(result))
    return src_tensor.grad.numpy()
"#,
            dim
        );
        run_py_prg(prg.to_string(), &[src, ids], &[], DType::F32)
    }

    fn run_gather_backward_trial(problem: GatherBackwardProblem) -> anyhow::Result<()> {
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        let GatherBackwardProblem { B, M, N, dim } = problem;
        let src = Tensor::randn::<f32>(0., 1., shape![B, M, N], Device::CPU);

        // Create the shape for ids tensor
        let mut ids_shape = vec![B, M, N];
        ids_shape[dim] = 1;
        let ids = Tensor::randint::<i32>(0, src.shape()[dim] as i32, ids_shape.into(), Device::CPU);

        let ground = ground_truth_backward(&src, &ids, dim)?;

        let src_gpu = src.to(&device)?;
        let ids_gpu = ids.to(&device)?;
        let src_var = Var::from_tensor(&src_gpu)?;
        let result_gpu = src_var.as_tensor().clone().gather(ids_gpu, dim)?;

        let grads = result_gpu.backward()?;
        device.try_gpu()?.mark_step()?;

        let src_grad = grads.get(src_var.as_tensor()).unwrap().clone();

        let ours = src_grad.to(&Device::CPU)?;
        let src_cpu = src.to(&Device::CPU)?;
        let ids_cpu = ids.to(&Device::CPU)?;

        println!("src = {:?}", src_cpu);
        println!("ids = {:?}", ids_cpu);
        println!("ours = {:?}", ours);
        println!("ground = {:?}", ground);
        ground.all_close(&ours, 1e-5, 1e-5)?;
        Ok(())
    }

    #[proptest(cases = 8)]
    fn test_gather_backward(prob: GatherBackwardProblem) {
        let _ = env_logger::builder().is_test(true).try_init();
        let GatherBackwardProblem { B, M, N, dim } = prob;
        println!("B = {}, M = {}, N = {}, dim = {}", B, M, N, dim);
        run_gather_backward_trial(prob).unwrap();
    }
}
