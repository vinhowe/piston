use crate::{
    Array, BindingMode, BuiltIn, DType, GPUOperation, Kernel, KernelElement, KernelRenderable,
    KernelSource, OpGuards, OpTensor, Operation, OperationError, RVec, Scalar, Shape, StorageView,
    Stride, WgslKernelBuilder, WgslPrimitive, WorkgroupSize, Workload,
    gpu::BindGroupLayoutDescriptor, rvec,
};
use derive_new::new;
use encase::ShaderType;
use inline_wgsl::wgsl;
use piston_macros::{IrFields, WgslMetadata};

#[derive(new, Debug, Clone, IrFields)]
pub struct OneHot {
    pub indices: OpTensor,
    pub num_classes: usize,
}

#[derive(Debug, derive_new::new, ShaderType, WgslMetadata)]
pub struct OneHotMeta {
    input_numel: u32,
    num_classes: u32,
}

impl OpGuards for OneHot {
    fn check_shapes(&self) {}

    fn check_dtypes(&self) {
        assert!(
            matches!(self.indices.dtype(), DType::I32),
            "one_hot: indices must be I32"
        );
    }

    fn check_custom(&self) {
        assert!(self.num_classes > 0, "one_hot: num_classes must be > 0");
    }
}

impl Operation for OneHot {
    fn name(&self) -> &'static str {
        "OneHot"
    }

    fn compute_view(&self) -> Result<StorageView, OperationError> {
        let mut out_shape = self.indices.shape().to_vec();
        out_shape.push(self.num_classes);
        let shape: Shape = out_shape.into();
        let stride = Stride::from(&shape);
        Ok(StorageView::new(shape, DType::I32, stride))
    }

    fn srcs(&self) -> RVec<&OpTensor> {
        rvec![&self.indices]
    }

    fn supports_inplace(&self) -> bool {
        false
    }
}

pub enum OneHotKernels {
    Standard(OneHot),
}

impl GPUOperation for OneHot {
    type KernelEnum = OneHotKernels;

    fn select_kernel(&self) -> Self::KernelEnum {
        OneHotKernels::Standard(self.clone())
    }
}

impl KernelRenderable for OneHotKernels {
    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        _: bool,
    ) -> Result<(), OperationError> {
        builder.register_storage("X", BindingMode::ReadOnly, Array::<Scalar<i32>>::default());
        builder.register_storage("Y", BindingMode::ReadWrite, Array::<Scalar<i32>>::default());
        builder.register_uniform();
        Ok(())
    }

    fn render<P: WgslPrimitive>(
        &self,
        _: bool,
        dst: &OpTensor,
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

        kernel_builder.write_main(wgsl! {
            let x_offset = workgroup_id.x * 64u;
            let index = (workgroup_id.y * num_workgroups.x * 64u) + x_offset + local_invocation_index;
            let total = metadata.input_numel * metadata.num_classes;
            if (index >= total) {
                return;
            }

            for (var i: u32 = index; i < total; i += 64u * num_workgroups.x) {
                let sample_idx = i / metadata.num_classes;
                let class_idx = i % metadata.num_classes;
                let idx_val = i32(X[sample_idx]);
                if (idx_val == i32(class_idx)) {
                    Y[i] = 1;
                } else {
                    Y[i] = 0;
                }
            }
        });

        Ok(kernel_builder.build()?)
    }
}

impl Kernel for OneHotKernels {
    type Metadata = OneHotMeta;

    fn storage_bind_group_layout(
        &self,
        inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        if inplace {
            panic!("Only non-inplace one_hot is supported");
        }
        Ok(BindGroupLayoutDescriptor::unary())
    }

    fn calculate_dispatch(&self, dst: &OpTensor) -> Result<Workload, OperationError> {
        Ok(Workload::std(dst.shape().numel(), KernelElement::Scalar))
    }

    fn kernel_name(&self) -> String {
        match self {
            OneHotKernels::Standard(_) => "one_hot".to_string(),
        }
    }

    fn kernel_element(&self, _dst: &OpTensor) -> KernelElement {
        KernelElement::Scalar
    }

    fn metadata(
        &self,
        _dst: &OpTensor,
        _ke: &KernelElement,
    ) -> Result<Self::Metadata, OperationError> {
        let OneHotKernels::Standard(inner) = self;
        Ok(OneHotMeta {
            input_numel: inner.indices.shape().numel() as u32,
            num_classes: inner.num_classes as u32,
        })
    }

    fn build_kernel(
        &self,
        inplace: bool,
        dst: &OpTensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        match dst.dtype() {
            DType::I32 => self.render::<Scalar<i32>>(inplace, dst, workgroup_size),
            _ => Err(OperationError::CompileError(format!(
                "Unsupported dtype {:?} for one_hot",
                dst.dtype()
            ))),
        }
    }
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use crate::test_util::run_py_prg;
    use crate::{DType, Device, DeviceRequest, Tensor, TensorOptions, randint};
    use test_strategy::{Arbitrary, proptest};

    fn ground_truth(indices: &Tensor, num_classes: usize) -> anyhow::Result<Tensor> {
        let prg = r#"
import torch
def one_hot(indices, num_classes):
    t = torch.from_numpy(indices).long()
    out = torch.nn.functional.one_hot(t, num_classes)
    return out.to(torch.int32).numpy()
"#;
        run_py_prg(prg.to_string(), &[indices], &[&num_classes], DType::I32)
    }

    #[derive(Arbitrary, Debug)]
    struct OneHotProblem {
        #[strategy(1..=3usize)]
        B: usize,
        #[strategy(1..=16usize)]
        M: usize,
        #[strategy(1..=8usize)]
        N: usize,
        #[strategy(2..=16usize)]
        num_classes: usize,
        #[strategy(1..=3usize)]
        rank: usize,
    }

    fn shape_for(prob: &OneHotProblem) -> crate::Shape {
        match prob.rank {
            1 => crate::Shape::from(vec![prob.M]),
            2 => crate::Shape::from(vec![prob.B, prob.M]),
            _ => crate::Shape::from(vec![prob.B, prob.M, prob.N]),
        }
    }

    #[proptest(cases = 8)]
    fn test_one_hot_matches_pytorch(prob: OneHotProblem) {
        let _ = env_logger::builder().is_test(true).try_init();
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        let shape = shape_for(&prob);

        let indices = randint(0, prob.num_classes as i32, shape, TensorOptions::new()).unwrap();

        let ground = ground_truth(&indices, prob.num_classes).unwrap();

        let ours = indices
            .clone()
            .to(&device)
            .unwrap()
            .one_hot(prob.num_classes)
            .unwrap()
            .to(&crate::Device::CPU)
            .unwrap();

        let ground_f32 = ground.cast(DType::F32).unwrap();
        let ours_f32 = ours.cast(DType::F32).unwrap();
        ground_f32.all_close(&ours_f32, 0.0f32, 0.0f32).unwrap();
    }
}
