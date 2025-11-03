use derive_new::new;
use encase::ShaderType;
use half::f16;
use inline_wgsl::wgsl;
use ratchet_macros::{IrFields, WgslMetadata};

use crate::{
    gpu::{dtype::WgslDType, BindGroupLayoutDescriptor},
    rvec, Array, BindingMode, BuiltIn, DType, GPUOperation, Kernel, KernelElement,
    KernelRenderable, KernelSource, OpGuards, Operation, OperationError, RVec, Scalar, Shape,
    StorageView, Strides, Tensor, Vec2, Vec4, WgslKernelBuilder, WgslPrimitive, WorkgroupSize,
    Workload,
};

#[derive(new, Debug, Clone, IrFields)]
pub struct FillConstant {
    pub shape: Shape,
    pub value: f32,
}

#[derive(Debug, derive_new::new, ShaderType, WgslMetadata)]
pub struct FillConstantMeta {
    numel: u32,
    value: f32,
}

impl Operation for FillConstant {
    fn name(&self) -> &'static str {
        "FillConstant"
    }
    fn compute_view(&self) -> Result<StorageView, OperationError> {
        let shape: Shape = self.shape.clone();
        let strides = Strides::from(&shape);
        Ok(StorageView::new(shape, DType::F32, strides))
    }

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![]
    }

    fn supports_inplace(&self) -> bool {
        false
    }
}

impl OpGuards for FillConstant {
    fn check_shapes(&self) {
        // No input shapes to check
    }

    fn check_dtypes(&self) {
        // No input dtypes to check
    }
}

pub enum FillConstantKernels {
    Standard(FillConstant),
}

impl GPUOperation for FillConstant {
    type KernelEnum = FillConstantKernels;

    fn select_kernel(&self) -> Self::KernelEnum {
        FillConstantKernels::Standard(self.clone())
    }
}

impl KernelRenderable for FillConstantKernels {
    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        _: bool,
    ) -> Result<(), OperationError> {
        builder.register_storage("Y", BindingMode::ReadWrite, Array::<P>::default());
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

        let N = (P::W as u32).render();
        let dt = P::render_type();

        kernel_builder.write_main(wgsl! {
            let x_offset = workgroup_id.x * 64u;
            let index = (workgroup_id.y * num_workgroups.x * 64u) + x_offset + local_invocation_index;
            if (index >= metadata.numel / 'N) {
                return;
            }
            Y[index] = 'dt(metadata.value);
        });

        Ok(kernel_builder.build()?)
    }
}

impl Kernel for FillConstantKernels {
    type Metadata = FillConstantMeta;

    fn kernel_name(&self) -> String {
        match self {
            FillConstantKernels::Standard(_) => "fill_constant".to_string(),
        }
    }

    fn kernel_element(&self, dst: &Tensor) -> KernelElement {
        let rank = dst.shape().rank();
        let N = if rank > 0 { dst.shape()[rank - 1] } else { 1 };

        if N % 4 == 0 {
            KernelElement::Vec4
        } else if N % 2 == 0 {
            KernelElement::Vec2
        } else {
            KernelElement::Scalar
        }
    }

    fn calculate_dispatch(&self, dst: &Tensor) -> Result<Workload, OperationError> {
        Ok(Workload::std(dst.shape().numel(), self.kernel_element(dst)))
    }

    fn storage_bind_group_layout(
        &self,
        _inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        Ok(BindGroupLayoutDescriptor::unary_inplace())
    }

    fn metadata(&self, _: &Tensor, _: &KernelElement) -> Result<Self::Metadata, OperationError> {
        let FillConstantKernels::Standard(inner) = self;
        Ok(FillConstantMeta {
            numel: inner.shape.clone().numel() as u32,
            value: inner.value,
        })
    }

    fn build_kernel(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let kernel_element = self.kernel_element(dst);
        match (dst.dt(), &kernel_element) {
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
            (DType::I32, KernelElement::Scalar) => {
                self.render::<Scalar<i32>>(inplace, dst, workgroup_size)
            }
            (DType::I32, KernelElement::Vec2) => {
                self.render::<Vec2<i32>>(inplace, dst, workgroup_size)
            }
            (DType::I32, KernelElement::Vec4) => {
                self.render::<Vec4<i32>>(inplace, dst, workgroup_size)
            }
            _ => Err(OperationError::CompileError(format!(
                "Unsupported dtype {:?} or kernel element {:?}",
                dst.dt(),
                kernel_element
            ))),
        }
    }
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use test_strategy::{proptest, Arbitrary};

    use crate::{shape, test_util::run_py_prg, DType, Device, DeviceRequest, Tensor};

    fn ground_truth(shape: &[usize], value: f32) -> anyhow::Result<Tensor> {
        let prg = r#"
import torch
def fill_constant(shape, value):
    return torch.full(shape, value, dtype=torch.float32).cpu().numpy()
"#;

        run_py_prg(prg.to_string(), &[], &[&shape, &value], DType::F32)
    }

    fn run_fill_constant_trial(problem: FillConstantProblem, device: Device) {
        let FillConstantProblem { B, M, N, value } = problem;

        let a = Tensor::full(&shape![B, M, N], value, &device);
        let ground = ground_truth(&[B, M, N], value).unwrap();

        let a_gpu = a.to(&device).unwrap();

        let ours = a_gpu.to(&Device::CPU).unwrap();

        println!("ours = {:?}", ours);
        println!("ground = {:?}", ground);

        // Compare our result with ground truth
        ground.all_close(&ours, 1e-6, 1e-6).unwrap();
    }

    #[derive(Arbitrary, Debug)]
    struct FillConstantProblem {
        #[strategy(1..=128usize)]
        B: usize,
        #[strategy(1..=128usize)]
        M: usize,
        #[strategy(1..=128usize)]
        N: usize,
        value: f32,
    }

    #[proptest(cases = 8)]
    fn test_fill_constant(prob: FillConstantProblem) {
        let FillConstantProblem { B, M, N, value } = prob;
        println!("B = {}, M = {}, N = {}, value = {}", B, M, N, value);
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        run_fill_constant_trial(prob, device);
    }
}
