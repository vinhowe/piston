use derive_new::new;
use encase::ShaderType;
use half::f16;
use inline_wgsl::wgsl;
use ratchet_macros::{IrFields, WgslMetadata};

use crate::{
    gpu::{dtype::WgslDType, BindGroupLayoutDescriptor},
    rvec, Array, BindingMode, BuiltIn, DType, GPUOperation, Kernel, KernelElement,
    KernelRenderable, KernelSource, OpGuards, Operation, OperationError, RVec, Scalar, StorageView,
    Tensor, Vec2, Vec4, WgslKernelBuilder, WgslPrimitive, WorkgroupSize, Workload,
};

#[derive(new, Debug, Clone, IrFields)]
pub struct Powf {
    pub src: Tensor,
    pub e: f32,
}

#[derive(Debug, ShaderType, WgslMetadata)]
pub struct PowfMeta {
    numel: u32,
    e: f32,
}

impl OpGuards for Powf {
    fn check_shapes(&self) {}

    fn check_dtypes(&self) {
        let a = &self.src;
        assert!(matches!(a.dtype(), crate::DType::F32));
    }
}

impl Operation for Powf {
    fn name(&self) -> &'static str {
        "Powf"
    }

    fn compute_view(&self) -> Result<StorageView, OperationError> {
        Ok(self.src.storage_view().clone())
    }

    #[inline]
    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.src]
    }

    fn supports_inplace(&self) -> bool {
        true
    }
}

pub enum PowfKernels {
    Standard(Powf),
}

impl GPUOperation for Powf {
    type KernelEnum = PowfKernels;

    fn select_kernel(&self) -> Self::KernelEnum {
        PowfKernels::Standard(self.clone())
    }
}

impl KernelRenderable for PowfKernels {
    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        inplace: bool,
    ) -> Result<(), OperationError> {
        let arr = Array::<P>::default();
        if inplace {
            builder.register_storage("X", BindingMode::ReadWrite, arr);
        } else {
            builder.register_storage("X", BindingMode::ReadOnly, arr);
            builder.register_storage("Y", BindingMode::ReadWrite, arr);
        }
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

        let N = (P::W as u32).render();
        let dtype = P::render_type();

        kernel_builder.write_main(wgsl! {
            let x_offset = workgroup_id.x * 64u;
            let index = (workgroup_id.y * num_workgroups.x * 64u) + x_offset + local_invocation_index;
            if (index >= metadata.numel / 'N) {
                return;
            }

            let val = X[index];
        });

        // pow(x, e) is undefined for x < 0 in Dawn, but apparently not in wgpu,
        // but only when the compiler doesn't have enough information to coerce
        // e into an integer. We supply e through the metadata, so, at compile-time,
        // its type is unknown.
        //
        // Multiplying by the sign is a fix to make this shader work correctly in Chrome.
        let apply = if inplace {
            wgsl! {
                X[index] = sign(val) * pow(abs(val), 'dtype(metadata.e));
            }
        } else {
            wgsl! { Y[index] = sign(val) * pow(abs(val), 'dtype(metadata.e)); }
        };

        kernel_builder.write_main(apply);
        Ok(kernel_builder.build()?)
    }
}

impl Kernel for PowfKernels {
    type Metadata = PowfMeta;

    fn kernel_name(&self) -> String {
        match self {
            PowfKernels::Standard(_) => "powf".to_string(),
        }
    }

    fn calculate_dispatch(&self, dst: &Tensor) -> Result<Workload, OperationError> {
        let PowfKernels::Standard(inner) = self;
        Ok(Workload::std(
            inner.src.shape().numel(),
            self.kernel_element(dst),
        ))
    }

    fn storage_bind_group_layout(
        &self,
        inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        if inplace {
            Ok(BindGroupLayoutDescriptor::unary_inplace())
        } else {
            Ok(BindGroupLayoutDescriptor::unary())
        }
    }

    fn metadata(&self, _: &Tensor, _: &KernelElement) -> Result<Self::Metadata, OperationError> {
        let PowfKernels::Standard(inner) = self;
        let numel = inner.src.shape().numel() as u32;
        Ok(PowfMeta { numel, e: inner.e })
    }

    fn kernel_element(&self, _dst: &Tensor) -> KernelElement {
        let PowfKernels::Standard(inner) = self;
        let a_rank = inner.src.shape().rank();
        let N = if a_rank > 0 {
            inner.src.shape()[a_rank - 1]
        } else {
            1
        };

        if N % 4 == 0 {
            KernelElement::Vec4
        } else if N % 2 == 0 {
            KernelElement::Vec2
        } else {
            KernelElement::Scalar
        }
    }

    fn build_kernel(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let kernel_element = self.kernel_element(dst);
        let PowfKernels::Standard(inner) = self;
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
    use crate::{shape, Device, DeviceRequest, Tensor};

    fn ground_truth(a: &Tensor, e: f32) -> anyhow::Result<Tensor> {
        let func_prg = r#"
import torch
def powf(a, e):
    a_tensor = torch.from_numpy(a)
    sign = torch.sign(a_tensor)
    return (torch.pow(torch.abs(a_tensor), e) * sign).numpy()
"#
        .to_string();

        let prg = func_prg;

        run_py_prg(prg.to_string(), &[a], &[&e], a.dtype())
    }

    fn run_powf_trial(problem: PowfProblem, device: Device) {
        let PowfProblem { B, M, N, e } = problem;
        let a = Tensor::randn::<f32>(0., 1., shape![B, M, N], Device::CPU);
        let ground = ground_truth(&a, e).unwrap();

        let a_gpu = a.to(&device).unwrap();
        let b = a_gpu.powf(e).unwrap();

        let ours = b.to(&Device::CPU).unwrap();

        ground.all_close(&ours, 1e-5, 1e-5).unwrap();
    }

    #[derive(Arbitrary, Debug)]
    struct PowfProblem {
        #[strategy(1..=128usize)]
        B: usize,
        #[strategy(1..=128usize)]
        M: usize,
        #[strategy(1..=128usize)]
        N: usize,
        #[strategy(-10.0f32..=10.0f32)]
        e: f32,
    }

    #[proptest(cases = 16)]
    fn test_powf(prob: PowfProblem) {
        let PowfProblem { B, M, N, e } = prob;
        println!("B = {}, M = {}, N = {}, e = {}", B, M, N, e);
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        run_powf_trial(prob, device);
    }
}
