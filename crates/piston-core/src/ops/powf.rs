use derive_new::new;
use half::f16;
use inline_wgsl::wgsl;
use piston_macros::IrFields;

use crate::{
    gpu::{dtype::WgslDType, BindGroupLayoutDescriptor},
    rvec, Array, BindingMode, BuiltIn, DType, DynKernelMetadata, GPUOperation, InvariantError,
    Kernel, KernelElement, KernelRenderable, KernelSource, OpGuards, OpTensor, Operation,
    OperationError, RVec, Scalar, Shape, StorageView, Stride, TensorTypeOrScalarEnum, Vec2, Vec4,
    WgslKernelBuilder, WgslPrimitive, WorkgroupSize, Workload,
};

#[derive(new, Debug, Clone, IrFields)]
pub struct Powf {
    pub src: OpTensor,
    pub e: TensorTypeOrScalarEnum<OpTensor>,
}

impl OpGuards for Powf {
    fn check_shapes(&self) {
        if let TensorTypeOrScalarEnum::Tensor(e) = &self.e {
            let shapes = [self.src.shape(), e.shape()];
            let broadcasted = Shape::multi_broadcast(&shapes);
            assert!(broadcasted.is_some());
        }
    }

    fn check_dtypes(&self) {
        let a = &self.src;
        assert!(matches!(a.dtype(), crate::DType::F32));
        if let TensorTypeOrScalarEnum::Tensor(e) = &self.e {
            assert_eq!(self.src.dtype(), e.dtype());
        }
    }
}

impl Operation for Powf {
    fn name(&self) -> &'static str {
        "Powf"
    }

    fn compute_view(&self) -> Result<StorageView, OperationError> {
        if let TensorTypeOrScalarEnum::Tensor(e) = &self.e {
            let shapes = &[self.src.shape(), e.shape()];
            if self.src.is_scalar() || e.is_scalar() {
                let other = if self.src.is_scalar() { e } else { &self.src };
                return Ok(other.storage_view().clone());
            }
            let broadcasted = Shape::multi_broadcast(shapes);
            if broadcasted.is_none() {
                let failed = shapes.iter().map(|s| (*s).clone()).collect::<Vec<_>>();
                return Err(InvariantError::BroadcastingFailed(failed).into());
            }
            let broadcasted = broadcasted.unwrap();
            let ostride = Stride::from(&broadcasted);
            Ok(StorageView::new(broadcasted, self.src.dtype(), ostride))
        } else {
            Ok(self.src.storage_view().clone())
        }
    }

    #[inline]
    fn srcs(&self) -> RVec<&OpTensor> {
        if let TensorTypeOrScalarEnum::Tensor(e) = &self.e {
            rvec![&self.src, e]
        } else {
            rvec![&self.src]
        }
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
        let PowfKernels::Standard(inner) = self;

        if inplace {
            builder.register_storage("X", BindingMode::ReadWrite, arr);
        } else {
            builder.register_storage("X", BindingMode::ReadOnly, arr);
        }

        if let TensorTypeOrScalarEnum::Tensor(_) = &inner.e {
            builder.register_storage("E", BindingMode::ReadOnly, arr);
        }

        if !inplace {
            builder.register_storage("Y", BindingMode::ReadWrite, arr);
        }

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

        let PowfKernels::Standard(inner) = self;

        // pow(x, e) is undefined for x < 0 in Dawn, but apparently not in wgpu,
        // but only when the compiler doesn't have enough information to coerce
        // e into an integer. We supply e through the metadata, so, at compile-time,
        // its type is unknown.
        //
        // Multiplying by the sign is a fix to make this shader work correctly in Chrome.
        let exponent_expr = match &inner.e {
            TensorTypeOrScalarEnum::Tensor(_) => "E[index]".to_string(),
            TensorTypeOrScalarEnum::Scalar(_) => wgsl! { 'dtype(metadata.e) },
        };

        let apply = if inplace {
            wgsl! {
                X[index] = sign(val) * pow(abs(val), 'exponent_expr);
            }
        } else {
            wgsl! { Y[index] = sign(val) * pow(abs(val), 'exponent_expr); }
        };

        kernel_builder.write_main(apply);
        Ok(kernel_builder.build()?)
    }
}

impl Kernel for PowfKernels {
    type Metadata = DynKernelMetadata;

    fn kernel_name(&self) -> String {
        match self {
            PowfKernels::Standard(_) => "powf".to_string(),
        }
    }

    fn calculate_dispatch(&self, dst: &OpTensor) -> Result<Workload, OperationError> {
        let PowfKernels::Standard(_) = self;
        Ok(Workload::std(dst.shape().numel(), self.kernel_element(dst)))
    }

    fn storage_bind_group_layout(
        &self,
        inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        let PowfKernels::Standard(inner) = self;
        match &inner.e {
            TensorTypeOrScalarEnum::Tensor(_) => {
                if inplace {
                    Ok(BindGroupLayoutDescriptor::binary_inplace())
                } else {
                    Ok(BindGroupLayoutDescriptor::binary())
                }
            }
            TensorTypeOrScalarEnum::Scalar(_) => {
                if inplace {
                    Ok(BindGroupLayoutDescriptor::unary_inplace())
                } else {
                    Ok(BindGroupLayoutDescriptor::unary())
                }
            }
        }
    }

    fn metadata(
        &self,
        dst: &OpTensor,
        _: &KernelElement,
    ) -> Result<Self::Metadata, OperationError> {
        let PowfKernels::Standard(inner) = self;
        let mut dyn_meta = DynKernelMetadata::new();
        dyn_meta.add_field("numel", dst.shape().numel() as u32);
        if let TensorTypeOrScalarEnum::Scalar(value) = &inner.e {
            dyn_meta.add_field("e", *value);
        }
        Ok(dyn_meta)
    }

    fn kernel_element(&self, _dst: &OpTensor) -> KernelElement {
        let PowfKernels::Standard(inner) = self;
        let a_rank = inner.src.shape().dim();
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
        dst: &OpTensor,
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
    use crate::{Device, DeviceRequest, OpTensor};

    fn ground_truth(a: &OpTensor, e: f32) -> anyhow::Result<OpTensor> {
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

    fn ground_truth_tensor(a: &OpTensor, e: &OpTensor) -> anyhow::Result<OpTensor> {
        let func_prg = r#"
import torch
def powf(a, e):
    a_tensor = torch.from_numpy(a)
    e_tensor = torch.from_numpy(e)
    sign = torch.sign(a_tensor)
    return (torch.pow(torch.abs(a_tensor), e_tensor) * sign).numpy()
"#
        .to_string();

        let prg = func_prg;

        run_py_prg(prg.to_string(), &[a, e], &[], a.dtype())
    }

    fn run_powf_trial(problem: PowfProblem, device: Device) {
        let PowfProblem { B, M, N, e } = problem;
        let a = OpTensor::randn::<f32, _>(0., 1., (B, M, N), Device::CPU, false).unwrap();
        let ground = ground_truth(&a, e).unwrap();

        let a_gpu = a.to(&device).unwrap();
        let b = a_gpu.pow(e).unwrap();

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

    #[derive(Arbitrary, Debug)]
    struct PowfTensorProblem {
        #[strategy(1..=32usize)]
        B: usize,
        #[strategy(1..=32usize)]
        M: usize,
        #[strategy(1..=32usize)]
        N: usize,
    }

    fn run_powf_tensor_trial(problem: PowfTensorProblem, device: Device) {
        let PowfTensorProblem { B, M, N } = problem;
        let a = OpTensor::randn::<f32, _>(0., 1., (B, M, N), Device::CPU, false).unwrap();
        let e = OpTensor::randn::<f32, _>(0.1, 2.0, (B, M, N), Device::CPU, false).unwrap();
        let ground = ground_truth_tensor(&a, &e).unwrap();

        let a_gpu = a.to(&device).unwrap();
        let e_gpu = e.to(&device).unwrap();
        let b = a_gpu.pow(e_gpu).unwrap();

        let ours = b.to(&Device::CPU).unwrap();

        ground.all_close(&ours, 1e-4, 1e-4).unwrap();
    }

    #[proptest(cases = 16)]
    fn test_powf_scalar(prob: PowfProblem) {
        let PowfProblem { B, M, N, e } = prob;
        println!("B = {}, M = {}, N = {}, e = {}", B, M, N, e);
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        run_powf_trial(prob, device);
    }

    #[proptest(cases = 8)]
    fn test_powf_tensor(prob: PowfTensorProblem) {
        let PowfTensorProblem { B, M, N } = prob;
        println!("B = {}, M = {}, N = {}", B, M, N);
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        run_powf_tensor_trial(prob, device);
    }
}
