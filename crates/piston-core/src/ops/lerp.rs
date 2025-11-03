use derive_new::new;
use half::f16;
use inline_wgsl::wgsl;
use piston_macros::IrFields;

use crate::{
    Array, BindingMode, BuiltIn, DType, DynKernelMetadata, GPUOperation, InvariantError, Kernel,
    KernelElement, KernelRenderable, KernelSource, OpGuards, OpTensor, Operation, OperationError,
    RVec, Scalar, Shape, StorageView, Stride, TensorTypeOrScalarEnum, Vec2, Vec4,
    WgslKernelBuilder, WgslPrimitive, WorkgroupSize, Workload,
    gpu::{BindGroupLayoutDescriptor, dtype::WgslDType},
    rvec,
};

#[derive(new, Debug, Clone, IrFields)]
pub struct Lerp {
    pub input: OpTensor,
    pub end: OpTensor,
    pub weight: TensorTypeOrScalarEnum<OpTensor>,
}

impl KernelRenderable for LerpKernels {
    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        inplace: bool,
    ) -> Result<(), OperationError> {
        let LerpKernels::Standard(inner) = self;
        if inplace {
            builder.register_storage("Input", BindingMode::ReadWrite, Array::<P>::default());
            builder.register_storage("End", BindingMode::ReadOnly, Array::<P>::default());
            if let TensorTypeOrScalarEnum::Tensor(_) = &inner.weight {
                builder.register_storage("Weight", BindingMode::ReadOnly, Array::<P>::default());
            }
        } else {
            builder.register_storage("Input", BindingMode::ReadOnly, Array::<P>::default());
            builder.register_storage("End", BindingMode::ReadOnly, Array::<P>::default());
            if let TensorTypeOrScalarEnum::Tensor(_) = &inner.weight {
                builder.register_storage("Weight", BindingMode::ReadOnly, Array::<P>::default());
            }
            builder.register_storage("Output", BindingMode::ReadWrite, Array::<P>::default());
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
        });

        let LerpKernels::Standard(inner) = self;
        let lerp_expression = match &inner.weight {
            TensorTypeOrScalarEnum::Tensor(_) => {
                wgsl! {
                    fma(Weight[index], (End[index] - Input[index]), Input[index])
                }
            }
            TensorTypeOrScalarEnum::Scalar(_) => {
                wgsl! {
                    fma('dtype(metadata.weight), (End[index] - Input[index]), Input[index])
                }
            }
        };

        let assignment = if inplace {
            wgsl! {
                Input[index] = 'lerp_expression;
            }
        } else {
            wgsl! {
                Output[index] = 'lerp_expression;
            }
        };

        kernel_builder.write_main(assignment);
        Ok(kernel_builder.build()?)
    }
}

impl Lerp {
    pub fn input(&self) -> &OpTensor {
        &self.input
    }

    pub fn end(&self) -> &OpTensor {
        &self.end
    }

    pub fn weight(&self) -> &TensorTypeOrScalarEnum<OpTensor> {
        &self.weight
    }
}

impl OpGuards for Lerp {
    fn check_shapes(&self) {
        // All tensors should be broadcastable to the same shape
        let mut shapes = vec![self.input.shape(), self.end.shape()];
        if let TensorTypeOrScalarEnum::Tensor(weight) = &self.weight {
            shapes.push(weight.shape());
        }
        let broadcasted = Shape::multi_broadcast(&shapes);
        assert!(broadcasted.is_some());
    }

    fn check_dtypes(&self) {
        assert_eq!(self.input.dtype(), self.end.dtype());
        if let TensorTypeOrScalarEnum::Tensor(weight) = &self.weight {
            assert_eq!(self.input.dtype(), weight.dtype());
        }
    }
}

impl Operation for Lerp {
    fn name(&self) -> &'static str {
        "Lerp"
    }

    fn compute_view(&self) -> Result<StorageView, OperationError> {
        let mut shapes = vec![self.input.shape(), self.end.shape()];
        if let TensorTypeOrScalarEnum::Tensor(weight) = &self.weight {
            shapes.push(weight.shape());
        }

        let broadcasted = Shape::multi_broadcast(&shapes);
        if broadcasted.is_none() {
            let failed = shapes.iter().map(|s| (*s).clone()).collect::<Vec<_>>();
            return Err(InvariantError::BroadcastingFailed(failed).into());
        }

        let broadcasted = broadcasted.unwrap();
        let ostride = Stride::from(&broadcasted);
        Ok(StorageView::new(broadcasted, self.input.dtype(), ostride))
    }

    #[inline]
    fn srcs(&self) -> RVec<&OpTensor> {
        if let TensorTypeOrScalarEnum::Tensor(weight) = &self.weight {
            rvec![&self.input, &self.end, weight]
        } else {
            rvec![&self.input, &self.end]
        }
    }

    fn supports_inplace(&self) -> bool {
        true
    }
}

impl GPUOperation for Lerp {
    type KernelEnum = LerpKernels;

    fn select_kernel(&self) -> Self::KernelEnum {
        LerpKernels::Standard(self.clone())
    }
}

pub enum LerpKernels {
    Standard(Lerp),
}

impl Kernel for LerpKernels {
    type Metadata = DynKernelMetadata;

    fn storage_bind_group_layout(
        &self,
        inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        let LerpKernels::Standard(inner) = self;
        match &inner.weight {
            TensorTypeOrScalarEnum::Tensor(_) => {
                if inplace {
                    Ok(BindGroupLayoutDescriptor::ternary_inplace())
                } else {
                    Ok(BindGroupLayoutDescriptor::ternary())
                }
            }
            TensorTypeOrScalarEnum::Scalar(_) => {
                if inplace {
                    Ok(BindGroupLayoutDescriptor::binary_inplace())
                } else {
                    Ok(BindGroupLayoutDescriptor::binary())
                }
            }
        }
    }

    fn kernel_name(&self) -> String {
        "lerp".to_string()
    }

    fn metadata(
        &self,
        dst: &OpTensor,
        _: &KernelElement,
    ) -> Result<Self::Metadata, OperationError> {
        let LerpKernels::Standard(inner) = self;
        let mut dyn_meta = DynKernelMetadata::new();
        dyn_meta.add_field("numel", dst.shape().numel() as u32);
        if let TensorTypeOrScalarEnum::Scalar(value) = &inner.weight {
            if inner.input.dtype().is_float() {
                dyn_meta.add_field("weight", *value);
            } else {
                dyn_meta.add_field("weight", *value as i32);
            }
        }
        Ok(dyn_meta)
    }

    fn kernel_element(&self, dst: &OpTensor) -> KernelElement {
        let numel = dst.shape().numel();

        if numel.is_multiple_of(4) {
            KernelElement::Vec4
        } else if numel.is_multiple_of(2) {
            KernelElement::Vec2
        } else {
            KernelElement::Scalar
        }
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
        let LerpKernels::Standard(inner) = self;
        let kernel_element = self.kernel_element(dst);
        match (inner.input.dtype(), &kernel_element) {
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
                inner.input.dtype(),
                kernel_element
            ))),
        }
    }
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use crate::{Device, DeviceRequest, Shape, Tensor, randn, test_util::run_py_prg};
    use proptest::arbitrary::any;
    use test_strategy::{Arbitrary, proptest};

    #[derive(Arbitrary, Debug)]
    struct LerpProblem {
        #[any(vec![1..=4, 1..=4, 1..=1, 1..=256])]
        shape: Shape,
    }

    #[derive(Arbitrary, Debug)]
    struct LerpScalarProblem {
        #[any(vec![1..=4, 1..=4, 1..=1, 1..=256])]
        shape: Shape,
        #[strategy(any::<f32>())]
        weight: f32,
    }

    fn ground_truth_tensor(
        input: &Tensor,
        end: &Tensor,
        weight: &Tensor,
    ) -> anyhow::Result<Tensor> {
        let prg = r#"
import torch
def lerp(input, end, weight):
    return torch.lerp(torch.from_numpy(input), torch.from_numpy(end), torch.from_numpy(weight)).numpy()
"#;
        run_py_prg(prg.to_string(), &[input, end, weight], &[], input.dtype())
    }

    fn ground_truth_scalar(input: &Tensor, end: &Tensor, weight: f32) -> anyhow::Result<Tensor> {
        let prg = r#"
import torch
def lerp(input, end, weight):
    return torch.lerp(torch.from_numpy(input), torch.from_numpy(end), weight).numpy()
"#;
        run_py_prg(prg.to_string(), &[input, end], &[&weight], input.dtype())
    }

    fn run_lerp_tensor_trial(prob: LerpProblem, device: Device) -> anyhow::Result<()> {
        let LerpProblem { shape } = prob;
        let input = randn(shape.clone(), None, None, Default::default())?;
        let end = randn(shape.clone(), None, None, Default::default())?;
        let weight = randn(shape, None, None, Default::default())?;
        let ground = ground_truth_tensor(&input, &end, &weight)?;

        let input = input.to(&device)?;
        let end = end.to(&device)?;
        let weight = weight.to(&device)?;

        let result = input.lerp(end, weight)?;
        let result = result.to(&Device::CPU)?;
        ground.all_close(&result, 1e-4, 1e-4)?;
        Ok(())
    }

    fn run_lerp_scalar_trial(prob: LerpScalarProblem, device: Device) -> anyhow::Result<()> {
        let LerpScalarProblem { shape, weight } = prob;
        let input = randn(shape.clone(), None, None, Default::default())?;
        let end = randn(shape, None, None, Default::default())?;
        let ground = ground_truth_scalar(&input, &end, weight)?;

        let input = input.to(&device)?;
        let end = end.to(&device)?;

        let result = input.lerp(end, weight)?;
        let result = result.to(&Device::CPU)?;
        ground.all_close(&result, 1e-4, 1e-4)?;
        Ok(())
    }

    #[proptest(cases = 8)]
    fn test_lerp_tensor_gpu(prob: LerpProblem) {
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        run_lerp_tensor_trial(prob, device).unwrap();
    }

    #[proptest(cases = 8)]
    fn test_lerp_scalar_gpu(prob: LerpScalarProblem) {
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        run_lerp_scalar_trial(prob, device).unwrap();
    }
}
