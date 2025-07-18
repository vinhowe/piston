use derive_new::new;
use encase::ShaderType;
use half::f16;
use inline_wgsl::wgsl;
use piston_macros::{IrFields, WgslMetadata};

use crate::{
    gpu::{dtype::WgslDType, BindGroupLayoutDescriptor},
    rvec, Array, BindingMode, BuiltIn, DType, GPUOperation, InvariantError, Kernel, KernelElement,
    KernelRenderable, KernelSource, OpGuards, OpTensor, Operation, OperationError, RVec, Scalar,
    Shape, StorageView, Stride, Vec2, Vec4, WgslKernelBuilder, WgslPrimitive, WorkgroupSize,
    Workload,
};
#[cfg(test)]
use test_strategy::Arbitrary;

#[cfg_attr(test, derive(Arbitrary))]
#[derive(Debug, Clone, Hash, IrFields)]
pub enum TernaryOp {
    Addcdiv,
    Addcmul,
}

impl TernaryOp {
    pub fn kernel_name(&self) -> &'static str {
        match self {
            TernaryOp::Addcdiv => "addcdiv",
            TernaryOp::Addcmul => "addcmul",
        }
    }

    pub fn kernel_expression(&self) -> String {
        match self {
            TernaryOp::Addcdiv => wgsl! {
                input + metadata.value * (tensor1 / tensor2)
            },
            TernaryOp::Addcmul => wgsl! {
                input + metadata.value * (tensor1 * tensor2)
            },
        }
    }
}

#[derive(new, Debug, Clone, IrFields)]
pub struct Ternary {
    pub input: OpTensor,
    pub tensor1: OpTensor,
    pub tensor2: OpTensor,
    pub value: f32,
    pub op: TernaryOp,
}

impl KernelRenderable for TernaryKernels {
    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        inplace: bool,
    ) -> Result<(), OperationError> {
        if inplace {
            builder.register_storage("Input", BindingMode::ReadWrite, Array::<P>::default());
            builder.register_storage("Tensor1", BindingMode::ReadOnly, Array::<P>::default());
            builder.register_storage("Tensor2", BindingMode::ReadOnly, Array::<P>::default());
        } else {
            builder.register_storage("Input", BindingMode::ReadOnly, Array::<P>::default());
            builder.register_storage("Tensor1", BindingMode::ReadOnly, Array::<P>::default());
            builder.register_storage("Tensor2", BindingMode::ReadOnly, Array::<P>::default());
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

        kernel_builder.write_main(wgsl! {
            let x_offset = workgroup_id.x * 64u;
            let index = (workgroup_id.y * num_workgroups.x * 64u) + x_offset + local_invocation_index;
            if (index >= metadata.numel / 'N) {
                return;
            }
        });

        let TernaryKernels::Standard(inner) = self;
        let expression = inner.op.kernel_expression();
        let assignment_expression = if inplace {
            wgsl! {
                Input[index] = 'expression;
            }
        } else {
            wgsl! {
                Output[index] = 'expression;
            }
        };

        kernel_builder.write_main(wgsl! {
            let input = Input[index];
            let tensor1 = Tensor1[index];
            let tensor2 = Tensor2[index];
            'assignment_expression
        });

        Ok(kernel_builder.build()?)
    }
}

impl Ternary {
    pub fn op(&self) -> &TernaryOp {
        &self.op
    }

    pub fn input(&self) -> &OpTensor {
        &self.input
    }

    pub fn tensor1(&self) -> &OpTensor {
        &self.tensor1
    }

    pub fn tensor2(&self) -> &OpTensor {
        &self.tensor2
    }

    pub fn value(&self) -> f32 {
        self.value
    }
}

#[derive(Debug, ShaderType, WgslMetadata)]
pub struct TernaryMeta {
    numel: u32,
    value: f32,
}

impl OpGuards for Ternary {
    fn check_shapes(&self) {
        let shapes = [
            self.input.shape(),
            self.tensor1.shape(),
            self.tensor2.shape(),
        ];
        let broadcasted = Shape::multi_broadcast(&shapes);
        assert!(broadcasted.is_some());
    }

    fn check_dtypes(&self) {
        assert_eq!(self.input.dtype(), self.tensor1.dtype());
        assert_eq!(self.input.dtype(), self.tensor2.dtype());
    }
}

impl Operation for Ternary {
    fn name(&self) -> &'static str {
        match self.op {
            TernaryOp::Addcdiv => "Addcdiv",
            TernaryOp::Addcmul => "Addcmul",
        }
    }

    fn compute_view(&self) -> Result<StorageView, OperationError> {
        let input = &self.input;
        let tensor1 = &self.tensor1;
        let tensor2 = &self.tensor2;
        let shapes = &[input.shape(), tensor1.shape(), tensor2.shape()];

        let broadcasted = Shape::multi_broadcast(shapes);
        if broadcasted.is_none() {
            let failed = shapes.iter().map(|s| (*s).clone()).collect::<Vec<_>>();
            return Err(InvariantError::BroadcastingFailed(failed).into());
        }

        let broadcasted = broadcasted.unwrap();
        let ostride = Stride::from(&broadcasted);
        Ok(StorageView::new(broadcasted, input.dtype(), ostride))
    }

    #[inline]
    fn srcs(&self) -> RVec<&OpTensor> {
        rvec![&self.input, &self.tensor1, &self.tensor2]
    }

    fn supports_inplace(&self) -> bool {
        true
    }
}

impl GPUOperation for Ternary {
    type KernelEnum = TernaryKernels;

    fn select_kernel(&self) -> Self::KernelEnum {
        TernaryKernels::Standard(self.clone())
    }
}

pub enum TernaryKernels {
    Standard(Ternary),
}

impl Kernel for TernaryKernels {
    type Metadata = TernaryMeta;

    fn storage_bind_group_layout(
        &self,
        inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        if inplace {
            Ok(BindGroupLayoutDescriptor::ternary_inplace())
        } else {
            Ok(BindGroupLayoutDescriptor::ternary())
        }
    }

    fn kernel_name(&self) -> String {
        match self {
            TernaryKernels::Standard(k) => k.op.kernel_name().to_string(),
        }
    }

    fn metadata(
        &self,
        dst: &OpTensor,
        _: &KernelElement,
    ) -> Result<Self::Metadata, OperationError> {
        let numel = dst.shape().numel() as _;
        let TernaryKernels::Standard(inner) = self;
        Ok(TernaryMeta {
            numel,
            value: inner.value,
        })
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
        let TernaryKernels::Standard(inner) = self;
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
#[cfg(test)]
mod tests {
    use crate::{test_util::run_py_prg, Device, DeviceRequest, OpTensor, Shape, TernaryOp};
    use test_strategy::{proptest, Arbitrary};

    #[derive(Arbitrary, Debug)]
    struct TernaryProblem {
        op: TernaryOp,
        #[any(vec![1..=4, 1..=4, 1..=1, 1..=256])]
        shape: Shape,
    }

    fn ground_truth(
        input: &OpTensor,
        tensor1: &OpTensor,
        tensor2: &OpTensor,
        value: f32,
        op: &TernaryOp,
    ) -> anyhow::Result<OpTensor> {
        let kn = op.kernel_name();
        let prg = match op {
            TernaryOp::Addcdiv => format!(
                r#"
import torch
def {kn}(input, tensor1, tensor2):
    return torch.addcdiv(torch.from_numpy(input), torch.from_numpy(tensor1), torch.from_numpy(tensor2), value={value}).numpy()
"#,
            ),
            TernaryOp::Addcmul => format!(
                r#"
import torch
def {kn}(input, tensor1, tensor2):
    return torch.addcmul(torch.from_numpy(input), torch.from_numpy(tensor1), torch.from_numpy(tensor2), value={value}).numpy()
"#,
            ),
        };
        run_py_prg(
            prg.to_string(),
            &[input, tensor1, tensor2],
            &[],
            input.dtype(),
        )
    }

    fn run_ternary_trial(prob: TernaryProblem, device: Device) -> anyhow::Result<()> {
        let cpu_device = Device::request_device(DeviceRequest::CPU)?;
        let TernaryProblem { op, shape } = prob;
        let input = OpTensor::randn::<f32, _>(0., 1., shape.clone(), cpu_device.clone(), false)?;
        let tensor1 = OpTensor::randn::<f32, _>(0., 1., shape.clone(), cpu_device.clone(), false)?;
        let tensor2 = OpTensor::randn::<f32, _>(0., 1., shape, cpu_device.clone(), false)?;
        let value = 0.5;
        let ground = ground_truth(&input, &tensor1, &tensor2, value, &op)?;

        let input = input.to(&device)?;
        let tensor1 = tensor1.to(&device)?;
        let tensor2 = tensor2.to(&device)?;

        let c = match op {
            TernaryOp::Addcdiv => input.addcdiv(tensor1, tensor2, value)?,
            TernaryOp::Addcmul => input.addcmul(tensor1, tensor2, value)?,
        };

        let d = c.to(&Device::CPU)?;
        ground.all_close(&d, 1e-4, 1e-4)?;
        Ok(())
    }

    #[proptest(cases = 8)]
    fn test_ternary_gpu(prob: TernaryProblem) {
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        run_ternary_trial(prob, device).unwrap();
    }
}
