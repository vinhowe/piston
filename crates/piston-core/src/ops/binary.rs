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
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Maximum,
}

impl BinaryOp {
    pub fn kernel_name(&self) -> &'static str {
        match self {
            BinaryOp::Add => "add",
            BinaryOp::Sub => "sub",
            BinaryOp::Mul => "mul",
            BinaryOp::Div => "div",
            BinaryOp::Maximum => "maximum",
        }
    }

    pub fn kernel_expression(&self, op1: &'static str, op2: &'static str) -> String {
        match self {
            BinaryOp::Add => wgsl! { 'op1 + 'op2 },
            BinaryOp::Sub => wgsl! { 'op1 - 'op2 },
            BinaryOp::Mul => wgsl! { 'op1 * 'op2 },
            BinaryOp::Div => wgsl! { 'op1 / 'op2 },
            BinaryOp::Maximum => wgsl! { max('op1, 'op2) },
        }
    }
}

#[derive(new, Debug, Clone, IrFields)]
pub struct Binary {
    pub lhs: OpTensor,
    pub rhs: OpTensor,
    pub op: BinaryOp,
}

impl KernelRenderable for BinaryKernels {
    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        inplace: bool,
    ) -> Result<(), OperationError> {
        if inplace {
            builder.register_storage("A", BindingMode::ReadWrite, Array::<P>::default());
            builder.register_storage("B", BindingMode::ReadOnly, Array::<P>::default());
        } else {
            builder.register_storage("A", BindingMode::ReadOnly, Array::<P>::default());
            builder.register_storage("B", BindingMode::ReadOnly, Array::<P>::default());
            builder.register_storage("Y", BindingMode::ReadWrite, Array::<P>::default());
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

        let BinaryKernels::Standard(inner) = self;
        let apply = if inplace {
            let expression = inner.op.kernel_expression("val", "B[index]");
            wgsl! {
                let val = A[index];
                A[index] = 'expression;
            }
        } else {
            let expression = inner.op.kernel_expression("A[index]", "B[index]");
            wgsl! { Y[index] = 'expression; }
        };
        kernel_builder.write_main(apply);
        Ok(kernel_builder.build()?)
    }
}

impl Binary {
    pub fn op(&self) -> &BinaryOp {
        &self.op
    }

    pub fn lhs(&self) -> &OpTensor {
        &self.lhs
    }

    pub fn rhs(&self) -> &OpTensor {
        &self.rhs
    }
}

#[derive(Debug, ShaderType, WgslMetadata)]
pub struct BinaryMeta {
    numel: u32,
}

impl OpGuards for Binary {
    fn check_shapes(&self) {
        let shapes = [self.lhs.shape(), self.rhs.shape()];
        let broadcasted = Shape::multi_broadcast(&shapes);
        assert!(broadcasted.is_some());
    }

    fn check_dtypes(&self) {
        assert_eq!(self.lhs.dtype(), self.rhs.dtype());
    }
}

impl Operation for Binary {
    fn name(&self) -> &'static str {
        match self.op {
            BinaryOp::Add => "Add",
            BinaryOp::Sub => "Sub",
            BinaryOp::Mul => "Mul",
            BinaryOp::Div => "Div",
            BinaryOp::Maximum => "Maximum",
        }
    }

    fn compute_view(&self) -> Result<StorageView, OperationError> {
        let lhs = &self.lhs;
        let rhs = &self.rhs;
        let shapes = &[lhs.shape(), rhs.shape()];
        if lhs.is_scalar() || rhs.is_scalar() {
            let other = if lhs.is_scalar() { rhs } else { lhs };
            return Ok(other.storage_view().clone());
        }
        let broadcasted = Shape::multi_broadcast(shapes);
        if broadcasted.is_none() {
            let failed = shapes.iter().map(|s| (*s).clone()).collect::<Vec<_>>();
            return Err(InvariantError::BroadcastingFailed(failed).into());
        }
        let broadcasted = broadcasted.unwrap();
        let ostride = Stride::from(&broadcasted);
        Ok(StorageView::new(broadcasted, lhs.dtype(), ostride))
    }

    #[inline]
    fn srcs(&self) -> RVec<&OpTensor> {
        rvec![&self.lhs, &self.rhs]
    }

    fn supports_inplace(&self) -> bool {
        true
    }
}

impl GPUOperation for Binary {
    type KernelEnum = BinaryKernels;

    fn select_kernel(&self) -> Self::KernelEnum {
        BinaryKernels::Standard(self.clone())
    }
}

pub enum BinaryKernels {
    Standard(Binary),
}

impl Kernel for BinaryKernels {
    type Metadata = BinaryMeta;

    fn storage_bind_group_layout(
        &self,
        inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        if inplace {
            Ok(BindGroupLayoutDescriptor::binary_inplace())
        } else {
            Ok(BindGroupLayoutDescriptor::binary())
        }
    }

    fn kernel_name(&self) -> String {
        match self {
            BinaryKernels::Standard(k) => k.op.kernel_name().to_string(),
        }
    }

    fn metadata(
        &self,
        dst: &OpTensor,
        _: &KernelElement,
    ) -> Result<Self::Metadata, OperationError> {
        let numel = dst.shape().numel() as _;
        let meta = BinaryMeta { numel };

        Ok(meta)
    }

    fn kernel_element(&self, dst: &OpTensor) -> KernelElement {
        let numel = dst.shape().numel();

        if numel % 4 == 0 {
            KernelElement::Vec4
        } else if numel % 2 == 0 {
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
        let BinaryKernels::Standard(inner) = self;
        let kernel_element = self.kernel_element(dst);
        match (inner.lhs.dtype(), &kernel_element) {
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
                inner.lhs.dtype(),
                kernel_element
            ))),
        }
    }
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use crate::{test_util::run_py_prg, BinaryOp, Device, DeviceRequest, OpTensor, Shape};
    use test_strategy::{proptest, Arbitrary};

    #[derive(Arbitrary, Debug)]
    struct BinaryProblem {
        op: BinaryOp,
        #[any(vec![1..=4, 1..=4, 1..=1, 1..=256])]
        shape: Shape,
    }

    fn ground_truth(a: &OpTensor, b: &OpTensor, op: &BinaryOp) -> anyhow::Result<OpTensor> {
        let kn = op.kernel_name();
        let prg = format!(
            r#"
import torch
def {}(a, b):
    return torch.{}(torch.from_numpy(a), torch.from_numpy(b)).numpy()
"#,
            kn, kn
        );
        run_py_prg(prg.to_string(), &[a, b], &[], a.dtype())
    }

    fn run_binary_trial(prob: BinaryProblem, device: Device) -> anyhow::Result<()> {
        let cpu_device = Device::request_device(DeviceRequest::CPU)?;
        let BinaryProblem { op, shape } = prob;
        let a = OpTensor::randn::<f32, _>(0., 1., shape.clone(), cpu_device.clone(), false)?;
        let b = OpTensor::randn::<f32, _>(0., 1., shape, cpu_device.clone(), false)?;
        let ground = ground_truth(&a, &b, &op)?;

        let a = a.to(&device)?;
        let b = b.to(&device)?;

        let c = match op {
            BinaryOp::Add => a.add(b)?,
            BinaryOp::Sub => a.sub(b)?,
            BinaryOp::Mul => a.mul(b)?,
            BinaryOp::Div => a.div(b)?,
            BinaryOp::Maximum => a.maximum(b)?,
        };

        let d = c.to(&Device::CPU)?;
        ground.all_close(&d, 1e-4, 1e-4)?;
        Ok(())
    }

    #[proptest(cases = 8)]
    fn test_binary_gpu(prob: BinaryProblem) {
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        run_binary_trial(prob, device).unwrap();
    }

    #[proptest(cases = 8)]
    fn test_binary_cpu(prob: BinaryProblem) {
        let device = Device::request_device(DeviceRequest::CPU).unwrap();
        run_binary_trial(prob, device).unwrap();
    }
}
