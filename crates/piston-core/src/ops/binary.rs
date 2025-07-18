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
#[cfg(test)]
use test_strategy::Arbitrary;

#[cfg_attr(test, derive(Arbitrary))]
#[derive(Debug, Clone, Hash, IrFields)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Maximum,
    Minimum,
}

impl BinaryOp {
    pub fn kernel_name(&self) -> &'static str {
        match self {
            BinaryOp::Add => "add",
            BinaryOp::Sub => "sub",
            BinaryOp::Mul => "mul",
            BinaryOp::Div => "div",
            BinaryOp::Pow => "pow",
            BinaryOp::Maximum => "maximum",
            BinaryOp::Minimum => "minimum",
        }
    }

    pub fn kernel_expression_tensor(&self, op1: &'static str, op2: &'static str) -> String {
        match self {
            BinaryOp::Add => wgsl! { 'op1 + 'op2 },
            BinaryOp::Sub => wgsl! { 'op1 - 'op2 },
            BinaryOp::Mul => wgsl! { 'op1 * 'op2 },
            BinaryOp::Div => wgsl! { 'op1 / 'op2 },
            BinaryOp::Pow => wgsl! { sign('op1) * pow(abs('op1), 'op2) },
            BinaryOp::Maximum => wgsl! { max('op1, 'op2) },
            BinaryOp::Minimum => wgsl! { min('op1, 'op2) },
        }
    }

    pub fn kernel_expression_scalar(&self, dtype: &str, op1: &'static str) -> String {
        match self {
            BinaryOp::Add => wgsl! { fma('op1, 'dtype(1.0), 'dtype(metadata.value)) },
            BinaryOp::Sub => wgsl! { fma('op1, 'dtype(1.0), -'dtype(metadata.value)) },
            BinaryOp::Mul => wgsl! { fma('op1, 'dtype(metadata.value), 'dtype(0.0)) },
            BinaryOp::Div => wgsl! { fma('op1, 'dtype(1.0 / metadata.value), 'dtype(0.0)) },
            BinaryOp::Pow => wgsl! { sign('op1) * pow(abs('op1), 'dtype(metadata.value)) },
            BinaryOp::Maximum => panic!("Maximum with scalar is not supported"),
            BinaryOp::Minimum => panic!("Minimum with scalar is not supported"),
        }
    }
}

#[derive(new, Debug, Clone, IrFields)]
pub struct Binary {
    pub lhs: OpTensor,
    pub rhs: TensorTypeOrScalarEnum<OpTensor>,
    pub op: BinaryOp,
}

impl KernelRenderable for BinaryKernels {
    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        inplace: bool,
    ) -> Result<(), OperationError> {
        let BinaryKernels::Standard(inner) = self;
        if inplace {
            builder.register_storage("A", BindingMode::ReadWrite, Array::<P>::default());
            if let TensorTypeOrScalarEnum::Tensor(_) = &inner.rhs {
                builder.register_storage("B", BindingMode::ReadOnly, Array::<P>::default());
            }
        } else {
            builder.register_storage("A", BindingMode::ReadOnly, Array::<P>::default());
            if let TensorTypeOrScalarEnum::Tensor(_) = &inner.rhs {
                builder.register_storage("B", BindingMode::ReadOnly, Array::<P>::default());
            }
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
        let dtype = P::render_type();

        kernel_builder.write_main(wgsl! {
            let x_offset = workgroup_id.x * 64u;
            let index = (workgroup_id.y * num_workgroups.x * 64u) + x_offset + local_invocation_index;
            if (index >= metadata.numel / 'N) {
                return;
            }
        });

        let BinaryKernels::Standard(inner) = self;
        let apply = if inplace {
            let expression = match &inner.rhs {
                TensorTypeOrScalarEnum::Tensor(_) => {
                    inner.op.kernel_expression_tensor("val", "B[index]")
                }
                TensorTypeOrScalarEnum::Scalar(_) => {
                    inner.op.kernel_expression_scalar(&dtype, "val")
                }
            };
            wgsl! {
                let val = A[index];
                A[index] = 'expression;
            }
        } else {
            let expression = match &inner.rhs {
                TensorTypeOrScalarEnum::Tensor(_) => {
                    inner.op.kernel_expression_tensor("A[index]", "B[index]")
                }
                TensorTypeOrScalarEnum::Scalar(_) => {
                    inner.op.kernel_expression_scalar(&dtype, "A[index]")
                }
            };
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

    pub fn rhs(&self) -> &TensorTypeOrScalarEnum<OpTensor> {
        &self.rhs
    }
}

impl OpGuards for Binary {
    fn check_shapes(&self) {
        if let TensorTypeOrScalarEnum::Tensor(rhs) = &self.rhs {
            let shapes = [self.lhs.shape(), rhs.shape()];
            let broadcasted = Shape::multi_broadcast(&shapes);
            assert!(broadcasted.is_some());
        }
    }

    fn check_dtypes(&self) {
        if let TensorTypeOrScalarEnum::Tensor(rhs) = &self.rhs {
            assert_eq!(self.lhs.dtype(), rhs.dtype());
        }
    }
}

impl Operation for Binary {
    fn name(&self) -> &'static str {
        match self.op {
            BinaryOp::Add => "Add",
            BinaryOp::Sub => "Sub",
            BinaryOp::Mul => "Mul",
            BinaryOp::Div => "Div",
            BinaryOp::Pow => "Pow",
            BinaryOp::Maximum => "Maximum",
            BinaryOp::Minimum => "Minimum",
        }
    }

    fn compute_view(&self) -> Result<StorageView, OperationError> {
        if let TensorTypeOrScalarEnum::Tensor(rhs) = &self.rhs {
            let shapes = &[self.lhs.shape(), rhs.shape()];
            if self.lhs.is_scalar() || rhs.is_scalar() {
                let other = if self.lhs.is_scalar() { rhs } else { &self.lhs };
                return Ok(other.storage_view().clone());
            }
            let broadcasted = Shape::multi_broadcast(shapes);
            if broadcasted.is_none() {
                let failed = shapes.iter().map(|s| (*s).clone()).collect::<Vec<_>>();
                return Err(InvariantError::BroadcastingFailed(failed).into());
            }
            let broadcasted = broadcasted.unwrap();
            let ostride = Stride::from(&broadcasted);
            Ok(StorageView::new(broadcasted, self.lhs.dtype(), ostride))
        } else {
            Ok(self.lhs.storage_view().clone())
        }
    }

    #[inline]
    fn srcs(&self) -> RVec<&OpTensor> {
        if let TensorTypeOrScalarEnum::Tensor(rhs) = &self.rhs {
            rvec![&self.lhs, rhs]
        } else {
            rvec![&self.lhs]
        }
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
    type Metadata = DynKernelMetadata;

    fn storage_bind_group_layout(
        &self,
        inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        let BinaryKernels::Standard(inner) = self;
        match &inner.rhs {
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
        let BinaryKernels::Standard(inner) = self;
        let mut dyn_meta = DynKernelMetadata::new();
        dyn_meta.add_field("numel", dst.shape().numel() as u32);
        if let TensorTypeOrScalarEnum::Scalar(value) = &inner.rhs {
            if inner.lhs.dtype().is_float() {
                dyn_meta.add_field("value", *value);
            } else {
                dyn_meta.add_field("value", *value as i32);
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
        let prg = if matches!(op, BinaryOp::Pow) {
            format!(
                r#"
import torch
def {kn}(a, b):
    a_tensor = torch.from_numpy(a)
    b_tensor = torch.from_numpy(b)
    sign = torch.sign(a_tensor)
    return (torch.pow(torch.abs(a_tensor), b_tensor) * sign).numpy()
"#,
            )
        } else {
            format!(
                r#"
import torch
def {kn}(a, b):
    return torch.{kn}(torch.from_numpy(a), torch.from_numpy(b)).numpy()
"#,
            )
        };
        run_py_prg(prg.to_string(), &[a, b], &[], a.dtype())
    }

    fn run_binary_trial(prob: BinaryProblem, device: Device) -> anyhow::Result<()> {
        if device.is_cpu() && matches!(prob.op, BinaryOp::Pow) {
            // Fail silently for now
            return Ok(());
        }
        let cpu_device = Device::request_device(DeviceRequest::CPU)?;
        let BinaryProblem { op, shape } = prob;
        let a = OpTensor::randn::<f32, _>(0., 1., shape.clone(), cpu_device.clone(), false)?;
        let b = OpTensor::randn::<f32, _>(0.1, 2.0, shape, cpu_device.clone(), false)?;
        let ground = ground_truth(&a, &b, &op)?;

        let a = a.to(&device)?;
        let b = b.to(&device)?;

        let c = match op {
            BinaryOp::Add => a.add(b)?,
            BinaryOp::Sub => a.sub(b)?,
            BinaryOp::Mul => a.mul(b)?,
            BinaryOp::Div => a.div(b)?,
            BinaryOp::Pow => a.pow(b)?,
            BinaryOp::Maximum => a.maximum(b)?,
            BinaryOp::Minimum => a.minimum(b)?,
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
