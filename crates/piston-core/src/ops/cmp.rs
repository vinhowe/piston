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
#[cfg(test)]
use test_strategy::Arbitrary;

#[cfg_attr(test, derive(Arbitrary))]
#[derive(Debug, Clone, Hash, IrFields)]
pub enum CmpOp {
    Eq,
    Ne,
    Le,
    Ge,
    Lt,
    Gt,
}

impl CmpOp {
    pub fn kernel_name(&self) -> &'static str {
        match self {
            CmpOp::Eq => "eq",
            CmpOp::Ne => "ne",
            CmpOp::Le => "le",
            CmpOp::Ge => "ge",
            CmpOp::Lt => "lt",
            CmpOp::Gt => "gt",
        }
    }

    pub fn op_str(&self) -> &'static str {
        match self {
            CmpOp::Eq => "==",
            CmpOp::Ne => "!=",
            CmpOp::Le => "<=",
            CmpOp::Ge => ">=",
            CmpOp::Lt => "<",
            CmpOp::Gt => ">",
        }
    }
}

#[derive(new, Debug, Clone, IrFields)]
pub struct Cmp {
    pub lhs: OpTensor,
    pub rhs: TensorTypeOrScalarEnum<OpTensor>,
    pub op: CmpOp,
}

impl KernelRenderable for CmpKernels {
    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        _: bool,
    ) -> Result<(), OperationError> {
        builder.register_storage("A", BindingMode::ReadOnly, Array::<P>::default());
        let CmpKernels::Standard(inner) = self;
        if let TensorTypeOrScalarEnum::Tensor(_) = &inner.rhs {
            builder.register_storage("B", BindingMode::ReadOnly, Array::<P>::default());
        }
        let CmpKernels::Standard(inner) = self;
        match self.kernel_element(&inner.lhs) {
            KernelElement::Scalar => {
                builder.register_storage(
                    "Y",
                    BindingMode::ReadWrite,
                    Array::<Scalar<i32>>::default(),
                );
            }
            KernelElement::Vec2 => {
                builder.register_storage(
                    "Y",
                    BindingMode::ReadWrite,
                    Array::<Vec2<i32>>::default(),
                );
            }
            KernelElement::Vec4 => {
                builder.register_storage(
                    "Y",
                    BindingMode::ReadWrite,
                    Array::<Vec4<i32>>::default(),
                );
            }
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

        let CmpKernels::Standard(inner) = self;

        let N = (P::W as u32).render();
        let out_dtype = match self.kernel_element(dst) {
            KernelElement::Scalar => "i32",
            KernelElement::Vec2 => "vec2<i32>",
            KernelElement::Vec4 => "vec4<i32>",
        };
        let op = inner.op.op_str();

        let assignment_expr = match &inner.rhs {
            TensorTypeOrScalarEnum::Tensor(_) => wgsl! {
                'out_dtype(A[index] 'op B[index])
            },
            TensorTypeOrScalarEnum::Scalar(_) => {
                let casted_scalar_dtype = match self.kernel_element(dst) {
                    KernelElement::Scalar => inner.lhs.dtype().as_wgsl().to_string(),
                    KernelElement::Vec2 => {
                        format!("vec2<{}>", inner.lhs.dtype().as_wgsl())
                    }
                    KernelElement::Vec4 => {
                        format!("vec4<{}>", inner.lhs.dtype().as_wgsl())
                    }
                };
                wgsl! {
                    'out_dtype(A[index] 'op 'casted_scalar_dtype(metadata.value))
                }
            }
        };

        kernel_builder.write_main(wgsl! {
            let x_offset = workgroup_id.x * 64u;
            let index = (workgroup_id.y * num_workgroups.x * 64u) + x_offset + local_invocation_index;
            if (index >= metadata.numel / 'N) {
                return;
            }

            Y[index] = 'assignment_expr;
        });

        Ok(kernel_builder.build()?)
    }
}

impl Cmp {
    pub fn op(&self) -> &CmpOp {
        &self.op
    }
}

impl OpGuards for Cmp {
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

impl Operation for Cmp {
    fn name(&self) -> &'static str {
        match self.op {
            CmpOp::Eq => "Eq",
            CmpOp::Ne => "Ne",
            CmpOp::Le => "Le",
            CmpOp::Ge => "Ge",
            CmpOp::Lt => "Lt",
            CmpOp::Gt => "Gt",
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
            Ok(StorageView::new(broadcasted, crate::DType::I32, ostride))
        } else {
            let shape = self.lhs.shape().clone();
            let stride = Stride::from(&shape);
            Ok(StorageView::new(shape, crate::DType::I32, stride))
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
}

pub enum CmpKernels {
    Standard(Cmp),
}

impl GPUOperation for Cmp {
    type KernelEnum = CmpKernels;

    fn select_kernel(&self) -> Self::KernelEnum {
        CmpKernels::Standard(self.clone())
    }
}

impl Kernel for CmpKernels {
    type Metadata = DynKernelMetadata;

    fn kernel_name(&self) -> String {
        match self {
            CmpKernels::Standard(k) => k.op.kernel_name().to_string(),
        }
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

    fn storage_bind_group_layout(
        &self,
        inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        if inplace {
            panic!("Cmp cannot be done in place");
        }
        let CmpKernels::Standard(inner) = self;
        if let TensorTypeOrScalarEnum::Tensor(_) = &inner.rhs {
            Ok(BindGroupLayoutDescriptor::binary())
        } else {
            Ok(BindGroupLayoutDescriptor::unary())
        }
    }

    fn metadata(
        &self,
        dst: &OpTensor,
        _: &KernelElement,
    ) -> Result<Self::Metadata, OperationError> {
        let CmpKernels::Standard(inner) = self;
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

    fn build_kernel(
        &self,
        inplace: bool,
        dst: &OpTensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let CmpKernels::Standard(inner) = self;
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
                inner.lhs.dtype(),
                kernel_element
            ))),
        }
    }
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use crate::{CmpOp, DType, Device, DeviceRequest, Shape, Tensor, randn, test_util::run_py_prg};
    use proptest::arbitrary::any;
    use test_strategy::{Arbitrary, proptest};

    #[derive(Arbitrary, Debug)]
    struct BinaryProblem {
        op: CmpOp,
        #[any(vec![1..=4, 1..=4, 1..=1, 1..=256])]
        shape: Shape,
    }

    #[derive(Arbitrary, Debug)]
    struct CmpScalarProblem {
        op: CmpOp,
        #[any(vec![1..=4, 1..=4, 1..=1, 1..=256])]
        shape: Shape,
        #[strategy(any::<f32>())]
        scalar: f32,
    }

    fn ground_truth(a: &Tensor, b: &Tensor, op: &CmpOp) -> anyhow::Result<Tensor> {
        let kn = op.kernel_name();
        let prg = format!(
            r#"
import torch
import numpy as np
def {kn}(a, b):
    return torch.{kn}(torch.from_numpy(a), torch.from_numpy(b)).numpy().astype(np.int32)
"#,
        );
        run_py_prg(prg.to_string(), &[a, b], &[], DType::I32)
    }

    fn ground_truth_scalar(a: &Tensor, scalar: f32, op: &CmpOp) -> anyhow::Result<Tensor> {
        let kn = op.kernel_name();
        let prg = format!(
            r#"
import torch
import numpy as np
def {kn}(a, scalar):
    return torch.{kn}(torch.from_numpy(a), scalar).numpy().astype(np.int32)
"#,
        );
        run_py_prg(prg.to_string(), &[a], &[&scalar], DType::I32)
    }

    fn run_cmp_trial(prob: BinaryProblem, device: Device) -> anyhow::Result<()> {
        let BinaryProblem { op, shape } = prob;
        let a = randn(shape.clone(), None, None, Default::default())?;
        let b = randn(shape, None, None, Default::default())?;
        let ground = ground_truth(&a, &b, &op)?.cast(DType::F32)?;

        let a_gpu = a.to(&device)?;
        let b_gpu = b.to(&device)?;
        let c_gpu = match op {
            CmpOp::Eq => a_gpu.eq(b_gpu)?,
            CmpOp::Ne => a_gpu.ne(b_gpu)?,
            CmpOp::Le => a_gpu.le(b_gpu)?,
            CmpOp::Ge => a_gpu.ge(b_gpu)?,
            CmpOp::Lt => a_gpu.le(b_gpu)?,
            CmpOp::Gt => a_gpu.gt(b_gpu)?,
        };

        let d_gpu = c_gpu.to(&Device::CPU)?.cast(DType::F32)?;
        ground.all_close(&d_gpu, 1e-4, 1e-4)?;
        Ok(())
    }

    fn run_cmp_scalar_trial(prob: CmpScalarProblem, device: Device) -> anyhow::Result<()> {
        let CmpScalarProblem { op, shape, scalar } = prob;
        let a = randn(shape, None, None, Default::default())?;
        let ground = ground_truth_scalar(&a, scalar, &op)?.cast(DType::F32)?;

        let a_gpu = a.to(&device)?;
        let c_gpu = match op {
            CmpOp::Eq => a_gpu.eq(scalar)?,
            CmpOp::Ne => a_gpu.ne(scalar)?,
            CmpOp::Le => a_gpu.le(scalar)?,
            CmpOp::Ge => a_gpu.ge(scalar)?,
            CmpOp::Lt => a_gpu.lt(scalar)?,
            CmpOp::Gt => a_gpu.gt(scalar)?,
        };

        let d_gpu = c_gpu.to(&Device::CPU)?.cast(DType::F32)?;
        ground.all_close(&d_gpu, 1e-4, 1e-4)?;
        Ok(())
    }

    #[proptest(cases = 16)]
    fn test_binary(prob: BinaryProblem) {
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        run_cmp_trial(prob, device).unwrap();
    }

    #[proptest(cases = 16)]
    fn test_scalar(prob: CmpScalarProblem) {
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        run_cmp_scalar_trial(prob, device).unwrap();
    }
}
