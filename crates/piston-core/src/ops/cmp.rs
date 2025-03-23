use derive_new::new;
use encase::ShaderType;
use half::f16;
use inline_wgsl::wgsl;
use piston_macros::{IrFields, WgslMetadata};

use crate::{
    gpu::{dtype::WgslDType, BindGroupLayoutDescriptor},
    rvec, Array, BindingMode, BuiltIn, DType, GPUOperation, InvariantError, Kernel, KernelElement,
    KernelRenderable, KernelSource, OpGuards, Operation, OperationError, RVec, Scalar, Shape,
    StorageView, Stride, Tensor, Vec2, Vec4, WgslKernelBuilder, WgslPrimitive, WorkgroupSize,
    Workload,
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
    pub lhs: Tensor,
    pub rhs: Tensor,
    pub op: CmpOp,
}

impl KernelRenderable for CmpKernels {
    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        _: bool,
    ) -> Result<(), OperationError> {
        builder.register_storage("A", BindingMode::ReadOnly, Array::<P>::default());
        builder.register_storage("B", BindingMode::ReadOnly, Array::<P>::default());
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

        let CmpKernels::Standard(inner) = self;

        let N = (P::W as u32).render();
        let dtype = match self.kernel_element(dst) {
            KernelElement::Scalar => "i32",
            KernelElement::Vec2 => "vec2<i32>",
            KernelElement::Vec4 => "vec4<i32>",
        };
        let op = inner.op.op_str();

        kernel_builder.write_main(wgsl! {
            let x_offset = workgroup_id.x * 64u;
            let index = (workgroup_id.y * num_workgroups.x * 64u) + x_offset + local_invocation_index;
            if (index >= metadata.numel / 'N) {
                return;
            }

            Y[index] = 'dtype(A[index] 'op B[index]);
        });

        Ok(kernel_builder.build()?)
    }
}

impl Cmp {
    pub fn op(&self) -> &CmpOp {
        &self.op
    }
}

#[derive(Debug, ShaderType, WgslMetadata)]
pub struct CmpMeta {
    numel: u32,
}

impl OpGuards for Cmp {
    fn check_shapes(&self) {
        let shapes = [self.lhs.shape(), self.rhs.shape()];
        let broadcasted = Shape::multi_broadcast(&shapes);
        assert!(broadcasted.is_some());
    }

    fn check_dtypes(&self) {
        assert_eq!(self.lhs.dtype(), self.rhs.dtype());
    }
}

impl Operation for Cmp {
    fn name(&self) -> &'static str {
        "Cmp"
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
        Ok(StorageView::new(broadcasted, crate::DType::I32, ostride))
    }

    #[inline]
    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.lhs, &self.rhs]
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
    type Metadata = CmpMeta;

    fn kernel_name(&self) -> String {
        match self {
            CmpKernels::Standard(_) => "cmp".to_string(),
        }
    }

    fn kernel_element(&self, dst: &Tensor) -> KernelElement {
        let numel = dst.shape().numel();

        if numel % 4 == 0 {
            KernelElement::Vec4
        } else if numel % 2 == 0 {
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
        inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        if inplace {
            panic!("Cmp cannot be done in place");
        }
        Ok(BindGroupLayoutDescriptor::binary())
    }

    fn metadata(&self, dst: &Tensor, _: &KernelElement) -> Result<Self::Metadata, OperationError> {
        let numel = dst.shape().numel() as _;
        Ok(CmpMeta { numel })
    }

    fn build_kernel(
        &self,
        inplace: bool,
        dst: &Tensor,
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
    use crate::{test_util::run_py_prg, CmpOp, DType, Device, DeviceRequest, Shape, Tensor};
    use test_strategy::{proptest, Arbitrary};

    #[derive(Arbitrary, Debug)]
    struct BinaryProblem {
        op: CmpOp,
        #[any(vec![1..=4, 1..=4, 1..=1, 1..=256])]
        shape: Shape,
    }

    fn ground_truth(a: &Tensor, b: &Tensor, op: &CmpOp) -> anyhow::Result<Tensor> {
        let kn = op.kernel_name();
        let prg = format!(
            r#"
import torch
import numpy as np
def {}(a, b):
    return torch.{}(torch.from_numpy(a), torch.from_numpy(b)).numpy().astype(np.int32)
"#,
            kn, kn
        );
        run_py_prg(prg.to_string(), &[a, b], &[], DType::I32)
    }

    fn run_cmp_trial(prob: BinaryProblem, device: Device) -> anyhow::Result<()> {
        let cpu_device = Device::request_device(DeviceRequest::CPU)?;
        let BinaryProblem { op, shape } = prob;
        let a = Tensor::randn::<f32, _>(0., 1., shape.clone(), cpu_device.clone())?;
        let b = Tensor::randn::<f32, _>(0., 1., shape, cpu_device.clone())?;
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

    #[proptest(cases = 8)]
    fn test_binary(prob: BinaryProblem) {
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        run_cmp_trial(prob, device).unwrap();
    }
}
