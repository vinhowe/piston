use derive_new::new;
use half::f16;
use inline_wgsl::wgsl;
use piston_macros::IrFields;

use crate::{
    Array, BindingMode, BuiltIn, DType, DynKernelMetadata, GPUOperation, InvariantError, Kernel,
    KernelElement, KernelRenderable, KernelSource, OpGuards, OpTensor, Operation, OperationError,
    RVec, Scalar, Shape, StorageView, Stride, TensorTypeOrScalar, TensorTypeOrScalarEnum, Vec2,
    Vec4, WgslKernelBuilder, WgslPrimitive, WorkgroupSize, Workload,
    gpu::{BindGroupLayoutDescriptor, dtype::WgslDType},
    rvec,
};

#[derive(new, Debug, Clone, IrFields)]
pub struct WhereCond {
    pub condition: OpTensor,
    pub on_true: TensorTypeOrScalarEnum<OpTensor>,
    pub on_false: TensorTypeOrScalarEnum<OpTensor>,
}

impl WhereCond {
    pub fn dtype(&self) -> Result<DType, OperationError> {
        let tensor_dtype = self
            .on_true
            .map_tensor(|t| t.dtype())
            .or_else(|_| self.on_false.map_tensor(|t| t.dtype()))?;
        match tensor_dtype {
            TensorTypeOrScalarEnum::Tensor(t) => Ok(t),
            TensorTypeOrScalarEnum::Scalar(_) => Ok(DType::F32),
        }
    }
}

impl OpGuards for WhereCond {
    fn check_shapes(&self) {
        let (a, b, c) = (&self.condition, &self.on_true, &self.on_false);
        if let TensorTypeOrScalarEnum::Tensor(b) = b {
            assert_eq!(a.shape(), b.shape());
        }
        if let TensorTypeOrScalarEnum::Tensor(c) = c {
            assert_eq!(a.shape(), c.shape());
        }
    }

    fn check_dtypes(&self) {
        let (a, b, c) = (&self.condition, &self.on_true, &self.on_false);
        assert!(matches!(a.dtype(), crate::DType::F32 | crate::DType::I32));
        if let TensorTypeOrScalarEnum::Tensor(b) = b {
            assert!(matches!(b.dtype(), crate::DType::F32 | crate::DType::I32));
        }
        if let TensorTypeOrScalarEnum::Tensor(c) = c {
            assert!(matches!(c.dtype(), crate::DType::F32 | crate::DType::I32));
        }
        if let TensorTypeOrScalarEnum::Tensor(b) = b
            && let TensorTypeOrScalarEnum::Tensor(c) = c
        {
            assert!(b.dtype() == c.dtype())
        }
    }
}

impl Operation for WhereCond {
    fn name(&self) -> &'static str {
        "WhereCond"
    }

    fn compute_view(&self) -> Result<StorageView, OperationError> {
        let on_true_shape = self
            .on_true
            .tensor()
            .map(|t| t.shape().clone())
            .unwrap_or(Shape::scalar());
        let on_false_shape = self
            .on_false
            .tensor()
            .map(|t| t.shape().clone())
            .unwrap_or(Shape::scalar());
        let shapes = &[self.condition.shape(), &on_true_shape, &on_false_shape];
        let broadcasted = Shape::multi_broadcast(shapes);
        if broadcasted.is_none() {
            let failed = shapes.iter().map(|s| (*s).clone()).collect::<Vec<_>>();
            return Err(InvariantError::BroadcastingFailed(failed).into());
        }
        let broadcasted = broadcasted.unwrap();
        let ostride = Stride::from(&broadcasted);
        Ok(StorageView::new(broadcasted, self.dtype()?, ostride))
    }

    #[inline]
    fn srcs(&self) -> RVec<&OpTensor> {
        let mut srcs = rvec![&self.condition];
        if let TensorTypeOrScalarEnum::Tensor(on_true) = &self.on_true {
            srcs.push(on_true);
        }
        if let TensorTypeOrScalarEnum::Tensor(on_false) = &self.on_false {
            srcs.push(on_false);
        }
        srcs
    }

    fn supports_inplace(&self) -> bool {
        // For inplace, the on_{true,false} tensors must be the same dtype as the condition tensor
        self.on_true
            .tensor()
            .is_none_or(|t| t.dtype() == self.condition.dtype())
            && self
                .on_false
                .tensor()
                .is_none_or(|t| t.dtype() == self.condition.dtype())
    }
}

pub enum WhereCondKernels {
    Standard(WhereCond),
}

impl GPUOperation for WhereCond {
    type KernelEnum = WhereCondKernels;

    fn select_kernel(&self) -> Self::KernelEnum {
        WhereCondKernels::Standard(self.clone())
    }
}

impl Kernel for WhereCondKernels {
    type Metadata = DynKernelMetadata;

    fn kernel_name(&self) -> String {
        match self {
            WhereCondKernels::Standard(_) => "where_cond".to_string(),
        }
    }

    fn kernel_element(&self, _dst: &OpTensor) -> KernelElement {
        let WhereCondKernels::Standard(inner) = self;
        let a_rank = inner.condition.shape().dim();
        let N = if a_rank > 0 {
            inner.condition.shape()[a_rank - 1]
        } else {
            1
        };

        if N.is_multiple_of(4) {
            KernelElement::Vec4
        } else if N.is_multiple_of(2) {
            KernelElement::Vec2
        } else {
            KernelElement::Scalar
        }
    }

    fn calculate_dispatch(&self, dst: &OpTensor) -> Result<Workload, OperationError> {
        let WhereCondKernels::Standard(inner) = self;
        Ok(Workload::std(
            inner.condition.shape().numel(),
            self.kernel_element(dst),
        ))
    }

    fn storage_bind_group_layout(
        &self,
        inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        let WhereCondKernels::Standard(inner) = self;
        let tensor_count = 1 + // condition is always a tensor
            if matches!(inner.on_true, TensorTypeOrScalarEnum::Tensor(_)) { 1 } else { 0 } +
            if matches!(inner.on_false, TensorTypeOrScalarEnum::Tensor(_)) { 1 } else { 0 };

        match tensor_count {
            1 => {
                // Only condition is a tensor, both on_true and on_false are scalars
                if inplace {
                    Ok(BindGroupLayoutDescriptor::unary_inplace())
                } else {
                    Ok(BindGroupLayoutDescriptor::unary())
                }
            }
            2 => {
                // Condition + one of on_true/on_false is a tensor
                if inplace {
                    Ok(BindGroupLayoutDescriptor::binary_inplace())
                } else {
                    Ok(BindGroupLayoutDescriptor::binary())
                }
            }
            3 => {
                // All three are tensors
                if inplace {
                    Ok(BindGroupLayoutDescriptor::ternary_inplace())
                } else {
                    Ok(BindGroupLayoutDescriptor::ternary())
                }
            }
            _ => unreachable!("Invalid tensor count for WhereCond"),
        }
    }

    fn metadata(
        &self,
        dst: &OpTensor,
        _: &KernelElement,
    ) -> Result<Self::Metadata, OperationError> {
        let WhereCondKernels::Standard(inner) = self;
        let mut dyn_meta = DynKernelMetadata::new();
        dyn_meta.add_field("numel", dst.shape().numel() as u32);

        // Add scalar values to metadata
        if let TensorTypeOrScalarEnum::Scalar(value) = &inner.on_true {
            if dst.dtype().is_float() {
                dyn_meta.add_field("on_true_value", *value);
            } else {
                dyn_meta.add_field("on_true_value", *value as i32);
            }
        }

        if let TensorTypeOrScalarEnum::Scalar(value) = &inner.on_false {
            if dst.dtype().is_float() {
                dyn_meta.add_field("on_false_value", *value);
            } else {
                dyn_meta.add_field("on_false_value", *value as i32);
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
        let kernel_element = self.kernel_element(dst);
        let WhereCondKernels::Standard(inner) = self;
        let dtype = inner.dtype()?;
        match (dtype, &kernel_element) {
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
                "Unsupported dtype {dtype:?} or kernel element {kernel_element:?}"
            ))),
        }
    }
}

impl KernelRenderable for WhereCondKernels {
    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        inplace: bool,
    ) -> Result<(), OperationError> {
        let WhereCondKernels::Standard(inner) = self;
        let arr = Array::<P>::default();
        builder.register_storage(
            "A",
            if inplace {
                BindingMode::ReadWrite
            } else {
                BindingMode::ReadOnly
            },
            arr,
        );

        // Only register storage for tensor inputs, not scalars
        if matches!(inner.on_true, TensorTypeOrScalarEnum::Tensor(_)) {
            builder.register_storage("B", BindingMode::ReadOnly, Array::<P>::default());
        }
        if matches!(inner.on_false, TensorTypeOrScalarEnum::Tensor(_)) {
            builder.register_storage("C", BindingMode::ReadOnly, Array::<P>::default());
        }

        if !inplace {
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
                BuiltIn::NumWorkgroups,
                BuiltIn::LocalInvocationIndex
            ],
            device.compute_features().clone(),
        );

        let kernel_element = self.kernel_element(dst);

        self.register_bindings::<P>(&mut kernel_builder, inplace)?;
        kernel_builder.render_metadata(&self.metadata(dst, &kernel_element)?);

        let N = (P::W as u32).render();

        kernel_builder.write_main(wgsl! {
            let x_offset = workgroup_id.x * 64u;
            let index = (workgroup_id.y * num_workgroups.x * 64u) + x_offset + local_invocation_index;
            if (index >= metadata.numel / 'N) {
                return;
            }
        });

        let dtype = P::T::DT;

        let kernel_element_str = match kernel_element {
            KernelElement::Scalar => dtype.to_string(),
            KernelElement::Vec2 => format!("{}<{}>", kernel_element.as_str(), dtype),
            KernelElement::Vec4 => format!("{}<{}>", kernel_element.as_str(), dtype),
        };

        let WhereCondKernels::Standard(inner) = self;

        // Determine how to access on_true and on_false values
        let on_true_expr = match &inner.on_true {
            TensorTypeOrScalarEnum::Tensor(_) => "B[index]".to_string(),
            TensorTypeOrScalarEnum::Scalar(_) => {
                let casted_scalar_dtype = match kernel_element {
                    KernelElement::Scalar => dtype.to_string(),
                    KernelElement::Vec2 => format!("{}<{}>", kernel_element.as_str(), dtype),
                    KernelElement::Vec4 => format!("{}<{}>", kernel_element.as_str(), dtype),
                };
                format!("{casted_scalar_dtype}(metadata.on_true_value)")
            }
        };

        let on_false_expr = match &inner.on_false {
            TensorTypeOrScalarEnum::Tensor(_) => "C[index]".to_string(),
            TensorTypeOrScalarEnum::Scalar(_) => {
                let casted_scalar_dtype = match kernel_element {
                    KernelElement::Scalar => dtype.to_string(),
                    KernelElement::Vec2 => format!("{}<{}>", kernel_element.as_str(), dtype),
                    KernelElement::Vec4 => format!("{}<{}>", kernel_element.as_str(), dtype),
                };
                format!("{casted_scalar_dtype}(metadata.on_false_value)")
            }
        };

        let apply = if inplace {
            wgsl! {
                let val = A[index];
                A[index] = select('on_true_expr, 'on_false_expr, val != 'kernel_element_str(0));
            }
        } else {
            wgsl! { Y[index] = select('on_true_expr, 'on_false_expr, A[index] != 'kernel_element_str(0)); }
        };
        kernel_builder.write_main(apply);
        Ok(kernel_builder.build()?)
    }
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use proptest::arbitrary::any;
    use test_strategy::{Arbitrary, proptest};

    use crate::test_util::run_py_prg;
    use crate::{Device, DeviceRequest, Tensor, randn};

    fn ground_truth(a: &Tensor, b: &Tensor, c: &Tensor) -> anyhow::Result<Tensor> {
        let prg = r#"
import torch
def where_cond(a, b, c):
    return torch.where(torch.from_numpy(a) == 0, torch.from_numpy(b), torch.from_numpy(c)).numpy()
"#;
        run_py_prg(prg.to_string(), &[a, b, c], &[], b.dtype())
    }

    fn ground_truth_scalar(a: &Tensor, b: &Tensor, scalar: f32) -> anyhow::Result<Tensor> {
        let prg = r#"
import torch
def where_cond_scalar(a, b, scalar):
    return torch.where(torch.from_numpy(a) == 0, torch.from_numpy(b), scalar).numpy()
"#;
        run_py_prg(prg.to_string(), &[a, b], &[&scalar], b.dtype())
    }

    fn run_where_cond_trial(problem: WhereCondProblem, device: Device) {
        let WhereCondProblem { B, M, N } = problem;
        // Put through a ReLU so some of its entries are 0
        let a = randn((B, M, N), None, None, Default::default())
            .unwrap()
            .relu()
            .unwrap();
        let b = randn((B, M, N), None, None, Default::default()).unwrap();
        let c = randn((B, M, N), None, None, Default::default()).unwrap();
        let ground = ground_truth(&b, &a, &c).unwrap();

        let a_gpu = a.to(&device).unwrap();
        let b_gpu = b.to(&device).unwrap();
        let c_gpu = c.to(&device).unwrap();
        let b = a_gpu.where_cond(b_gpu, c_gpu).unwrap();

        let ours = b.to(&Device::CPU).unwrap();

        log::debug!("ours = {ours:?}");
        log::debug!("ground = {ground:?}");

        ground.all_close(&ours, 1e-6, 1e-6).unwrap();
    }

    fn run_where_cond_scalar_trial(problem: WhereCondScalarProblem, device: Device) {
        let WhereCondScalarProblem { B, M, N, scalar } = problem;
        // Put through a ReLU so some of its entries are 0
        let a = randn((B, M, N), None, None, Default::default())
            .unwrap()
            .relu()
            .unwrap();
        let b = randn((B, M, N), None, None, Default::default()).unwrap();
        let ground = ground_truth_scalar(&b, &a, scalar).unwrap();

        let a_gpu = a.to(&device).unwrap();
        let b_gpu = b.to(&device).unwrap();
        let result = a_gpu.where_cond(b_gpu, scalar).unwrap();

        let ours = result.to(&Device::CPU).unwrap();

        log::debug!("ours = {ours:?}");
        log::debug!("ground = {ground:?}");

        ground.all_close(&ours, 1e-6, 1e-6).unwrap();
    }

    #[derive(Arbitrary, Debug)]
    struct WhereCondProblem {
        #[strategy(1..=3usize)]
        B: usize,
        #[strategy(1..=256usize)]
        M: usize,
        #[strategy(1..=256usize)]
        N: usize,
    }

    #[derive(Arbitrary, Debug)]
    struct WhereCondScalarProblem {
        #[strategy(1..=3usize)]
        B: usize,
        #[strategy(1..=256usize)]
        M: usize,
        #[strategy(1..=256usize)]
        N: usize,
        #[strategy(any::<f32>())]
        scalar: f32,
    }

    #[proptest(cases = 8)]
    fn test_where_cond(prob: WhereCondProblem) {
        let _ = env_logger::builder().try_init();
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        run_where_cond_trial(prob, device);
    }

    #[proptest(cases = 8)]
    fn test_where_cond_scalar(prob: WhereCondScalarProblem) {
        let _ = env_logger::builder().try_init();
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        run_where_cond_scalar_trial(prob, device);
    }
}
