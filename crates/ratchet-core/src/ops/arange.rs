use derive_new::new;
use inline_wgsl::wgsl;
use ratchet_macros::IrFields;

use crate::{
    gpu::BindGroupLayoutDescriptor, rvec, shape, wgc, wgs, Array, BindingMode, BuiltIn, DType,
    DynKernelMetadata, GPUOperation, Kernel, KernelElement, KernelRenderable, KernelSource,
    OpGuards, Operation, OperationError, RVec, Scalar, StorageView, Strides, Tensor,
    WgslKernelBuilder, WgslPrimitive, WorkgroupCount, WorkgroupSize, Workload,
};

#[derive(new, Debug, Clone, IrFields)]
pub struct Arange {
    pub start: f32,
    pub end: f32,
    pub step: f32,
}

/// The main trait that describes an operation on the GPU. We implement
/// how to compute its output shape and type (`compute_view`), among other
/// things.
impl Operation for Arange {
    fn name(&self) -> &'static str {
        "Arange"
    }

    /// Determine the shape of the output tensor (1D of length `numel`).
    fn compute_view(&self) -> Result<StorageView, OperationError> {
        let numel = self.numel();
        let shape = shape![numel];
        let strides = Strides::from(&shape);
        Ok(StorageView::new(shape, DType::F32, strides))
    }

    /// The arange op has no input tensors, so return an empty array.
    fn srcs(&self) -> RVec<&Tensor> {
        rvec![]
    }

    fn supports_inplace(&self) -> bool {
        // We are creating a new buffer from scratch, so no in-place support needed.
        false
    }
}

impl Arange {
    pub(crate) fn numel(&self) -> usize {
        if self.step == 0.0 {
            return 0;
        }

        let diff = self.end - self.start;

        // If step > 0 and stop <= start, then empty. If step < 0 and stop >= start, also empty.
        if (self.step > 0.0 && self.end <= self.start)
            || (self.step < 0.0 && self.end >= self.start)
        {
            return 0;
        }

        let n = diff / self.step;
        (n.ceil() as isize).max(0) as usize
    }
}

/// For safety checks (e.g., verifying shapes/dtypes of inputs, if any).
impl OpGuards for Arange {
    fn check_shapes(&self) {
        // No input shapes to check
    }

    fn check_dtypes(&self) {
        // No input dtypes to check
    }
}

/// An enum of possible "kernels" for the Arange op. Here we only have one variant.
pub enum ArangeKernels {
    Standard(Arange),
}

/// We implement GPUOperation, which selects a kernel variant.
impl GPUOperation for Arange {
    type KernelEnum = ArangeKernels;

    fn select_kernel(&self) -> Self::KernelEnum {
        ArangeKernels::Standard(self.clone())
    }
}

/// The kernel‐rendering trait describes how to build and dispatch the WGSL code.
impl KernelRenderable for ArangeKernels {
    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        _inplace: bool,
    ) -> Result<(), OperationError> {
        // We'll store our output array and a uniform buffer with metadata.
        builder.register_storage("Y", BindingMode::ReadWrite, Array::<P>::default());
        builder.register_uniform();
        Ok(())
    }

    fn render<P: WgslPrimitive>(
        &self,
        _inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let device = dst.device().try_gpu()?;

        // Build the WGSL code similarly to the FillRandn example
        let mut kernel_builder = WgslKernelBuilder::new(
            workgroup_size.clone(),
            rvec![BuiltIn::GlobalInvocationId],
            device.compute_features().clone(),
        );

        self.register_bindings::<P>(&mut kernel_builder, false)?;

        kernel_builder.render_metadata(&self.metadata(dst, &self.kernel_element(dst))?);

        let dt = P::render_type();

        // Minimal compute shader: for each index, compute start + step * idx.
        // We skip any index >= metadata.numel.
        kernel_builder.write_main(wgsl! {
            let idx = global_invocation_id.x;
            if (idx >= metadata.numel) {
                return;
            }

            Y[idx] = metadata.start + 'dt(idx) * metadata.step;
        });

        Ok(kernel_builder.build()?)
    }
}

/// We implement the `Kernel` trait for our ArangeKernels to specify how to compute the final
/// WGSL code, dispatch sizes, etc.
impl Kernel for ArangeKernels {
    type Metadata = DynKernelMetadata;

    fn kernel_name(&self) -> String {
        match self {
            ArangeKernels::Standard(_) => "arange".to_string(),
        }
    }

    fn kernel_element(&self, _dst: &Tensor) -> KernelElement {
        KernelElement::Scalar
    }

    fn calculate_dispatch(&self, dst: &Tensor) -> Result<Workload, OperationError> {
        let workgroup_size = wgs![256, 1, 1];
        let numel = dst.shape().numel();

        // Calculate number of workgroups needed to cover all elements
        let x_groups = WorkgroupCount::div_ceil(numel as _, workgroup_size.product() as _);
        let (x_groups, y_groups) = if x_groups > WorkgroupCount::MAX_WGS_PER_DIM {
            let y_groups = WorkgroupCount::div_ceil(x_groups, WorkgroupCount::MAX_WGS_PER_DIM);
            (WorkgroupCount::MAX_WGS_PER_DIM, y_groups)
        } else {
            (x_groups, 1)
        };

        Ok(Workload {
            workgroup_count: wgc![x_groups as _, y_groups as _, 1],
            workgroup_size,
        })
    }

    fn storage_bind_group_layout(
        &self,
        _inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        Ok(BindGroupLayoutDescriptor::unary_inplace())
    }

    fn metadata(&self, dst: &Tensor, _: &KernelElement) -> Result<Self::Metadata, OperationError> {
        let ArangeKernels::Standard(op) = self;
        let mut dyn_meta = DynKernelMetadata::new();
        if dst.dt().is_float() {
            dyn_meta.add_field("start", op.start);
            dyn_meta.add_field("step", op.step);
        } else {
            dyn_meta.add_field("start", op.start as i32);
            dyn_meta.add_field("step", op.step as i32);
        }
        dyn_meta.add_field("numel", dst.shape().numel() as u32);
        Ok(dyn_meta)
    }

    fn build_kernel(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let kernel_element = self.kernel_element(dst);
        // Our arange op is f32 only, but you could add more branches for f16, etc.
        match (dst.dt(), &kernel_element) {
            (DType::F32, KernelElement::Scalar) => {
                self.render::<Scalar<f32>>(inplace, dst, workgroup_size)
            }
            (DType::I32, KernelElement::Scalar) => {
                self.render::<Scalar<i32>>(inplace, dst, workgroup_size)
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
    use std::ops::Neg;

    use num_traits::AsPrimitive;
    use pyo3::ToPyObject;
    use test_strategy::{proptest, Arbitrary};

    use crate::{test_util::run_py_prg, DType, Device, DeviceRequest, Tensor, TensorDType};

    fn ground_truth(
        start: &dyn ToPyObject,
        stop: &dyn ToPyObject,
        step: &dyn ToPyObject,
    ) -> anyhow::Result<Tensor> {
        let prg = r#"
import torch
def arange(start, stop, step):
    return torch.arange(start, stop, step, dtype=torch.float32).cpu().numpy()
"#;

        run_py_prg(prg.to_string(), &[], &[start, stop, step], DType::F32)
    }

    fn run_arange_trial<
        T: TensorDType + PartialOrd + Neg<Output = T> + AsPrimitive<f32> + ToPyObject,
    >(
        start: T,
        stop: T,
        step: T,
        device: &Device,
    ) {
        fn abs<T: TensorDType + PartialOrd + Neg<Output = T>>(x: T) -> T {
            if x >= T::zero() {
                x
            } else {
                -x
            }
        }

        // Determine correct sign for step based on start/stop relationship
        let step = if stop >= start { abs(step) } else { -abs(step) };

        let a = Tensor::arange_step(start, stop, step, device)
            .unwrap()
            .cast(DType::F32)
            .unwrap()
            .resolve_deferred()
            .unwrap();
        let ground = ground_truth(&start, &stop, &step).unwrap();

        let a_gpu = a.to(device).unwrap();
        let ours = a_gpu.to(&Device::CPU).unwrap();

        println!("ours = {:?}", ours);
        println!("ground = {:?}", ground);

        // Compare our result with ground truth
        ground.all_close(&ours, 1e-6, 1e-6).unwrap();
    }

    #[derive(Arbitrary, Debug)]
    struct ArangeProblemF32 {
        #[strategy(-100.0..=100.0)]
        start: f64,
        #[strategy(-100.0..=100.0)]
        stop: f64,
        #[strategy(0.1..=10.0)]
        step: f64,
    }

    #[proptest(cases = 8)]
    fn test_arange_f32(prob: ArangeProblemF32) {
        let ArangeProblemF32 { start, stop, step } = prob;
        println!("start = {}, stop = {}, step = {}", start, stop, step);
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        run_arange_trial::<f32>(start as f32, stop as f32, step as f32, &device);
    }

    #[derive(Arbitrary, Debug)]
    struct ArangeProblemI32 {
        #[strategy(-100..=100)]
        start: i32,
        #[strategy(-100..=100)]
        stop: i32,
        #[strategy(1..=10)]
        step: i32,
    }

    #[proptest(cases = 8)]
    fn test_arange_i32(prob: ArangeProblemI32) {
        let ArangeProblemI32 { start, stop, step } = prob;
        println!("start = {}, stop = {}, step = {}", start, stop, step);
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        run_arange_trial::<i32>(start, stop, step, &device);
    }
}
