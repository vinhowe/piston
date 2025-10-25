use derive_new::new;
use half::f16;
use inline_wgsl::wgsl;
use piston_macros::IrFields;

use crate::{
    Array, BindingMode, BuiltIn, DType, DynKernelMetadata, GPUOperation, Kernel, KernelElement,
    KernelRenderable, KernelSource, OpGuards, OpTensor, Operation, OperationError, RVec, Scalar,
    Shape, StorageView, Stride, WgslKernelBuilder, WgslPrimitive, WorkgroupSize, Workload,
    gpu::{BindGroupLayoutDescriptor, dtype::WgslDType},
    rvec,
};

#[derive(new, Debug, Clone, IrFields)]
pub struct Eye {
    pub shape: Shape,
}

impl Operation for Eye {
    fn name(&self) -> &'static str {
        "Eye"
    }

    fn compute_view(&self) -> Result<StorageView, OperationError> {
        let shape: Shape = self.shape.clone();
        let stride = Stride::from(&shape);
        Ok(StorageView::new(shape, DType::F32, stride))
    }

    fn srcs(&self) -> RVec<&OpTensor> {
        rvec![]
    }

    fn supports_inplace(&self) -> bool {
        false
    }
}

impl OpGuards for Eye {
    fn check_shapes(&self) {
        assert!(
            self.shape.dim() == 2,
            "Eye expects a 2D shape, got {:?}",
            self.shape
        );
    }

    fn check_dtypes(&self) {}
}

pub enum EyeKernels {
    Standard(Eye),
}

impl GPUOperation for Eye {
    type KernelEnum = EyeKernels;

    fn select_kernel(&self) -> Self::KernelEnum {
        EyeKernels::Standard(self.clone())
    }
}

impl KernelRenderable for EyeKernels {
    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        _: bool,
    ) -> Result<(), OperationError> {
        builder.register_storage("Y", BindingMode::ReadWrite, Array::<P>::default());
        builder.register_uniform();
        Ok(())
    }

    fn render<P: WgslPrimitive>(
        &self,
        _: bool,
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

        self.register_bindings::<P>(&mut kernel_builder, false)?;
        kernel_builder.render_metadata(&self.metadata(dst, &self.kernel_element(dst))?);

        let N = (P::W as u32).render();
        let dtype = P::render_type();

        kernel_builder.write_main(wgsl! {
            let x_offset = workgroup_id.x * 64u;
            let index = (workgroup_id.y * num_workgroups.x * 64u) + x_offset + local_invocation_index;
            if (index >= metadata.numel / 'N) {
                return;
            }

            // Scalar-only kernel element, so 'N == 1
            let cols = metadata.cols;
            let row = index / cols;
            let col = index - row * cols;
            if (row == col) {
                Y[index] = 'dtype(1);
            } else {
                Y[index] = 'dtype(0);
            }
        });

        Ok(kernel_builder.build()?)
    }
}

impl Kernel for EyeKernels {
    type Metadata = DynKernelMetadata;

    fn kernel_name(&self) -> String {
        match self {
            EyeKernels::Standard(_) => "eye".to_string(),
        }
    }

    fn kernel_element(&self, _dst: &OpTensor) -> KernelElement {
        // Keep scalar for correctness and simplicity.
        KernelElement::Scalar
    }

    fn calculate_dispatch(&self, dst: &OpTensor) -> Result<Workload, OperationError> {
        Ok(Workload::std(dst.shape().numel(), self.kernel_element(dst)))
    }

    fn storage_bind_group_layout(
        &self,
        _inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        Ok(BindGroupLayoutDescriptor::unary_inplace())
    }

    fn metadata(
        &self,
        dst: &OpTensor,
        _: &KernelElement,
    ) -> Result<Self::Metadata, OperationError> {
        let EyeKernels::Standard(op) = self;
        let mut dyn_meta = DynKernelMetadata::new();
        dyn_meta.add_field("numel", dst.shape().numel() as u32);
        let rows = op.shape[0] as u32;
        let cols = op.shape[1] as u32;
        dyn_meta.add_field("rows", rows);
        dyn_meta.add_field("cols", cols);
        Ok(dyn_meta)
    }

    fn build_kernel(
        &self,
        inplace: bool,
        dst: &OpTensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let kernel_element = self.kernel_element(dst);
        match (dst.dtype(), &kernel_element) {
            (DType::F32, KernelElement::Scalar) => {
                self.render::<Scalar<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Scalar) => {
                self.render::<Scalar<f16>>(inplace, dst, workgroup_size)
            }
            (DType::I32, KernelElement::Scalar) => {
                self.render::<Scalar<i32>>(inplace, dst, workgroup_size)
            }
            (DType::U32, KernelElement::Scalar) => {
                self.render::<Scalar<u32>>(inplace, dst, workgroup_size)
            }
            _ => Err(OperationError::CompileError(format!(
                "Unsupported dtype {:?} or kernel element {:?}",
                dst.dtype(),
                kernel_element
            ))),
        }
    }
}
