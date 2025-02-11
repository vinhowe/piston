use derive_new::new;
use encase::ShaderType;
use half::f16;
use inline_wgsl::wgsl;
use ratchet_macros::{IrFields, WgslMetadata};

use crate::{
    gpu::{dtype::WgslDType, BindGroupLayoutDescriptor},
    rvec, Array, BindingMode, BuiltIn, DType, GPUOperation, Kernel, KernelElement,
    KernelRenderable, KernelSource, OpGuards, Operation, OperationError, RVec, Scalar, StorageView,
    Tensor, Vec2, Vec4, WgslKernelBuilder, WgslPrimitive, WorkgroupSize, Workload,
};

#[derive(new, Debug, Clone, IrFields)]
pub struct WhereCond {
    pub input: Tensor,
    pub on_true: Tensor,
    pub on_false: Tensor,
}

#[derive(Debug, ShaderType, WgslMetadata)]
pub struct WhereCondMeta {
    numel: u32,
}

impl OpGuards for WhereCond {
    fn check_shapes(&self) {
        let (a, b, c) = (&self.input, &self.on_true, &self.on_false);
        assert_eq!(a.shape(), b.shape());
        assert_eq!(a.shape(), c.shape());
    }

    fn check_dtypes(&self) {
        let (a, b, c) = (&self.input, &self.on_true, &self.on_false);
        assert!(matches!(a.dt(), crate::DType::F32 | crate::DType::I32));
        assert!(matches!(b.dt(), crate::DType::F32 | crate::DType::I32));
        assert!(b.dt() == c.dt())
    }
}

impl Operation for WhereCond {
    fn name(&self) -> &'static str {
        "WhereCond"
    }

    fn compute_view(&self) -> Result<StorageView, OperationError> {
        Ok(self.on_true.storage_view().clone())
    }

    #[inline]
    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.input, &self.on_true, &self.on_false]
    }

    fn supports_inplace(&self) -> bool {
        // For inplace, the on_{true,false} tensors must be the same dtype as the input tensor
        self.on_true.dt() == self.input.dt() && self.on_false.dt() == self.input.dt()
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
    type Metadata = WhereCondMeta;

    fn kernel_name(&self) -> String {
        match self {
            WhereCondKernels::Standard(_) => "where_cond".to_string(),
        }
    }

    fn kernel_element(&self, _dst: &Tensor) -> KernelElement {
        let WhereCondKernels::Standard(inner) = self;
        let a_rank = inner.input.shape().rank();
        let N = if a_rank > 0 {
            inner.input.shape()[a_rank - 1]
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

    fn calculate_dispatch(&self, dst: &Tensor) -> Result<Workload, OperationError> {
        let WhereCondKernels::Standard(inner) = self;
        Ok(Workload::std(
            inner.input.shape().numel(),
            self.kernel_element(dst),
        ))
    }

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

    fn metadata(&self, _: &Tensor, _: &KernelElement) -> Result<Self::Metadata, OperationError> {
        let WhereCondKernels::Standard(inner) = self;
        let numel = inner.input.shape().numel() as u32;
        Ok(WhereCondMeta { numel })
    }

    fn build_kernel(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let kernel_element = self.kernel_element(dst);
        let WhereCondKernels::Standard(inner) = self;
        match (inner.input.dt(), &kernel_element) {
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
                inner.input.dt(),
                kernel_element
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

        builder.register_storage("B", BindingMode::ReadOnly, Array::<P>::default());
        builder.register_storage("C", BindingMode::ReadOnly, Array::<P>::default());

        if !inplace {
            builder.register_storage("Y", BindingMode::ReadWrite, Array::<P>::default());
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

        let dt = P::T::DT;

        let kernel_element_str = match kernel_element {
            KernelElement::Scalar => dt.to_string(),
            KernelElement::Vec2 => format!("{}<{}>", kernel_element.as_str(), dt),
            KernelElement::Vec4 => format!("{}<{}>", kernel_element.as_str(), dt),
        };

        let apply = if inplace {
            wgsl! {
                let val = A[index];
                A[index] = select(B[index], C[index], val != 'kernel_element_str(0));
            }
        } else {
            wgsl! { Y[index] = select(B[index], C[index], A[index] != 'kernel_element_str(0)); }
        };
        kernel_builder.write_main(apply);
        Ok(kernel_builder.build()?)
    }
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use test_strategy::{proptest, Arbitrary};

    use crate::test_util::run_py_prg;
    use crate::{shape, Device, DeviceRequest, Tensor};

    fn ground_truth(a: &Tensor, b: &Tensor, c: &Tensor) -> anyhow::Result<Tensor> {
        let prg = r#"
import torch
def where_cond(a, b, c):
    return torch.where(torch.from_numpy(a) == 0, torch.from_numpy(b), torch.from_numpy(c)).numpy()
"#;
        run_py_prg(prg.to_string(), &[a, b, c], &[], b.dt())
    }

    fn run_where_cond_trial(problem: WhereCondProblem, device: Device) {
        let WhereCondProblem { B, M, N } = problem;
        // Put through a ReLU so some of its entries are 0
        let a = Tensor::randn::<f32>(0., 1., shape![B, M, N], Device::CPU)
            .relu()
            .unwrap()
            .resolve()
            .unwrap();
        let b = Tensor::randn::<f32>(0., 1., shape![B, M, N], Device::CPU);
        let c = Tensor::randn::<f32>(0., 1., shape![B, M, N], Device::CPU);
        let ground = ground_truth(&a, &b, &c).unwrap();

        let a_gpu = a.to(&device).unwrap();
        let b_gpu = b.to(&device).unwrap();
        let c_gpu = c.to(&device).unwrap();
        let b = a_gpu.where_cond(b_gpu, c_gpu).unwrap().resolve().unwrap();

        let ours = b.to(&Device::CPU).unwrap();

        log::debug!("ours = {:?}", ours);
        log::debug!("ground = {:?}", ground);

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

    #[proptest(cases = 8)]
    fn test_where_cond(prob: WhereCondProblem) {
        let _ = env_logger::builder().try_init();
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        run_where_cond_trial(prob, device);
    }
}
