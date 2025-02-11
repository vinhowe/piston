use derive_new::new;
use encase::ShaderType;
use half::f16;
use inline_wgsl::wgsl;
use ratchet_macros::WgslMetadata;

use crate::{
    gpu::{dtype::WgslDType, BindGroupLayoutDescriptor},
    rvec, Array, BindingMode, BuiltIn, DType, GPUOperation, Kernel, KernelElement,
    KernelRenderable, KernelSource, OpGuards, Operation, OperationError, RVec, Scalar, StorageView,
    Tensor, Vec2, Vec4, WgslKernelBuilder, WgslPrimitive, WorkgroupSize, Workload,
};

#[derive(new, Debug, Clone)]
pub struct Affine {
    pub src: Tensor,
    pub mul: f32,
    pub add: f32,
}

#[derive(Debug, derive_new::new, ShaderType, WgslMetadata)]
pub struct AffineMeta {
    numel: u32,
    mul: f32,
    add: f32,
}

impl OpGuards for Affine {
    fn check_shapes(&self) {}

    fn check_dtypes(&self) {
        let a = &self.src;
        assert!(matches!(a.dt(), crate::DType::F32));
    }
}

impl Operation for Affine {
    fn name(&self) -> &'static str {
        "Affine"
    }

    fn compute_view(&self) -> Result<StorageView, OperationError> {
        Ok(self.src.storage_view().clone())
    }

    #[inline]
    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.src]
    }

    fn supports_inplace(&self) -> bool {
        true
    }
}

pub enum AffineKernels {
    Standard(Affine),
}

impl GPUOperation for Affine {
    type KernelEnum = AffineKernels;

    fn select_kernel(&self) -> Self::KernelEnum {
        AffineKernels::Standard(self.clone())
    }
}

impl KernelRenderable for AffineKernels {
    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        inplace: bool,
    ) -> Result<(), OperationError> {
        let arr = Array::<P>::default();
        if inplace {
            builder.register_storage("X", BindingMode::ReadWrite, arr);
        } else {
            builder.register_storage("X", BindingMode::ReadOnly, arr);
            builder.register_storage("Y", BindingMode::ReadWrite, arr);
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

        let N = (P::W as u32).render();
        let dt = P::render_type();

        kernel_builder.write_main(wgsl! {
            let x_offset = workgroup_id.x * 64u;
            let index = (workgroup_id.y * num_workgroups.x * 64u) + x_offset + local_invocation_index;
            if (index >= metadata.numel / 'N) {
                return;
            }
        });

        let apply = if inplace {
            wgsl! {
                let val = X[index];
                X[index] = fma(val, 'dt(metadata.mul), 'dt(metadata.add));
            }
        } else {
            wgsl! { Y[index] = fma(X[index], 'dt(metadata.mul), 'dt(metadata.add)); }
        };
        kernel_builder.write_main(apply);
        Ok(kernel_builder.build()?)
    }
}

impl Kernel for AffineKernels {
    type Metadata = AffineMeta;

    fn kernel_name(&self) -> String {
        match self {
            AffineKernels::Standard(_) => "affine".to_string(),
        }
    }

    fn calculate_dispatch(&self, dst: &Tensor) -> Result<Workload, OperationError> {
        let AffineKernels::Standard(inner) = self;
        Ok(Workload::std(
            inner.src.shape().numel(),
            self.kernel_element(dst),
        ))
    }

    fn storage_bind_group_layout(
        &self,
        inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        if inplace {
            Ok(BindGroupLayoutDescriptor::unary_inplace())
        } else {
            Ok(BindGroupLayoutDescriptor::unary())
        }
    }

    fn metadata(&self, _: &Tensor, _: &KernelElement) -> Result<Self::Metadata, OperationError> {
        let AffineKernels::Standard(inner) = self;
        Ok(AffineMeta {
            numel: inner.src.shape().numel() as u32,
            mul: inner.mul,
            add: inner.add,
        })
    }

    fn kernel_element(&self, _dst: &Tensor) -> KernelElement {
        let AffineKernels::Standard(inner) = self;
        let a_rank = inner.src.shape().rank();
        let N = if a_rank > 0 {
            inner.src.shape()[a_rank - 1]
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

    fn build_kernel(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let kernel_element = self.kernel_element(dst);
        let AffineKernels::Standard(inner) = self;
        match (inner.src.dt(), &kernel_element) {
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
                inner.src.dt(),
                kernel_element
            ))),
        }
    }
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use proptest::arbitrary::any;
    use test_strategy::{proptest, Arbitrary};

    use crate::{shape, test_util::run_py_prg, Device, DeviceRequest, Tensor};

    fn ground_truth(a: &Tensor, mul: f32, add: f32) -> anyhow::Result<Tensor> {
        let prg = r#"
import torch
def affine(a, mul, add):
    return (torch.from_numpy(a) * mul + add).numpy()
"#
        .to_string();

        run_py_prg(prg.to_string(), &[a], &[&mul, &add], a.dt())
    }

    fn run_affine_trial(problem: AffineProblem, device: Device) {
        let AffineProblem { B, M, N, add, mul } = problem;
        let a = Tensor::randn::<f32>(0., 1., shape![B, M, N], Device::CPU);
        let ground = ground_truth(&a, mul, add).unwrap();

        let a_gpu = a.to(&device).unwrap();
        let b = a_gpu.affine(mul, add).unwrap().resolve().unwrap();

        let ours = b.to(&Device::CPU).unwrap();
        println!("ours = {:?}", ours);
        println!("ground = {:?}", ground);
        ground.all_close(&ours, 1e-4, 1e-4).unwrap();
    }

    #[derive(Arbitrary, Debug)]
    struct AffineProblem {
        #[strategy(1..=128usize)]
        B: usize,
        #[strategy(1..=128usize)]
        M: usize,
        #[strategy(1..=128usize)]
        N: usize,
        #[strategy(any::<f32>())]
        mul: f32,
        #[strategy(any::<f32>())]
        add: f32,
    }

    #[proptest(cases = 8)]
    fn test_affine(prob: AffineProblem) {
        let AffineProblem { B, M, N, mul, add } = prob;
        println!(
            "B = {}, M = {}, N = {}, mul = {}, add = {}",
            B, M, N, mul, add
        );
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        run_affine_trial(prob, device);
    }
}
