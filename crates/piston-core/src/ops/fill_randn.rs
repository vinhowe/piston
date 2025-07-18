use derive_new::new;
use encase::ShaderType;
use half::f16;
use inline_wgsl::wgsl;
use piston_macros::{IrFields, WgslMetadata};

use crate::{
    gpu::BindGroupLayoutDescriptor, rvec, Array, BindingMode, BuiltIn, DType, GPUOperation, Kernel,
    KernelElement, KernelRenderable, KernelSource, OpGuards, OpTensor, Operation, OperationError,
    RVec, Scalar, Shape, StorageView, Stride, Vec2, Vec4, WgslKernelBuilder, WgslPrimitive,
    WorkgroupSize, Workload,
};

#[derive(new, Debug, Clone, IrFields)]
pub struct FillRandn {
    pub shape: Shape,
    pub mean: f32,
    pub std: f32,
    pub seed: Option<u32>,
}

#[derive(Debug, derive_new::new, ShaderType, WgslMetadata)]
pub struct FillRandnMeta {
    numel: u32,
    mean: f32,
    stddev: f32,
    seed: u32,
}

impl Operation for FillRandn {
    fn name(&self) -> &'static str {
        "FillRandn"
    }

    fn compute_view(&self) -> Result<StorageView, OperationError> {
        let shape: Shape = self.shape.clone();
        let stride = Stride::from(&shape);
        Ok(StorageView::new(shape, crate::DType::F32, stride))
    }

    fn srcs(&self) -> RVec<&OpTensor> {
        rvec![]
    }

    fn supports_inplace(&self) -> bool {
        false
    }
}

impl OpGuards for FillRandn {
    fn check_shapes(&self) {
        // No input shapes to check
    }

    fn check_dtypes(&self) {
        // No input dtypes to check
    }
}

pub enum FillRandnKernels {
    Standard(FillRandn),
}

impl GPUOperation for FillRandn {
    type KernelEnum = FillRandnKernels;

    fn select_kernel(&self) -> Self::KernelEnum {
        FillRandnKernels::Standard(self.clone())
    }
}

impl KernelRenderable for FillRandnKernels {
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

        kernel_builder.write_global(wgsl! {
            fn pcg_hash(input: u32) -> u32 {
                let state = input * 747796405u + 2891336453u;
                let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
                return (word >> 22u) ^ word;
            }

            fn rand(seed: u32) -> f32 {
                return f32(pcg_hash(seed)) / 4294967295.0;
            }

            fn box_muller_1d(u1: f32, u2: f32) -> f32 {
                let r = sqrt(-2.0 * log(u1));
                let theta = 2.0 * 3.14159265359 * u2;
                return r * cos(theta);
            }
        });

        kernel_builder.write_main(wgsl! {
            let x_offset = workgroup_id.x * 64u;
            let index = (workgroup_id.y * num_workgroups.x * 64u) + x_offset + local_invocation_index;
            if (index >= metadata.numel) {
                return;
            }

            let seed1 = index;
            let seed2 = index ^ 2747636419u; // XOR with a prime for a different seed

            let u1 = rand(seed1 + metadata.seed);
            let u2 = rand(seed2 + metadata.seed);

            let normal = box_muller_1d(u1, u2);

            Y[index] = f32(normal) * metadata.stddev + metadata.mean;
        });

        Ok(kernel_builder.build()?)
    }
}

impl Kernel for FillRandnKernels {
    type Metadata = FillRandnMeta;

    fn kernel_name(&self) -> String {
        match self {
            FillRandnKernels::Standard(_) => "fill_randn".to_string(),
        }
    }

    fn kernel_element(&self, _dst: &OpTensor) -> KernelElement {
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

    fn metadata(&self, _: &OpTensor, _: &KernelElement) -> Result<Self::Metadata, OperationError> {
        let FillRandnKernels::Standard(inner) = self;
        Ok(FillRandnMeta {
            numel: inner.shape.clone().numel() as u32,
            mean: inner.mean,
            stddev: inner.std,
            seed: inner.seed.unwrap_or(0),
        })
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
                dst.dtype(),
                kernel_element
            ))),
        }
    }
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use test_strategy::{proptest, Arbitrary};

    use crate::{test_util::run_py_prg, DType, Device, DeviceRequest, Tensor};

    fn normal_parameters(output: &Tensor) -> anyhow::Result<Tensor> {
        let prg = r#"
import numpy as np

def check_normal(output):
    output_np = np.array(output)
    mean = np.mean(output_np)
    std = np.std(output_np)
    return np.array([mean, std], dtype=np.float32)
"#;

        run_py_prg(prg.to_string(), &[output], &[], DType::F32)
    }

    fn run_fill_randn_trial(problem: FillRandnProblem, device: Device) {
        let FillRandnProblem { B, M, N } = problem;

        let a = Tensor::randn::<f32, _>(0f32, 1f32, (B, M, N), device.clone(), false)
            .unwrap()
            .to(&Device::CPU)
            .unwrap();

        let params = normal_parameters(&a)
            .unwrap()
            .to(&device)
            .unwrap()
            .to(&Device::CPU)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();

        let mean = params[0];
        let std = params[1];

        // Check if the distribution is approximately normal
        // We use a tolerance of 0.1 for both mean and standard deviation
        if (mean - 0.0).abs() < 0.1 && (std - 1.0).abs() < 0.1 {
            println!("\x1b[1;32mDistribution approximately normal\x1b[0m - mean={mean} std={std}");
        } else {
            (|| -> anyhow::Result<()> {
                {
                    anyhow::bail!(
                        "\x1b[1;31mDistribution not normal\x1b[0m - mean={} std={}",
                        mean,
                        std
                    )
                }
            })()
            .unwrap();
        }
    }

    #[derive(Arbitrary, Debug)]
    struct FillRandnProblem {
        #[strategy(1..=128usize)]
        B: usize,
        #[strategy(1..=128usize)]
        M: usize,
        #[strategy(1..=128usize)]
        N: usize,
    }

    #[proptest(cases = 8)]
    fn test_fill_randn(prob: FillRandnProblem) {
        let FillRandnProblem { B, M, N } = prob;
        println!("B = {B}, M = {M}, N = {N}");
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        run_fill_randn_trial(prob, device);
    }
}
