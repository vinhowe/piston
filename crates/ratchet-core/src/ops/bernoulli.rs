use derive_new::new;
use encase::ShaderType;
use half::f16;
use inline_wgsl::wgsl;
use ratchet_macros::{IrFields, WgslMetadata};

use crate::{
    gpu::BindGroupLayoutDescriptor, rvec, Array, BindingMode, BuiltIn, DType, GPUOperation, Kernel,
    KernelElement, KernelRenderable, KernelSource, OpGuards, Operation, OperationError, RVec,
    Scalar, StorageView, Strides, Tensor, Vec2, Vec4, WgslKernelBuilder, WgslPrimitive,
    WorkgroupSize, Workload,
};

#[derive(new, Debug, Clone, IrFields)]
pub struct Bernoulli {
    pub probs: Tensor,
    pub seed: Option<u32>,
}

#[derive(Debug, derive_new::new, ShaderType, WgslMetadata)]
pub struct BernoulliMeta {
    numel: u32,
    seed: u32,
}

impl Operation for Bernoulli {
    fn name(&self) -> &'static str {
        "Bernoulli"
    }

    fn compute_view(&self) -> Result<StorageView, OperationError> {
        // Output has same shape as the input probabilities tensor
        let shape = self.probs.shape().clone();
        let strides = Strides::from(&shape);
        Ok(StorageView::new(shape, DType::F32, strides))
    }

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.probs]
    }

    fn supports_inplace(&self) -> bool {
        false
    }
}

impl OpGuards for Bernoulli {
    fn check_shapes(&self) {
        // No specific shape constraints for probabilities tensor
    }

    fn check_dtypes(&self) {
        // Ensure probabilities tensor has floating-point dtype
        assert!(
            self.probs.dtype().is_float(),
            "Probabilities tensor must have floating-point dtype"
        );
    }
}

pub enum BernoulliKernels {
    Standard(Bernoulli),
}

impl GPUOperation for Bernoulli {
    type KernelEnum = BernoulliKernels;

    fn select_kernel(&self) -> Self::KernelEnum {
        BernoulliKernels::Standard(self.clone())
    }
}

impl KernelRenderable for BernoulliKernels {
    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        _: bool,
    ) -> Result<(), OperationError> {
        builder.register_storage("X", BindingMode::ReadOnly, Array::<P>::default()); // Input probabilities
        builder.register_storage("Y", BindingMode::ReadWrite, Array::<P>::default()); // Output binary values
        builder.register_uniform();
        Ok(())
    }

    fn render<P: WgslPrimitive>(
        &self,
        _: bool,
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
        });

        kernel_builder.write_main(wgsl! {
            let x_offset = workgroup_id.x * 64u;
            let index = (workgroup_id.y * num_workgroups.x * 64u) + x_offset + local_invocation_index;
            if (index >= metadata.numel) {
                return;
            }

            let prob = X[index];
            let seed = index + metadata.seed;
            let random_value = rand(seed);

            // If random value is less than probability, set to 1.0, otherwise 0.0
            Y[index] = f32(random_value < prob);
        });

        Ok(kernel_builder.build()?)
    }
}

impl Kernel for BernoulliKernels {
    type Metadata = BernoulliMeta;

    fn kernel_name(&self) -> String {
        match self {
            BernoulliKernels::Standard(_) => "bernoulli".to_string(),
        }
    }

    fn kernel_element(&self, _dst: &Tensor) -> KernelElement {
        KernelElement::Scalar
    }

    fn calculate_dispatch(&self, dst: &Tensor) -> Result<Workload, OperationError> {
        Ok(Workload::std(dst.shape().numel(), self.kernel_element(dst)))
    }

    fn storage_bind_group_layout(
        &self,
        _inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        Ok(BindGroupLayoutDescriptor::unary())
    }

    fn metadata(&self, dst: &Tensor, _: &KernelElement) -> Result<Self::Metadata, OperationError> {
        let BernoulliKernels::Standard(inner) = self;
        Ok(BernoulliMeta {
            numel: dst.shape().numel() as u32,
            seed: inner.seed.unwrap_or(0),
        })
    }

    fn build_kernel(
        &self,
        inplace: bool,
        dst: &Tensor,
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

#[cfg(all(test, feature = "pyo3", feature = "rand"))]
mod tests {
    use test_strategy::{proptest, Arbitrary};

    use crate::{shape, Device, DeviceRequest, Tensor};

    fn run_bernoulli_trial(problem: BernoulliProblem, device: Device) {
        let BernoulliProblem { B, M, N, seed } = problem;

        device.set_seed(seed as u64);

        // Create a tensor with random probabilities between 0 and 1
        let probs = Tensor::rand::<f32>(0f32, 1f32, shape![B, M, N], Device::CPU);
        let probs_gpu = probs.to(&device).unwrap();

        // Apply Bernoulli sampling to the probabilities tensor
        let a = probs_gpu.bernoulli().unwrap();

        // Check that all values are either 0 or 1
        let values = a.to(&Device::CPU).unwrap().to_vec::<f32>().unwrap();
        for val in values.iter() {
            assert!(
                *val == 0.0 || *val == 1.0,
                "Expected binary values (0 or 1)"
            );
        }

        // Get statistics to verify that sampling distribution is reasonable
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let expected_mean = probs
            .to(&Device::CPU)
            .unwrap()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .sum::<f32>()
            / values.len() as f32;

        // Calculate observed std of the binary outcomes.
        let std =
            (values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32).sqrt();
        // For a uniformly distributed probabilities tensor, the overall variance (by total variance law) is expected to be ~0.25.
        // So we square root it to get the standard deviation, 0.5.
        let expected_std = 0.5;

        if (mean - expected_mean).abs() < 0.1 && (std - expected_std).abs() < 0.1 {
            println!(
                "\x1b[1;32mDistribution approximately bernoulli\x1b[0m - mean={} std={}",
                mean, std
            );
        } else {
            (|| -> anyhow::Result<()> {
                anyhow::bail!(
                    "\x1b[1;31mDistribution not bernoulli\x1b[0m - mean={} std={}",
                    mean,
                    std
                )
            })()
            .unwrap();
        }
    }

    #[derive(Arbitrary, Debug)]
    struct BernoulliProblem {
        #[strategy(1..=64usize)]
        B: usize,
        #[strategy(1..=64usize)]
        M: usize,
        #[strategy(1..=64usize)]
        N: usize,
        #[strategy(0..=1000u32)]
        seed: u32,
    }

    #[proptest(cases = 8)]
    fn test_bernoulli(prob: BernoulliProblem) {
        let BernoulliProblem { B, M, N, seed } = prob;
        println!("B = {}, M = {}, N = {}, seed = {}", B, M, N, seed);
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        run_bernoulli_trial(prob, device);
    }
}
