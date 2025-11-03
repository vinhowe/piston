use derive_new::new;
use inline_wgsl::wgsl;

use crate::{
    Array, BindingMode, BuiltIn, DType, DynKernelMetadata, GPUOperation, Ir, IrFields, Kernel,
    KernelElement, KernelRenderable, KernelSource, OpGuards, OpTensor, Operation, OperationError,
    RVec, Scalar, Shape, StorageView, Stride, Vec2, Vec4, WgslKernelBuilder, WgslPrimitive,
    WorkgroupSize, Workload,
    gpu::{BindGroupLayoutDescriptor, dtype::WgslDType},
    rvec,
};

#[derive(Debug, Clone)]
pub enum FillPointwiseKind {
    Constant {
        value: f32,
    },
    Randn {
        mean: f32,
        std: f32,
        seed: Option<u32>,
    },
    Rand {
        lo: f32,
        up: f32,
        seed: Option<u32>,
    },
}

#[derive(new, Debug, Clone)]
pub struct FillPointwise {
    pub shape: Shape,
    pub kind: FillPointwiseKind,
}

impl FillPointwise {
    pub fn kernel_name(&self) -> &'static str {
        match self.kind {
            FillPointwiseKind::Constant { .. } => "fill_constant",
            FillPointwiseKind::Randn { .. } => "fill_randn",
            FillPointwiseKind::Rand { .. } => "fill_rand",
        }
    }
}

impl Operation for FillPointwise {
    fn name(&self) -> &'static str {
        "FillPointwise"
    }

    fn compute_view(&self) -> Result<StorageView, OperationError> {
        // The dtype is determined by the OpTensor meta at construction time.
        // Returning F32 here mirrors existing fill ops and is unused in GPU path.
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

impl OpGuards for FillPointwise {
    fn check_shapes(&self) {}
    fn check_dtypes(&self) {}
}

pub enum FillPointwiseKernels {
    Standard(FillPointwise),
}

impl GPUOperation for FillPointwise {
    type KernelEnum = FillPointwiseKernels;

    fn select_kernel(&self) -> Self::KernelEnum {
        FillPointwiseKernels::Standard(self.clone())
    }
}

impl KernelRenderable for FillPointwiseKernels {
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

        // Random helpers used by Randn/Xavier variants
        let needs_rand = match self {
            FillPointwiseKernels::Standard(inner) => matches!(
                inner.kind,
                FillPointwiseKind::Randn { .. } | FillPointwiseKind::Rand { .. }
            ),
        };

        if needs_rand {
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
        }

        // Common indexing prelude. Use N for vector width (1 for scalar types).
        kernel_builder.write_main(wgsl! {
            let x_offset = workgroup_id.x * 64u;
            let index = (workgroup_id.y * num_workgroups.x * 64u) + x_offset + local_invocation_index;
            if (index >= metadata.numel / 'N) {
                return;
            }
        });

        match self {
            FillPointwiseKernels::Standard(inner) => match inner.kind {
                FillPointwiseKind::Constant { .. } => {
                    kernel_builder.write_main(wgsl! {
                        Y[index] = 'dtype(metadata.value);
                    });
                }
                FillPointwiseKind::Randn { .. } => {
                    kernel_builder.write_main(wgsl! {
                        let seed1 = index;
                        let seed2 = index ^ 2747636419u;
                        let u1 = rand(seed1 + metadata.seed);
                        let u2 = rand(seed2 + metadata.seed);
                        let normal = box_muller_1d(u1, u2);
                        Y[index] = 'dtype(f32(normal) * metadata.stddev + metadata.mean);
                    });
                }
                FillPointwiseKind::Rand { .. } => {
                    kernel_builder.write_main(wgsl! {
                        let u = rand(index + metadata.seed);
                        Y[index] = 'dtype(metadata.lo + (metadata.hi - metadata.lo) * u);
                    });
                }
            },
        }

        Ok(kernel_builder.build()?)
    }
}

impl Kernel for FillPointwiseKernels {
    type Metadata = DynKernelMetadata;

    fn kernel_name(&self) -> String {
        match self {
            FillPointwiseKernels::Standard(inner) => inner.kernel_name().to_string(),
        }
    }

    fn kernel_element(&self, dst: &OpTensor) -> KernelElement {
        match self {
            FillPointwiseKernels::Standard(inner) => match inner.kind {
                FillPointwiseKind::Constant { .. } => {
                    // Vectorize constant fills like the original FillConstant
                    let rank = dst.shape().dim();
                    let N = if rank > 0 { dst.shape()[rank - 1] } else { 1 };
                    if N.is_multiple_of(4) {
                        KernelElement::Vec4
                    } else if N.is_multiple_of(2) {
                        KernelElement::Vec2
                    } else {
                        KernelElement::Scalar
                    }
                }
                // Randomized variants: keep scalar for simplicity/correctness
                _ => KernelElement::Scalar,
            },
        }
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
        let mut dyn_meta = DynKernelMetadata::new();
        dyn_meta.add_field("numel", dst.shape().numel() as u32);

        match self {
            FillPointwiseKernels::Standard(inner) => match inner.kind {
                FillPointwiseKind::Constant { value } => {
                    if dst.dtype().is_float() {
                        dyn_meta.add_field("value", value);
                    } else {
                        dyn_meta.add_field("value", value as i32);
                    }
                }
                FillPointwiseKind::Randn { mean, std, seed } => {
                    dyn_meta.add_field("mean", mean);
                    dyn_meta.add_field("stddev", std);
                    dyn_meta.add_field("seed", seed.unwrap_or(0));
                }
                FillPointwiseKind::Rand { lo, up, seed } => {
                    dyn_meta.add_field("lo", lo);
                    dyn_meta.add_field("hi", up);
                    dyn_meta.add_field("seed", seed.unwrap_or(0));
                }
            },
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
        match (dst.dtype(), &kernel_element) {
            // Floating types: all variants supported (random & constant). Random variants are Scalar-only by design.
            (DType::F32, KernelElement::Scalar) => {
                self.render::<Scalar<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Scalar) => {
                self.render::<Scalar<half::f16>>(inplace, dst, workgroup_size)
            }

            // Vectorized constant fills for performance
            (DType::F32, KernelElement::Vec2) => {
                self.render::<Vec2<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F32, KernelElement::Vec4) => {
                self.render::<Vec4<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Vec2) => {
                self.render::<Vec2<half::f16>>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Vec4) => {
                self.render::<Vec4<half::f16>>(inplace, dst, workgroup_size)
            }

            // Integer dtype only meaningful for constant fills (random/Xavier require floats). KernelElement for random is Scalar.
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
                dst.dtype(),
                kernel_element
            ))),
        }
    }
}

impl IrFields for FillPointwise {
    fn ir_fields(&self, ir: &mut Ir) {
        ir.with_field("shape", self.shape.clone());
        match &self.kind {
            FillPointwiseKind::Constant { value } => {
                ir.with_field("kind", "Constant");
                ir.with_field("value", *value);
            }
            FillPointwiseKind::Randn { mean, std, seed } => {
                ir.with_field("kind", "Randn");
                ir.with_field("mean", *mean);
                ir.with_field("stddev", *std);
                ir.with_field("seed", seed.unwrap_or(0));
            }
            FillPointwiseKind::Rand { lo, up, seed } => {
                ir.with_field("kind", "Rand");
                ir.with_field("lo", *lo);
                ir.with_field("hi", *up);
                ir.with_field("seed", seed.unwrap_or(0));
            }
        }
    }
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use test_strategy::{Arbitrary, proptest};

    use crate::{
        DType, Device, DeviceRequest, Tensor, TensorOptions, randn, test_util::run_py_prg,
    };

    #[derive(Arbitrary, Debug)]
    struct FillPointwiseProblem {
        #[strategy(1..=64usize)]
        B: usize,
        #[strategy(1..=128usize)]
        M: usize,
        #[strategy(1..=128usize)]
        N: usize,
    }

    fn normal_parameters(output: &Tensor) -> anyhow::Result<(f32, f32)> {
        let prg = r#"
import numpy as np

def check_normal(output):
    output_np = np.array(output)
    mean = float(np.mean(output_np))
    std = float(np.std(output_np))
    return np.array([mean, std], dtype=np.float32)
"#;

        let params = run_py_prg(prg.to_string(), &[output], &[], DType::F32)?
            .to(&Device::CPU)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        Ok((params[0], params[1]))
    }

    #[proptest(cases = 16)]
    fn test_fill_pointwise_constant(prob: FillPointwiseProblem) {
        let FillPointwiseProblem { B, M, N } = prob;
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        let value: f32 = 0.5;

        let ours = Tensor::full(
            (B, M, N),
            value,
            TensorOptions::new().device(device.clone()),
        )
        .unwrap()
        .to(&Device::CPU)
        .unwrap();

        // Ground truth via torch.full
        let prg = r#"
import torch
def fill_constant(shape, value):
    return torch.full(shape, value, dtype=torch.float32).cpu().numpy()
"#;
        let ground = run_py_prg(
            prg.to_string(),
            &[],
            &[&vec![B as i64, M as i64, N as i64], &value],
            DType::F32,
        )
        .unwrap();

        ground.all_close(&ours, 1e-6, 1e-6).unwrap();
    }

    #[proptest(cases = 16)]
    fn test_fill_pointwise_randn(prob: FillPointwiseProblem) {
        let FillPointwiseProblem { B, M, N } = prob;
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        let a = randn((B, M, N), None, None, TensorOptions::new().device(device))
            .unwrap()
            .to(&Device::CPU)
            .unwrap();

        let (mean, std) = normal_parameters(&a).unwrap();
        assert!((mean - 0.0).abs() < 0.1, "mean={mean}");
        assert!((std - 1.0).abs() < 0.1, "std={std}");
    }
}
