mod groupnorm;

use encase::ShaderType;
pub use groupnorm::GroupNorm;
use half::f16;
use piston_macros::WgslMetadata;

use crate::{
    Array, BindingMode, BuiltIn, DType, GPUOperation, Kernel, KernelElement, KernelRenderable,
    KernelSource, OpGuards, OpTensor, Operation, OperationError, RVec, Scalar, StorageView, Vec2,
    Vec4, WgslKernelBuilder, WgslPrimitive, WorkgroupSize, Workload,
    gpu::{BindGroupLayoutDescriptor, dtype::WgslDType},
    rvec, wgc, wgs,
};
use derive_new::new;
use inline_wgsl::wgsl;
use piston_macros::IrFields;

#[derive(new, Debug, Clone, IrFields)]
pub struct Norm {
    pub(crate) input: OpTensor,
    pub(crate) scale: Option<OpTensor>,
    pub(crate) bias: Option<OpTensor>,
    pub(crate) eps: f32,
}

impl OpGuards for NormOp {
    fn check_shapes(&self) {
        let input = match self {
            NormOp::LayerNorm(Norm { input, .. }) | NormOp::RMSNorm(Norm { input, .. }) => input,
            NormOp::GroupNorm(GroupNorm { norm, .. }) => &norm.input,
        };
        assert!(input.dim() >= 2);
    }

    fn check_dtypes(&self) {
        let (input, scale, bias) = match self {
            NormOp::LayerNorm(Norm {
                input, scale, bias, ..
            }) => (input, scale, bias),
            NormOp::RMSNorm(Norm { input, scale, .. }) => (input, scale, &None),
            NormOp::GroupNorm(GroupNorm { norm, .. }) => (&norm.input, &norm.scale, &norm.bias),
        };

        input.dtype().is_float();
        if let Some(scale) = scale {
            scale.dtype().is_float();
        }
        if let Some(bias) = bias {
            bias.dtype().is_float();
        }
    }
}

impl Operation for NormOp {
    fn name(&self) -> &'static str {
        "Norm"
    }

    fn compute_view(&self) -> Result<StorageView, OperationError> {
        let input = match self {
            NormOp::LayerNorm(Norm { input, .. }) | NormOp::RMSNorm(Norm { input, .. }) => input,
            NormOp::GroupNorm(GroupNorm { norm, .. }) => &norm.input,
        };
        Ok(input.storage_view().clone())
    }

    #[inline]
    fn srcs(&self) -> RVec<&OpTensor> {
        let norm = match self {
            NormOp::LayerNorm(norm) | NormOp::RMSNorm(norm) => norm,
            NormOp::GroupNorm(GroupNorm { norm, .. }) => norm,
        };

        let mut sources = rvec![&norm.input];
        if let Some(scale) = &norm.scale {
            sources.push(scale);
        }
        if let Some(bias) = &norm.bias {
            sources.push(bias);
        }
        sources
    }
}

#[derive(Debug, Clone, IrFields)]
pub enum NormOp {
    LayerNorm(Norm),
    RMSNorm(Norm),
    GroupNorm(GroupNorm),
}

impl KernelRenderable for NormKernels {
    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        _: bool,
    ) -> Result<(), OperationError> {
        let arr = Array::<P>::default();
        builder.register_storage("X", BindingMode::ReadOnly, arr);

        let NormKernels::Standard(inner) = self;
        let norm = match inner {
            NormOp::LayerNorm(norm) | NormOp::RMSNorm(norm) => norm,
            NormOp::GroupNorm(GroupNorm { norm, .. }) => norm,
        };

        if norm.scale.is_some() {
            builder.register_storage("S", BindingMode::ReadOnly, arr);
        }
        if norm.bias.is_some() {
            builder.register_storage("B", BindingMode::ReadOnly, arr);
        }

        builder.register_storage("Y", BindingMode::ReadWrite, arr);
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
                BuiltIn::GlobalInvocationId,
                BuiltIn::LocalInvocationId,
                BuiltIn::WorkgroupId,
            ],
            device.compute_features().clone(),
        );
        self.register_bindings::<P>(&mut kernel_builder, inplace)?;
        kernel_builder.render_metadata(&self.metadata(dst, &self.kernel_element(dst))?);

        let NormKernels::Standard(inner) = self;

        let reduction_len = match P::W {
            1 => "metadata.N",
            2 => "metadata.ND2",
            4 => "metadata.ND4",
            v => panic!("Invalid reduction length: {v}"),
        };

        let dtype = P::T::DT;
        let accessor = P::render_type();
        let BLOCK_SIZE = workgroup_size.x.render();

        // Always use F32 for accumulation (numerical stability for F16 inputs)
        // smem stores per-thread partial sums, always in f32
        kernel_builder.write_global(wgsl! {
            var<workgroup> smem: array<f32, 'BLOCK_SIZE>;
            var<workgroup> sum: f32;
        });

        kernel_builder.write_global(wgsl! {
            fn block_sum(index: u32, stride: u32) {
                if index < stride {
                    smem[index] += smem[index + stride];
                }
                workgroupBarrier();
            }
        });

        kernel_builder.write_main(wgsl!{
            let anchor = (workgroup_id.y * metadata.M * 'reduction_len) + workgroup_id.x * 'reduction_len;
        });

        // threadSum is always f32 for numerical stability
        kernel_builder.write_main(wgsl! { var threadSum: f32 = 0.0; });

        // Handle vectorized vs scalar access for computing mu
        if matches!(inner, NormOp::LayerNorm(_)) {
            Self::compute_mu::<P>(
                &mut kernel_builder,
                accessor.clone(),
                reduction_len,
                workgroup_size,
            );
        }

        // Compute variance (sum of squared differences from mean)
        // For vectorized access, we need to handle each component
        let variance_loop = match P::W {
            1 => {
                let x_val = if matches!(inner, NormOp::LayerNorm(_)) {
                    wgsl! { let val = f32(X[anchor + i]) - mu; }
                } else {
                    wgsl! { let val = f32(X[anchor + i]); }
                };
                wgsl! {
                    'x_val
                    threadSum = fma(val, val, threadSum);
                }
            }
            2 => {
                let x_val = if matches!(inner, NormOp::LayerNorm(_)) {
                    wgsl! {
                        let v = X[anchor + i];
                        let val0 = f32(v.x) - mu;
                        let val1 = f32(v.y) - mu;
                    }
                } else {
                    wgsl! {
                        let v = X[anchor + i];
                        let val0 = f32(v.x);
                        let val1 = f32(v.y);
                    }
                };
                wgsl! {
                    'x_val
                    threadSum += val0 * val0 + val1 * val1;
                }
            }
            4 => {
                let x_val = if matches!(inner, NormOp::LayerNorm(_)) {
                    wgsl! {
                        let v = X[anchor + i];
                        let val0 = f32(v.x) - mu;
                        let val1 = f32(v.y) - mu;
                        let val2 = f32(v.z) - mu;
                        let val3 = f32(v.w) - mu;
                    }
                } else {
                    wgsl! {
                        let v = X[anchor + i];
                        let val0 = f32(v.x);
                        let val1 = f32(v.y);
                        let val2 = f32(v.z);
                        let val3 = f32(v.w);
                    }
                };
                wgsl! {
                    'x_val
                    threadSum += val0 * val0 + val1 * val1 + val2 * val2 + val3 * val3;
                }
            }
            _ => unreachable!(),
        };

        kernel_builder.write_main(wgsl! {
            threadSum = 0.0;
            for (var i: u32 = local_invocation_id.x; i < 'reduction_len; i += 'BLOCK_SIZE) {
                'variance_loop
            }
            workgroupBarrier();
            smem[local_invocation_id.x] = threadSum;
            workgroupBarrier();
        });

        let steps = (workgroup_size.x - 1).ilog2();
        for i in (0..=steps).rev().map(|x| 2u32.pow(x)) {
            let v = i.render();
            kernel_builder.write_main(wgsl! { block_sum(local_invocation_id.x, 'v); });
        }

        // sigma is always f32
        kernel_builder.write_main(wgsl! { let sigma = smem[0] / f32(metadata.N); });

        let norm = match inner {
            NormOp::LayerNorm(norm) | NormOp::RMSNorm(norm) => norm,
            NormOp::GroupNorm(GroupNorm { norm, .. }) => norm,
        };

        // Output in original dtype - handle vectorized output
        let output_loop = match P::W {
            1 => {
                let x_val = if matches!(inner, NormOp::LayerNorm(_)) {
                    wgsl! { let val = (f32(X[anchor + i]) - mu) * denom; }
                } else {
                    wgsl! { let val = f32(X[anchor + i]) * denom; }
                };
                let store = match (norm.scale.is_some(), norm.bias.is_some()) {
                    (true, true) => {
                        wgsl! { Y[anchor + i] = 'accessor(fma(val, f32(S[i]), f32(B[i]))); }
                    }
                    (true, false) => wgsl! { Y[anchor + i] = 'accessor(val * f32(S[i])); },
                    (false, true) => wgsl! { Y[anchor + i] = 'accessor(val + f32(B[i])); },
                    (false, false) => wgsl! { Y[anchor + i] = 'accessor(val); },
                };
                wgsl! {
                    'x_val
                    'store
                }
            }
            2 => {
                let x_val = if matches!(inner, NormOp::LayerNorm(_)) {
                    wgsl! {
                        let xv = X[anchor + i];
                        let val0 = (f32(xv.x) - mu) * denom;
                        let val1 = (f32(xv.y) - mu) * denom;
                    }
                } else {
                    wgsl! {
                        let xv = X[anchor + i];
                        let val0 = f32(xv.x) * denom;
                        let val1 = f32(xv.y) * denom;
                    }
                };
                let store = match (norm.scale.is_some(), norm.bias.is_some()) {
                    (true, true) => wgsl! {
                        let sv = S[i];
                        let bv = B[i];
                        Y[anchor + i] = 'accessor(vec2<f32>(
                            fma(val0, f32(sv.x), f32(bv.x)),
                            fma(val1, f32(sv.y), f32(bv.y))
                        ));
                    },
                    (true, false) => wgsl! {
                        let sv = S[i];
                        Y[anchor + i] = 'accessor(vec2<f32>(val0 * f32(sv.x), val1 * f32(sv.y)));
                    },
                    (false, true) => wgsl! {
                        let bv = B[i];
                        Y[anchor + i] = 'accessor(vec2<f32>(val0 + f32(bv.x), val1 + f32(bv.y)));
                    },
                    (false, false) => wgsl! {
                        Y[anchor + i] = 'accessor(vec2<f32>(val0, val1));
                    },
                };
                wgsl! {
                    'x_val
                    'store
                }
            }
            4 => {
                let x_val = if matches!(inner, NormOp::LayerNorm(_)) {
                    wgsl! {
                        let xv = X[anchor + i];
                        let val0 = (f32(xv.x) - mu) * denom;
                        let val1 = (f32(xv.y) - mu) * denom;
                        let val2 = (f32(xv.z) - mu) * denom;
                        let val3 = (f32(xv.w) - mu) * denom;
                    }
                } else {
                    wgsl! {
                        let xv = X[anchor + i];
                        let val0 = f32(xv.x) * denom;
                        let val1 = f32(xv.y) * denom;
                        let val2 = f32(xv.z) * denom;
                        let val3 = f32(xv.w) * denom;
                    }
                };
                let store = match (norm.scale.is_some(), norm.bias.is_some()) {
                    (true, true) => wgsl! {
                        let sv = S[i];
                        let bv = B[i];
                        Y[anchor + i] = 'accessor(vec4<f32>(
                            fma(val0, f32(sv.x), f32(bv.x)),
                            fma(val1, f32(sv.y), f32(bv.y)),
                            fma(val2, f32(sv.z), f32(bv.z)),
                            fma(val3, f32(sv.w), f32(bv.w))
                        ));
                    },
                    (true, false) => wgsl! {
                        let sv = S[i];
                        Y[anchor + i] = 'accessor(vec4<f32>(
                            val0 * f32(sv.x), val1 * f32(sv.y), val2 * f32(sv.z), val3 * f32(sv.w)
                        ));
                    },
                    (false, true) => wgsl! {
                        let bv = B[i];
                        Y[anchor + i] = 'accessor(vec4<f32>(
                            val0 + f32(bv.x), val1 + f32(bv.y), val2 + f32(bv.z), val3 + f32(bv.w)
                        ));
                    },
                    (false, false) => wgsl! {
                        Y[anchor + i] = 'accessor(vec4<f32>(val0, val1, val2, val3));
                    },
                };
                wgsl! {
                    'x_val
                    'store
                }
            }
            _ => unreachable!(),
        };

        kernel_builder.write_main(wgsl! {
            let denom = inverseSqrt(sigma + f32(metadata.eps));
            for(var i: u32 = local_invocation_id.x; i < 'reduction_len; i += 'BLOCK_SIZE) {
                'output_loop
            }
        });
        Ok(kernel_builder.build()?)
    }
}

impl NormKernels {
    fn compute_mu<P: WgslPrimitive>(
        kernel_builder: &mut WgslKernelBuilder,
        _accessor: String,
        reduction_len: &str,
        workgroup_size: &WorkgroupSize,
    ) {
        let BLOCK_SIZE = workgroup_size.x.render();

        // Cast input to f32 for accumulation, handling vectorized access
        let sum_loop = match P::W {
            1 => wgsl! { threadSum += f32(X[anchor + i]); },
            2 => wgsl! {
                let v = X[anchor + i];
                threadSum += f32(v.x) + f32(v.y);
            },
            4 => wgsl! {
                let v = X[anchor + i];
                threadSum += f32(v.x) + f32(v.y) + f32(v.z) + f32(v.w);
            },
            _ => unreachable!(),
        };

        kernel_builder.write_main(wgsl! {
            for (var i: u32 = local_invocation_id.x; i < 'reduction_len; i += 'BLOCK_SIZE) {
                'sum_loop
            }
            workgroupBarrier();
            smem[local_invocation_id.x] = threadSum;
            workgroupBarrier();
        });

        let steps = (workgroup_size.x - 1).ilog2();
        for i in (0..=steps).rev().map(|x| 2u32.pow(x)) {
            let v = i.render();
            kernel_builder.write_main(wgsl! { block_sum(local_invocation_id.x, 'v); });
        }

        // mu is always f32
        kernel_builder.write_main(wgsl! { let mu = smem[0] / f32(metadata.N); });
    }
}

#[derive(Debug, derive_new::new, ShaderType, WgslMetadata)]
pub struct NormMeta {
    M: u32,
    N: u32,
    ND2: u32,
    ND4: u32,
    eps: f32,
}

pub enum NormKernels {
    Standard(NormOp),
}

impl Kernel for NormKernels {
    type Metadata = NormMeta;

    fn kernel_name(&self) -> String {
        match self {
            NormKernels::Standard(n) => match n {
                NormOp::LayerNorm(_) => "layer_norm",
                NormOp::RMSNorm(_) => "rms_norm",
                NormOp::GroupNorm(_) => "group_norm",
            }
            .to_string(),
        }
    }

    fn metadata(&self, _: &OpTensor, _: &KernelElement) -> Result<Self::Metadata, OperationError> {
        let NormKernels::Standard(inner) = self;
        let input = inner.srcs()[0];
        let rank = input.dim();
        let meta = match inner {
            NormOp::RMSNorm(n) | NormOp::LayerNorm(n) => {
                let M = input.shape()[rank - 2] as u32;
                let N = input.shape()[rank - 1] as u32;
                let ND2 = N / 2;
                let ND4 = N / 4;
                NormMeta::new(M, N, ND2, ND4, n.eps)
            }
            NormOp::GroupNorm(GroupNorm {
                norm: Norm { eps, .. },
                num_groups,
            }) => {
                let img_size = input.shape()[rank - 1] as u32;
                let channels = input.shape()[1] as u32;
                let M = *num_groups as u32;
                let N = (channels / *num_groups as u32) * img_size;
                let ND2 = N / 2;
                let ND4 = N / 4;
                NormMeta::new(M, N, ND2, ND4, *eps)
            }
        };
        Ok(meta)
    }

    fn calculate_dispatch(&self, _: &OpTensor) -> Result<Workload, OperationError> {
        let NormKernels::Standard(inner) = self;

        let input = inner.srcs()[0];
        let rank = input.dim();
        let stacks = input.shape().slice(0..rank - 2).numel();

        let workgroup_count = match inner {
            NormOp::LayerNorm(_) | NormOp::RMSNorm(_) => {
                let M = input.shape()[rank - 2] as u32;
                wgc![M as _, stacks as _, 1]
            }
            NormOp::GroupNorm(GroupNorm { num_groups, .. }) => {
                let M = *num_groups;
                wgc![M as _, stacks as _, 1]
            }
        };

        Ok(Workload {
            workgroup_count,
            workgroup_size: wgs![128, 1, 1],
        })
    }

    fn kernel_element(&self, dst: &OpTensor) -> KernelElement {
        let rank = dst.dim();
        let N = dst.shape()[rank - 1] as u32;
        if N.is_multiple_of(4) {
            KernelElement::Vec4
        } else if N.is_multiple_of(2) {
            KernelElement::Vec2
        } else {
            KernelElement::Scalar
        }
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

    fn storage_bind_group_layout(
        &self,
        _inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        let NormKernels::Standard(inner) = self;
        let norm = match inner {
            NormOp::LayerNorm(norm) | NormOp::RMSNorm(norm) => norm,
            NormOp::GroupNorm(GroupNorm { norm, .. }) => norm,
        };

        let num_input_buffers = 1 + // X (input)
            if norm.scale.is_some() { 1 } else { 0 } + // S (scale)
            if norm.bias.is_some() { 1 } else { 0 }; // B (bias)

        // +1 for output buffer Y
        match num_input_buffers {
            1 => Ok(BindGroupLayoutDescriptor::unary()), // Only X and Y
            2 => Ok(BindGroupLayoutDescriptor::binary()), // X + (S or B) + Y
            3 => Ok(BindGroupLayoutDescriptor::ternary()), // X + S + B + Y
            _ => unreachable!("Invalid number of input buffers"),
        }
    }
}

impl GPUOperation for NormOp {
    type KernelEnum = NormKernels;

    fn select_kernel(&self) -> Self::KernelEnum {
        NormKernels::Standard(self.clone())
    }
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use test_strategy::{Arbitrary, proptest};

    use crate::test_util::run_py_prg;
    use crate::{Device, DeviceRequest, Tensor, randn, rvec};

    fn ground_truth(
        var: NormVariant,
        input: &Tensor,
        scale: Option<&Tensor>,
        bias: Option<&Tensor>,
    ) -> anyhow::Result<Tensor> {
        let ln_prg = r#"
import torch
import torch.nn.functional as F
def layer_norm(input, scale, bias):
    (input, scale, bias) = (torch.from_numpy(input), torch.from_numpy(scale), torch.from_numpy(bias))
    return F.layer_norm(input, (input.shape[-1],), weight=scale, bias=bias).numpy()
"#;

        let rms_prg = r#"
import torch
def manual_rms_norm(input, scale):
    (input, scale) = (torch.from_numpy(input), torch.from_numpy(scale))
    variance = input.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    input = input * torch.rsqrt(variance + 1e-5)
    return (scale * input).numpy()
"#;
        let prg = match var {
            NormVariant::LayerNorm => ln_prg,
            NormVariant::RMSNorm => rms_prg,
        };

        let mut inputs = rvec![input];
        if let Some(scale) = scale {
            inputs.push(scale);
        }
        if let Some(bias) = bias {
            inputs.push(bias);
        }

        run_py_prg(prg.to_string(), &inputs, &[], input.dtype())
    }

    fn run_norm_trial(device: &Device, problem: NormProblem) -> anyhow::Result<()> {
        let NormProblem { var, B, M, N } = problem;
        let input = randn((B, M, N), None, None, Default::default())?;
        let scale = randn(N, None, None, Default::default())?;

        let bias = match var {
            NormVariant::LayerNorm => Some(randn(N, None, None, Default::default())?),
            NormVariant::RMSNorm => None,
        };

        let ground = match var {
            NormVariant::LayerNorm => ground_truth(var, &input, Some(&scale), bias.as_ref())?,
            NormVariant::RMSNorm => ground_truth(var, &input, Some(&scale), None)?,
        };

        let input_gpu = input.to(device)?;
        let scale_gpu = scale.to(device)?;
        let bias_gpu = bias.map(|b| b.to(device)).transpose()?;

        let result = match var {
            NormVariant::LayerNorm => input_gpu.layer_norm(Some(scale_gpu), bias_gpu, 1e-5)?,
            NormVariant::RMSNorm => input_gpu.rms_norm(Some(scale_gpu), 1e-5)?,
        };

        let ours = result.to(&Device::CPU)?;
        ground.all_close(&ours, 1e-4, 1e-4)?;
        Ok(())
    }

    #[derive(Arbitrary, Debug, Copy, Clone)]
    pub enum NormVariant {
        LayerNorm,
        RMSNorm,
    }

    #[derive(Arbitrary, Debug)]
    struct NormProblem {
        var: NormVariant,
        #[strategy(1..=3usize)]
        B: usize,
        #[strategy(1..=256usize)]
        M: usize,
        #[strategy(1..=256usize)]
        N: usize,
    }

    #[test]
    fn debug_norm() {
        let device = Device::request_device(DeviceRequest::CPU).unwrap();
        let prob = NormProblem {
            var: NormVariant::LayerNorm,
            B: 2,
            M: 57,
            N: 1001,
        };
        println!("prob = {prob:#?}");
        run_norm_trial(&device, prob).unwrap();
    }

    #[proptest(cases = 64)]
    fn test_norm_gpu(prob: NormProblem) {
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        run_norm_trial(&device, prob).unwrap();
    }

    #[proptest(cases = 64)]
    fn test_norm_cpu(prob: NormProblem) {
        let device = Device::request_device(DeviceRequest::CPU).unwrap();
        run_norm_trial(&device, prob).unwrap();
    }
}
