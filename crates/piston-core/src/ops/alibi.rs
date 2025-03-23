use derive_new::new;
use encase::ShaderType;
use inline_wgsl::wgsl;
use piston_macros::{IrFields, WgslMetadata};

use crate::{
    gpu::{BindGroupLayoutDescriptor, WorkgroupCount},
    rvec, wgc, wgs, Array, BindingMode, BuiltIn, DType, GPUOperation, Kernel, KernelElement,
    KernelRenderable, KernelSource, OpGuards, Operation, OperationError, RVec, Scalar, StorageView,
    Tensor, WgslKernelBuilder, WgslPrimitive, WorkgroupSize, Workload,
};

/// Implements "alibi" (Attention Linear Bias)
#[derive(new, Debug, Clone, IrFields)]
pub struct Alibi {
    /// The input tensor, assumed to be rank 3: [B, n_head, seq].
    pub input: Tensor,
    /// The maximum bias to be distributed across heads.
    max_bias: f32,
}

impl OpGuards for Alibi {
    fn check_shapes(&self) {
        let shape = self.input.shape();
        assert!(
            shape.len() == 3,
            "Alibi expects a 3D shape [B, n_head, seq], got shape={:?}",
            shape
        );
    }

    fn check_dtypes(&self) {
        // For simplicity, require float32 for now.
        assert!(
            self.input.dtype() == DType::F32,
            "Alibi only supports F32 for now"
        );
    }
}

impl Operation for Alibi {
    fn name(&self) -> &'static str {
        "Alibi"
    }

    fn compute_view(&self) -> Result<StorageView, OperationError> {
        // Output has the same shape and dtype as the input.
        Ok(self.input.storage_view().clone())
    }

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.input]
    }

    fn supports_inplace(&self) -> bool {
        true
    }
}

pub enum AlibiKernels {
    Standard(Alibi),
}

impl GPUOperation for Alibi {
    type KernelEnum = AlibiKernels;

    fn select_kernel(&self) -> Self::KernelEnum {
        AlibiKernels::Standard(self.clone())
    }
}

#[derive(Debug, derive_new::new, ShaderType, WgslMetadata)]
pub struct AlibiMeta {
    /// number of columns (seq length)
    ncols: u32,
    /// number of rows (batch*n_head)
    nrows: u32,
    /// number of heads
    n_heads: u32,
    /// floor( log2(n_heads) ), stored as an integer
    n_heads_log2_floor: u32,
    /// base exponent for "lower" heads
    m0: f32,
    /// base exponent for "upper" heads
    m1: f32,
}

impl KernelRenderable for AlibiKernels {
    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        inplace: bool,
    ) -> Result<(), OperationError> {
        let arr = Array::<P>::default();
        if inplace {
            builder.register_storage("in", BindingMode::ReadWrite, arr);
        } else {
            builder.register_storage("in", BindingMode::ReadOnly, arr);
            builder.register_storage("out", BindingMode::ReadWrite, arr);
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
                BuiltIn::GlobalInvocationId,
                BuiltIn::LocalInvocationId,
                BuiltIn::NumWorkgroups,
            ],
            device.compute_features().clone(),
        );

        self.register_bindings::<P>(&mut kernel_builder, inplace)?;
        kernel_builder.render_metadata(&self.metadata(dst, &self.kernel_element(dst))?);

        let dtype = P::render_type();

        let kernel = if inplace {
            wgsl! {
                let row = global_invocation_id.y;
                let col = global_invocation_id.x;
                if (col >= metadata.ncols || row >= metadata.nrows) {
                    return;
                }

                let i = row * metadata.ncols + col;

                let head_idx = row % metadata.n_heads;

                var m_k = 0.0;
                if (head_idx < metadata.n_heads_log2_floor) {
                    m_k = pow(metadata.m0, f32(head_idx + 1u));
                } else {
                    let extra = head_idx - metadata.n_heads_log2_floor;
                    let exponent = f32( (extra * 2u) + 1u );
                    m_k = pow(metadata.m1, exponent);
                }

                in[i] = in[i] + 'dtype(col) * m_k;
            }
        } else {
            wgsl! {
                let row = global_invocation_id.y;
                let col = global_invocation_id.x;
                if (col >= metadata.ncols || row >= metadata.nrows) {
                    return;
                }

                let i = row * metadata.ncols + col;

                let head_idx = row % metadata.n_heads;

                var m_k = 0.0;
                if (head_idx < metadata.n_heads_log2_floor) {
                    m_k = pow(metadata.m0, f32(head_idx + 1u));
                } else {
                    let extra = head_idx - metadata.n_heads_log2_floor;
                    let exponent = f32( (extra * 2u) + 1u );
                    m_k = pow(metadata.m1, exponent);
                }

                out[i] = in[i] + 'dtype(col) * m_k;
            }
        };

        kernel_builder.write_main(kernel);
        Ok(kernel_builder.build()?)
    }
}

impl Kernel for AlibiKernels {
    type Metadata = AlibiMeta;

    fn kernel_name(&self) -> String {
        match self {
            AlibiKernels::Standard(_) => "alibi".to_string(),
        }
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
        let AlibiKernels::Standard(inner) = self;
        let shape = inner.input.shape();
        let b = shape[0];
        let n_head = shape[1];
        let seq = shape[2];

        let head_log2 = (n_head as f32).log2().floor().max(0.0) as u32;
        let n_heads_log2_floor = 1 << head_log2;

        let nrows = b * n_head;

        let m0 = 2f32.powf(-inner.max_bias / (n_heads_log2_floor as f32));
        let m1 = 2f32.powf(-(inner.max_bias / 2.0) / (n_heads_log2_floor as f32));

        Ok(AlibiMeta::new(
            seq as u32,
            nrows as u32,
            n_head as u32,
            n_heads_log2_floor as u32,
            m0,
            m1,
        ))
    }

    fn kernel_element(&self, _dst: &Tensor) -> KernelElement {
        KernelElement::Scalar
    }

    fn calculate_dispatch(&self, _dst: &Tensor) -> Result<Workload, OperationError> {
        const WGSX: usize = 16;
        const WGSY: usize = 16;
        const WGSZ: usize = 1;
        let workgroup_size = wgs![WGSX as u32, WGSY as u32, WGSZ as u32];

        let AlibiKernels::Standard(inner) = self;
        let shape = inner.input.shape();
        let b = shape[0];
        let n_head = shape[1];
        let seq = shape[2];

        let nrows = b * n_head;
        let ncols = seq;

        let wgcx = WorkgroupCount::div_ceil(ncols, WGSX) as u32;
        let wgcy = WorkgroupCount::div_ceil(nrows, WGSY) as u32;
        let wgcz = 1u32;

        Ok(Workload {
            workgroup_count: wgc![wgcx, wgcy, wgcz],
            workgroup_size,
        })
    }

    fn build_kernel(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let kernel_element = self.kernel_element(dst);
        let AlibiKernels::Standard(inner) = self;
        match (inner.input.dtype(), &kernel_element) {
            (DType::F32, KernelElement::Scalar) => {
                self.render::<Scalar<f32>>(inplace, dst, workgroup_size)
            }
            // TODO: extend to F16/vector types
            _ => Err(OperationError::CompileError(format!(
                "Unsupported dtype {:?} or kernel element {:?}",
                inner.input.dtype(),
                kernel_element
            ))),
        }
    }
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use crate::{test_util::run_py_prg, DType, Device, DeviceRequest, Tensor};
    use test_strategy::{proptest, Arbitrary};

    /// Ground truth reference, computed by a Python snippet:
    ///
    /// This snippet exactly follows the logic in the Alibi kernel:
    ///   - shape = [B, n_head, seq]
    ///   - n_heads_log2_floor = 1 << floor(log2(n_head))
    ///   - m0 = 2^(-(max_bias) / n_heads_log2_floor)
    ///   - m1 = 2^(-(max_bias / 2.0) / n_heads_log2_floor)
    ///   - for each head in 0..n_head, offset = head - n_heads_log2_floor if head >= that
    ///     then out[b, head, col] += col * factor
    fn ground_truth_alibi(a: &Tensor, max_bias: f32) -> anyhow::Result<Tensor> {
        let prg = r#"
import numpy as np
import math

def alibi(input, max_bias):
    # Input shape: [B, n_head, seq]
    b, n_head, seq = input.shape
    # floor(log2(n_head)), but we do "1 << floor(...)" in Rust.
    # In python, we can do the same with an int shift:
    exponent = int(math.floor(math.log2(n_head))) if n_head > 0 else 0
    n_heads_log2_floor = 1 << exponent

    m0 = 2.0 ** (-max_bias / n_heads_log2_floor)
    m1 = 2.0 ** (-(max_bias/2.0) / n_heads_log2_floor)

    out = np.copy(input)
    for ib in range(b):
        for ih in range(n_head):
            if ih < n_heads_log2_floor:
                factor_base = m0
                exponent_i = (ih + 1)
            else:
                factor_base = m1
                exponent_i = 2*(ih - n_heads_log2_floor) + 1
            factor = factor_base ** exponent_i

            for c in range(seq):
                out[ib, ih, c] += c * factor
    return out
"#;
        run_py_prg(prg.to_string(), &[a], &[&max_bias], a.dtype())
    }

    #[derive(Arbitrary, Debug)]
    struct AlibiProblem {
        #[strategy(1..=4usize)]
        b: usize,
        #[strategy(1..=8usize)]
        n_head: usize,
        #[strategy(1..=64usize)]
        seq: usize,
        /// We allow a smallish range for max_bias just to keep values from blowing up.
        #[strategy(0.1f32..=5.0f32)]
        max_bias: f32,
    }

    fn run_alibi_trial(problem: AlibiProblem, device: Device) {
        let AlibiProblem {
            b,
            n_head,
            seq,
            max_bias,
        } = problem;

        // shape = [B, n_head, seq]
        let shape = (b, n_head, seq);
        // let a_cpu = Tensor::randn::<f32, _>(0.0, 1.0, shape, Device::CPU).unwrap();
        let a_cpu = Tensor::zeros::<f32, _>(shape, &Device::CPU).unwrap();

        let ground = ground_truth_alibi(&a_cpu, max_bias).unwrap();
        let a_gpu = a_cpu.to(&device).unwrap();
        let out_gpu = a_gpu.alibi(max_bias).unwrap();

        let out_cpu = out_gpu.to(&Device::CPU).unwrap();

        println!("Problem: {:?}", problem);
        println!("Ground truth shape: {:?}", ground.shape());
        println!("Output shape: {:?}", out_cpu.shape());
        println!("Ground truth: {:?}", ground.to_ndarray_view::<f32>());
        println!("Output: {:?}", out_cpu.to_ndarray_view::<f32>());

        // Compare
        ground.all_close(&out_cpu, 1e-4, 1e-4).unwrap();
    }

    /// Tests over randomly generated shapes on the GPU
    #[proptest(cases = 8)]
    fn test_alibi_gpu(prob: AlibiProblem) {
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        run_alibi_trial(prob, device);
    }

    // /// Tests over randomly generated shapes on the CPU
    // #[proptest(cases = 8)]
    // fn test_alibi_cpu(prob: AlibiProblem) {
    //     let device = Device::request_device(DeviceRequest::CPU).unwrap();
    //     run_alibi_trial(prob, device);
    // }
}
