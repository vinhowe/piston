use derive_new::new;
use encase::ShaderType;
use half::f16;
use inline_wgsl::wgsl;
use piston_macros::{IrFields, WgslMetadata};

use crate::{
    gpu::BindGroupLayoutDescriptor, rvec, Array, BindingMode, BuiltIn, DType, GPUOperation, Kernel,
    KernelElement, KernelRenderable, KernelSource, OpGuards, Operation, OperationError, RVec,
    Scalar, Shape, StorageView, Stride, Tensor, Vec2, Vec4, WgslKernelBuilder, WgslPrimitive,
    WorkgroupSize, Workload,
};

#[derive(new, Debug, Clone, IrFields)]
pub struct Trilu {
    pub src: Tensor,
    pub upper: bool,
    pub k: Option<i32>,
}

#[derive(Debug, derive_new::new, ShaderType, WgslMetadata)]
pub struct TriluMeta {
    k: i32,
    rows: u32,
    cols: u32,
    stride: u32,
    numel: u32,
}

impl Operation for Trilu {
    fn name(&self) -> &'static str {
        "Trilu"
    }

    fn compute_view(&self) -> Result<StorageView, OperationError> {
        let shape: Shape = self.src.shape().clone();
        let stride = Stride::from(&shape);
        Ok(StorageView::new(shape, crate::DType::F32, stride))
    }

    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.src]
    }

    fn supports_inplace(&self) -> bool {
        true // Trilu can be done inplace
    }
}

impl OpGuards for Trilu {
    fn check_shapes(&self) {}

    fn check_dtypes(&self) {}
}

pub enum TriluKernels {
    Standard(Trilu),
}

impl GPUOperation for Trilu {
    type KernelEnum = TriluKernels;

    fn select_kernel(&self) -> Self::KernelEnum {
        TriluKernels::Standard(self.clone())
    }
}

impl KernelRenderable for TriluKernels {
    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        inplace: bool,
    ) -> Result<(), OperationError> {
        if inplace {
            builder.register_storage("X", BindingMode::ReadWrite, Array::<P>::default());
        } else {
            builder.register_storage("X", BindingMode::ReadOnly, Array::<P>::default());
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
                BuiltIn::LocalInvocationIndex,
                BuiltIn::NumWorkgroups
            ],
            device.compute_features().clone(),
        );

        self.register_bindings::<P>(&mut kernel_builder, inplace)?;
        kernel_builder.render_metadata(&self.metadata(dst, &self.kernel_element(dst))?);

        kernel_builder.write_main(wgsl! {
            let x_offset = workgroup_id.x * 64u;
            let index = (workgroup_id.y * num_workgroups.x * 64u) + x_offset + local_invocation_index;

            if (index >= metadata.numel) {
                return;
            }

            let batch_size = metadata.rows * metadata.cols * metadata.stride;
            let batch_index = index / batch_size;

            let local_index = index % batch_size;

            let col = local_index % metadata.cols;
            let row = (local_index / metadata.cols) % metadata.rows;
        });

        let TriluKernels::Standard(inner) = self;

        if inner.upper {
            kernel_builder.write_main(wgsl! {
                let condition = bool(i32(col) >= i32(row) + metadata.k);
            });
        } else {
            kernel_builder.write_main(wgsl! {
                let condition = bool(i32(col) <= i32(row) + metadata.k);
            });
        }

        if inplace {
            kernel_builder.write_main(wgsl! {
                if !condition {
                    X[index] = 0.0;
                }
            });
        } else {
            kernel_builder.write_main(wgsl! {
                if condition {
                    Y[index] = X[index];
                } else {
                    Y[index] = 0.0;
                }
            });
        }

        Ok(kernel_builder.build()?)
    }
}

impl Kernel for TriluKernels {
    type Metadata = TriluMeta;

    fn kernel_name(&self) -> String {
        match self {
            TriluKernels::Standard(_) => "trilu".to_string(),
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
        inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        if inplace {
            Ok(BindGroupLayoutDescriptor::unary_inplace())
        } else {
            Ok(BindGroupLayoutDescriptor::unary())
        }
    }

    fn metadata(&self, _: &Tensor, _: &KernelElement) -> Result<Self::Metadata, OperationError> {
        let TriluKernels::Standard(inner) = self;
        let shape = inner.src.shape();
        let ndim = shape.len();
        Ok(TriluMeta {
            numel: shape.clone().numel() as u32,
            k: inner.k.unwrap_or(0),
            rows: shape[ndim - 2] as u32,
            cols: shape[ndim - 1] as u32,
            stride: shape[..ndim - 2].iter().product::<usize>() as u32,
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

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use crate::{shape, test_util::run_py_prg, DType, Device, DeviceRequest, Tensor};
    use proptest::prelude::any;
    use test_strategy::{proptest, Arbitrary};

    /// Generates the ground truth tensor using NumPy's triu or tril functions.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor (expected to be at least 2D).
    /// * `upper` - If `true`, retains the upper triangular part; otherwise, the lower.
    /// * `k` - The diagonal offset. Positive values shift the diagonal upwards,
    ///         negative values shift it downwards.
    ///
    /// # Returns
    ///
    /// A `Tensor` containing the ground truth data.
    fn ground_truth(shape: &[usize], upper: bool, k: Option<i32>) -> anyhow::Result<Tensor> {
        let prg = r#"
import numpy as np
def trilu(shape, upper, k):
    if len(shape) < 2:
        raise ValueError("Trilu operation requires at least a 2D shape.")
    
    # Initialize a tensor with ones
    tensor = np.ones(shape, dtype=np.float32)
    
    # Apply triu or tril to each 2D slice if batched
    if len(shape) == 2:
        if upper:
            return np.triu(tensor, k=k)
        else:
            return np.tril(tensor, k=k)
    else:
        # Assume batched tensors; apply along the last two dimensions
        for i in range(shape[0]):
            if upper:
                tensor[i] = np.triu(tensor[i], k=k)
            else:
                tensor[i] = np.tril(tensor[i], k=k)
        return tensor
"#;

        run_py_prg(prg.to_string(), &[], &[&shape, &upper, &k], DType::F32)
    }

    /// Represents a single test case for the Trilu operation.
    #[derive(Arbitrary, Debug)]
    struct TriluProblem {
        /// Batch size. Supports both batched and single matrix operations.
        #[strategy(1..=16usize)]
        B: usize,

        /// Number of rows in the matrix.
        #[strategy(1..=256usize)]
        M: usize,

        /// Number of columns in the matrix.
        #[strategy(1..=256usize)]
        N: usize,

        /// Determines whether to retain the upper or lower triangular part.
        #[strategy(any::<bool>())]
        upper: bool,

        /// Diagonal offset. Positive values move the diagonal upwards, negative downwards.
        #[strategy(-10..=10i32)]
        k: i32,
    }

    /// Executes a single trial of the Trilu operation and compares the result with ground truth.
    ///
    /// # Arguments
    ///
    /// * `problem` - The test case parameters.
    /// * `device` - The GPU device to execute the operation on.
    ///
    /// # Panics
    ///
    /// The function will panic if any step of the operation fails or if the results do not match.
    fn run_trilu_trial(problem: TriluProblem, device: Device) {
        let TriluProblem { B, M, N, upper, k } = problem;

        // Define the shape of the tensor.
        let shape = shape![B, M, N];

        let src = Tensor::ones::<f32>(&shape, &device).unwrap();

        // Generate the ground truth using NumPy.
        let ground = ground_truth(&shape, upper, Some(k))
            .expect("Failed to generate ground truth using NumPy.");

        // Transfer the GPU result back to the CPU for comparison.
        let ours = match upper {
            true => src.triu(Some(k)),
            false => src.tril(Some(k)),
        }
        .unwrap()
        .to(&Device::CPU)
        .unwrap();

        println!("Ours: {:?}", ours);
        println!("Ground: {:?}", ground);

        // Compare the GPU result with the ground truth.
        ground
            .all_close(&ours, 1e-6, 1e-6)
            .expect("Trilu operation result does not match ground truth.");
    }

    /// Executes multiple trials of the Trilu operation with randomly generated test cases.
    ///
    /// # Arguments
    ///
    /// * `prob` - The test case parameters.
    #[proptest(cases = 16)]
    fn test_trilu(prob: TriluProblem) {
        // Request a GPU device. This will panic if no GPU device is available.
        let device =
            Device::request_device(DeviceRequest::GPU).expect("Failed to request GPU device.");

        // Run the trial with the generated problem and device.
        run_trilu_trial(prob, device);
    }
}
