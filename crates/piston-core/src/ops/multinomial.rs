use derive_new::new;
use encase::ShaderType;
use half::f16;
use inline_wgsl::wgsl;
use piston_macros::{IrFields, WgslMetadata};

use crate::{
    Array, BindingMode, BuiltIn, DType, GPUOperation, Kernel, KernelElement, KernelRenderable,
    KernelSource, OpGuards, OpTensor, Operation, OperationError, RVec, Scalar, StorageView, Stride,
    Vec2, Vec4, WgslKernelBuilder, WgslPrimitive, WorkgroupSize, Workload,
    gpu::BindGroupLayoutDescriptor, rvec,
};

#[derive(new, Debug, Clone, IrFields)]
pub struct Multinomial {
    pub probs: OpTensor,
    pub num_samples: usize,
    pub replacement: bool,
    pub seed: Option<u32>,
}

#[derive(Debug, derive_new::new, ShaderType, WgslMetadata)]
pub struct MultinomialMeta {
    num_rows: u32,
    row_size: u32,
    num_samples: u32,
    replacement: u32,
    seed: u32,
}

impl Operation for Multinomial {
    fn name(&self) -> &'static str {
        "Multinomial"
    }

    fn compute_view(&self) -> Result<StorageView, OperationError> {
        // Supports [V] -> [S] and [B, V] -> [B, S]
        let shape = self.probs.shape();
        assert!(
            shape.dim() == 1 || shape.dim() == 2,
            "Multinomial: input must be 1D or 2D"
        );
        let out_shape = if shape.dim() == 1 {
            crate::Shape::from(rvec![self.num_samples])
        } else {
            crate::Shape::from(rvec![shape[0], self.num_samples])
        };
        let stride = Stride::from(&out_shape);
        Ok(StorageView::new(out_shape, crate::DType::I32, stride))
    }

    fn srcs(&self) -> RVec<&OpTensor> {
        rvec![&self.probs]
    }

    fn supports_inplace(&self) -> bool {
        false
    }
}

impl OpGuards for Multinomial {
    fn check_shapes(&self) {
        let shape = self.probs.shape();
        assert!(
            shape.dim() == 1 || shape.dim() == 2,
            "Multinomial: input must be 1D or 2D"
        );
    }

    fn check_dtypes(&self) {
        assert!(
            self.probs.dtype().is_float(),
            "Multinomial: probs must be float dtype"
        );
    }
}

pub enum MultinomialKernels {
    Standard(Multinomial),
}

impl GPUOperation for Multinomial {
    type KernelEnum = MultinomialKernels;

    fn select_kernel(&self) -> Self::KernelEnum {
        MultinomialKernels::Standard(self.clone())
    }
}

impl KernelRenderable for MultinomialKernels {
    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        _: bool,
    ) -> Result<(), OperationError> {
        builder.register_storage("X", BindingMode::ReadOnly, Array::<P>::default());
        builder.register_storage("Y", BindingMode::ReadWrite, Array::<Scalar<i32>>::default());
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

            fn already_selected(row: u32, s: u32, j: u32, num_samples: u32) -> bool {
                var t: u32 = 0u;
                loop {
                    if (t >= s) { break; }
                    if (u32(Y[row * num_samples + t]) == j) { return true; }
                    t = t + 1u;
                }
                return false;
            }
        });

        kernel_builder.write_main(wgsl! {
            let x_offset = workgroup_id.x * 64u;
            let row = (workgroup_id.y * num_workgroups.x * 64u) + x_offset + local_invocation_index;
            if (row >= metadata.num_rows) {
                return;
            }

            let base = row * metadata.row_size;

            var s: u32 = 0u;
            loop {
                if (s >= metadata.num_samples) { break; }

                var total: f32 = 0.0;
                var j: u32 = 0u;
                loop {
                    if (j >= metadata.row_size) { break; }
                    if (
                        metadata.replacement == 1u
                        || !already_selected(row, s, j, metadata.num_samples)
                    ) {
                        total = total + X[base + j];
                    }
                    j = j + 1u;
                }

                let r = rand((row + s) ^ metadata.seed) * total;
                var acc: f32 = 0.0;
                var chosen: u32 = 0u;
                j = 0u;
                loop {
                    if (j >= metadata.row_size) { break; }
                    if (
                        metadata.replacement == 1u
                        || !already_selected(row, s, j, metadata.num_samples)
                    ) {
                        acc = acc + X[base + j];
                        if (acc >= r) {
                            chosen = j;
                            break;
                        }
                    }
                    j = j + 1u;
                }

                Y[row * metadata.num_samples + s] = i32(chosen);
                s = s + 1u;
            }
        });

        Ok(kernel_builder.build()?)
    }
}

impl Kernel for MultinomialKernels {
    type Metadata = MultinomialMeta;

    fn kernel_name(&self) -> String {
        match self {
            MultinomialKernels::Standard(_) => "multinomial".to_string(),
        }
    }

    fn kernel_element(&self, _dst: &OpTensor) -> KernelElement {
        KernelElement::Scalar
    }

    fn calculate_dispatch(&self, dst: &OpTensor) -> Result<Workload, OperationError> {
        let num_rows = if dst.shape().dim() == 1 {
            1
        } else {
            dst.shape()[0]
        };
        Ok(Workload::std(num_rows, self.kernel_element(dst)))
    }

    fn storage_bind_group_layout(
        &self,
        _inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        Ok(BindGroupLayoutDescriptor::unary())
    }

    fn metadata(
        &self,
        _dst: &OpTensor,
        _: &KernelElement,
    ) -> Result<Self::Metadata, OperationError> {
        let MultinomialKernels::Standard(inner) = self;
        let shape = inner.probs.shape();
        let (num_rows, row_size) = if shape.dim() == 1 {
            (1u32, shape[0] as u32)
        } else {
            (shape[0] as u32, shape[1] as u32)
        };
        Ok(MultinomialMeta {
            num_rows,
            row_size,
            num_samples: inner.num_samples as u32,
            replacement: if inner.replacement { 1 } else { 0 },
            seed: inner.seed.unwrap_or(0),
        })
    }

    fn build_kernel(
        &self,
        inplace: bool,
        dst: &OpTensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let MultinomialKernels::Standard(inner) = self;
        let kernel_element = self.kernel_element(dst);
        match (inner.probs.dtype(), &kernel_element) {
            (DType::F32, KernelElement::Scalar) => {
                self.render::<Scalar<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Scalar) => {
                self.render::<Scalar<f16>>(inplace, dst, workgroup_size)
            }
            (DType::F32, KernelElement::Vec2) => {
                self.render::<Vec2<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F32, KernelElement::Vec4) => {
                self.render::<Vec4<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Vec2) => {
                self.render::<Vec2<f16>>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Vec4) => {
                self.render::<Vec4<f16>>(inplace, dst, workgroup_size)
            }
            _ => Err(OperationError::CompileError(format!(
                "Unsupported dtype {:?} or kernel element {:?}",
                inner.probs.dtype(),
                kernel_element
            ))),
        }
    }
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use crate::{Device, DeviceRequest, Tensor, TensorOptions};

    #[test]
    fn multinomial_vector_single_hot_replacement_true() {
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        let probs = Tensor::from_data(vec![0.0f32, 0.0, 1.0, 0.0], 4, TensorOptions::new())
            .unwrap()
            .to(&device)
            .unwrap();
        // Sample 3 with replacement; should always pick index 2
        let out = probs
            .multinomial(3usize, true)
            .unwrap()
            .to(&Device::CPU)
            .unwrap();
        let vals = out.to_vec::<i32>().unwrap();
        assert!(vals.iter().all(|&x| x == 2));
    }

    #[test]
    fn multinomial_matrix_rowwise_single_hot_replacement_false() {
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        // Two rows: [0,1,0] and [1,0,0]
        let probs = Tensor::from_data(
            vec![0.0f32, 1.0, 0.0, 1.0, 0.0, 0.0],
            (2, 3),
            TensorOptions::new(),
        )
        .unwrap()
        .to(&device)
        .unwrap();
        // Sample 1 without replacement along last dim
        let out = probs
            .multinomial(1usize, false)
            .unwrap()
            .to(&Device::CPU)
            .unwrap();
        let vals = out.to_vec::<i32>().unwrap();
        // Shape is [2,1] => vals[0], vals[1]
        assert_eq!(vals.len(), 2);
        assert_eq!(vals[0], 1);
        assert_eq!(vals[1], 0);
    }
}
