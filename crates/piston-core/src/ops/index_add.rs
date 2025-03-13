use crate::{
    gpu::BindGroupLayoutDescriptor, rvec, Array, BindingMode, BuiltIn, DType, GPUOperation, Kernel,
    KernelElement, KernelRenderable, KernelSource, OpGuards, Operation, OperationError, RVec,
    Scalar, StorageView, Tensor, Vec2, Vec4, WgslKernelBuilder, WgslPrimitive, WorkgroupSize,
    Workload,
};
use derive_new::new;
use encase::ShaderType;
use half::f16;
use inline_wgsl::wgsl;
use piston_macros::{IrFields, WgslMetadata};

#[derive(new, Debug, Clone, IrFields)]
pub struct IndexAdd {
    pub dst: Tensor,
    pub src: Tensor,
    pub ids: Tensor,
    pub dim: usize,
}

#[derive(Debug, derive_new::new, ShaderType, WgslMetadata)]
pub struct IndexAddMeta {
    left_size: u32,
    right_size: u32,
    dst_dim_size: u32,
    ids_dim_size: u32,
}

impl OpGuards for IndexAdd {
    fn check_shapes(&self) {
        let (input, indices) = (&self.src, &self.ids);
        assert_eq!(input.rank(), 2);
        assert_eq!(indices.rank(), 1);
    }

    fn check_dtypes(&self) {
        let indices = &self.ids;
        assert_eq!(indices.dtype(), DType::I32);
    }
}

impl Operation for IndexAdd {
    fn name(&self) -> &'static str {
        "IndexAdd"
    }

    fn compute_view(&self) -> Result<StorageView, OperationError> {
        Ok(self.dst.storage_view().clone())
    }

    #[inline]
    fn srcs(&self) -> RVec<&Tensor> {
        rvec![&self.dst, &self.src, &self.ids]
    }

    fn supports_inplace(&self) -> bool {
        true
    }
}

impl GPUOperation for IndexAdd {
    type KernelEnum = IndexAddKernels;

    fn select_kernel(&self) -> Self::KernelEnum {
        IndexAddKernels::Standard(self.clone())
    }
}

pub enum IndexAddKernels {
    Standard(IndexAdd),
}

impl KernelRenderable for IndexAddKernels {
    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        _: bool,
    ) -> Result<(), OperationError> {
        let arr = Array::<P>::default();
        builder.register_storage("X", BindingMode::ReadWrite, arr);
        builder.register_storage("S", BindingMode::ReadOnly, arr);
        builder.register_storage("I", BindingMode::ReadOnly, Array::<Scalar<i32>>::default());
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
                BuiltIn::LocalInvocationIndex,
                BuiltIn::NumWorkgroups,
                BuiltIn::WorkgroupId,
            ],
            device.compute_features().clone(),
        );
        self.register_bindings::<P>(&mut kernel_builder, inplace)?;

        kernel_builder.render_metadata(&self.metadata(dst, &self.kernel_element(dst))?);
        kernel_builder.write_index_to_offset();

        kernel_builder.write_main(wgsl! {
            let numel = metadata.left_size * metadata.right_size;

            for (var i = workgroup_id.x * 64u + local_invocation_index; i < numel; i += 64u * num_workgroups.x) {
                let pre = i / metadata.right_size;
                let post = i % metadata.right_size;
                for (var j = 0u; j < metadata.ids_dim_size; j++) {
                    let idx = u32(I[j]);
                    let src_i = (pre * metadata.ids_dim_size + j) * metadata.right_size + post;
                    let dst_i = (pre * metadata.dst_dim_size + idx) * metadata.right_size + post;
                    X[dst_i] += S[src_i];
                }
            }
        });

        Ok(kernel_builder.build()?)
    }
}

impl Kernel for IndexAddKernels {
    type Metadata = IndexAddMeta;

    fn storage_bind_group_layout(
        &self,
        inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        if !inplace {
            panic!("IndexAdd only supports inplace operation");
        }
        Ok(BindGroupLayoutDescriptor::ternary_inplace())
    }

    fn calculate_dispatch(&self, _: &Tensor) -> Result<Workload, OperationError> {
        let IndexAddKernels::Standard(inner) = self;
        Ok(Workload::std(
            inner.src.shape().numel(),
            KernelElement::Scalar,
        ))
    }

    fn kernel_name(&self) -> String {
        match self {
            IndexAddKernels::Standard(_) => "index_add".to_string(),
        }
    }

    fn kernel_element(&self, _dst: &Tensor) -> KernelElement {
        KernelElement::Scalar
    }

    fn metadata(&self, _: &Tensor, _: &KernelElement) -> Result<Self::Metadata, OperationError> {
        let IndexAddKernels::Standard(inner) = self;
        let dim = inner.dim;
        let src_shape_vec = inner.src.shape().to_vec();
        let dst_shape_vec = inner.dst.shape().to_vec();
        let ids_shape_vec = inner.ids.shape().to_vec();
        let left_size = src_shape_vec[..dim].iter().product::<usize>() as u32;
        let right_size = src_shape_vec[dim + 1..].iter().product::<usize>() as u32;
        let dst_dim_size = dst_shape_vec[dim] as u32;
        let ids_dim_size = ids_shape_vec[0] as u32;
        Ok(IndexAddMeta {
            left_size,
            right_size,
            dst_dim_size,
            ids_dim_size,
        })
    }

    fn build_kernel(
        &self,
        inplace: bool,
        dst: &Tensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let kernel_element = self.kernel_element(dst);
        let IndexAddKernels::Standard(inner) = self;
        match (inner.src.dtype(), &kernel_element) {
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
                inner.src.dtype(),
                kernel_element
            ))),
        }
    }
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use proptest::arbitrary::Arbitrary;
    use proptest::strategy::{BoxedStrategy, Just, Strategy};
    use test_strategy::proptest;

    use crate::test_util::run_py_prg;
    use crate::{rvec, shape, Device, DeviceRequest, Shape, Tensor};

    impl Arbitrary for IndexAddProblem {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            Shape::arbitrary_with(vec![1..=512usize, 1..=16usize])
                .prop_flat_map(|input_shape| (Just(input_shape), 1..64usize))
                .prop_map(|(input_shape, num_indices)| {
                    let indices =
                        Tensor::randint(0, input_shape[0] as i32, shape![num_indices], Device::CPU)
                            .unwrap();
                    IndexAddProblem {
                        input_shape,
                        indices,
                    }
                })
                .boxed()
        }
    }

    fn ground_truth(
        input: &Tensor,
        source: &Tensor,
        indices: &Tensor,
        dim: usize,
    ) -> anyhow::Result<Tensor> {
        let prg = format!(
            r#"
import torch
def index_add(input, source, indices):
    return torch.index_add(torch.from_numpy(input),{},torch.from_numpy(indices),torch.from_numpy(source)).numpy()
"#,
            dim
        );
        run_py_prg(
            prg.to_string(),
            &[input, source, indices],
            &[],
            input.dtype(),
        )
    }

    fn run_index_add_trial(problem: IndexAddProblem, device: Device) {
        let IndexAddProblem {
            input_shape,
            indices,
        } = problem;
        let mut source_shape = input_shape.clone();
        source_shape[0] = indices.shape()[0];

        let input = Tensor::randn::<f32>(0., 1., input_shape.clone(), device.clone())
            .unwrap()
            .to(&Device::CPU)
            .unwrap();

        let source = Tensor::randn::<f32>(0., 1., source_shape.clone(), device.clone())
            .unwrap()
            .to(&Device::CPU)
            .unwrap();

        log::debug!(
            "shapes: input = {:?}, source = {:?}, indices = {:?}",
            input_shape,
            source_shape,
            indices.shape()
        );

        let ground_truth = ground_truth(&input, &source, &indices, 0).unwrap();

        log::debug!("input = {:?}", input);
        log::debug!("source = {:?}", source);
        log::debug!("ground_truth = {:?}", ground_truth);
        log::debug!("indices = {:?}", indices);

        let input = input.to(&device).unwrap();
        let indices = indices.to(&device).unwrap();
        let source = source.to(&device).unwrap();

        let result = input.index_add(indices.clone(), source.clone(), 0).unwrap();
        let x = result.to(&Device::CPU).unwrap();

        log::debug!("x = {:?}", x);

        ground_truth.all_close(&x, 1e-1, 1e-1).unwrap();
    }

    #[derive(Debug, Clone)]
    struct IndexAddProblem {
        input_shape: Shape,
        indices: Tensor,
    }

    #[proptest(cases = 16)]
    fn test_index_add(prob: IndexAddProblem) {
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        run_index_add_trial(prob, device);
    }
}
