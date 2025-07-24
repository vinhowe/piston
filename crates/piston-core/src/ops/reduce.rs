use encase::ShaderType;
use half::f16;
use inline_wgsl::wgsl;
use piston_macros::{IrFields, WgslMetadata};

use crate::{
    gpu::{dtype::WgslDType, BindGroupLayoutDescriptor},
    rvec, wgc, wgs, Array, BindingMode, BuiltIn, DType, GPUOperation, Kernel, KernelElement,
    KernelRenderable, KernelSource, OpGuards, OpTensor, Operation, OperationError, RVec, Scalar,
    Shape, StorageView, Stride, WgslKernelBuilder, WgslPrimitive, WorkgroupSize, Workload,
};

#[cfg(test)]
use test_strategy::Arbitrary;

#[cfg_attr(test, derive(Arbitrary))]
#[derive(Debug, Clone, Hash, IrFields)]
pub enum ReduceOp {
    Sum,
    Min,
    Max,
    ArgMin,
    ArgMax,
    Norm2,
}

impl ReduceOp {
    pub fn kernel_name(&self) -> &'static str {
        match self {
            ReduceOp::Sum => "sum",
            ReduceOp::Min => "min",
            ReduceOp::Max => "max",
            ReduceOp::ArgMin => "argmin",
            ReduceOp::ArgMax => "argmax",
            ReduceOp::Norm2 => "norm2",
        }
    }
}

#[derive(Debug, Clone, IrFields)]
pub struct Reduce {
    pub input: OpTensor,
    pub reduced_shape: Shape,
    pub keepdim: bool,
    pub op: ReduceOp,
    reduce_dims: RVec<usize>,
    src_numel: usize,
    dst_numel: usize,
    dims: RVec<usize>,
    stride: RVec<usize>,
    el_to_reduce_per_block: usize,
}

impl Reduce {
    pub fn new(input: OpTensor, op: ReduceOp, reduce_dims: RVec<usize>, keepdim: bool) -> Self {
        // TODO: These are common to all reduce operations; we should make this more general
        let src_stride = input.stride().to_vec();
        let src_dims = input.shape().to_vec();
        let src_numel = src_dims.iter().product();

        let mut dims = rvec![];
        let mut stride = rvec![];
        let mut dst_numel: usize = 1;

        for (dim_idx, &d) in src_dims.iter().enumerate() {
            if !reduce_dims.contains(&dim_idx) {
                dst_numel *= d;
                dims.push(d);
                stride.push(src_stride[dim_idx] as usize);
            }
        }
        for &dim_idx in reduce_dims.iter() {
            dims.push(src_dims[dim_idx]);
            stride.push(src_stride[dim_idx] as usize);
        }

        // I _think_ this stays the same if we split on the right dimension
        let el_to_sum_per_block = src_numel / dst_numel;

        let mut reduced_shape_vec = src_dims.clone();

        for &dim in reduce_dims.iter() {
            reduced_shape_vec[dim] = 1;
        }

        Self {
            input,
            reduce_dims,
            reduced_shape: Shape::from(reduced_shape_vec),
            op,
            keepdim,
            src_numel,
            dst_numel,
            dims,
            stride,
            el_to_reduce_per_block: el_to_sum_per_block,
        }
    }

    fn render_block_reduce(
        &self,
        kernel_builder: &mut WgslKernelBuilder,
    ) -> Result<(), OperationError> {
        let op_name = self.op.kernel_name();

        let stride_condition = match self.op {
            ReduceOp::ArgMax => wgsl! {index < stride && smem[index + stride] > smem[index]},
            ReduceOp::ArgMin => wgsl! {index < stride && smem[index + stride] < smem[index]},
            _ => wgsl! {index < stride},
        };

        let smem_update = match self.op {
            ReduceOp::Sum | ReduceOp::Norm2 => wgsl! {
                smem[index] += smem[index + stride];
            },
            ReduceOp::ArgMax | ReduceOp::ArgMin => wgsl! {
                smem[index] = smem[index + stride];
            },
            ReduceOp::Max | ReduceOp::Min => wgsl! {
                smem[index] = 'op_name(smem[index], smem[index + stride]);
            },
        };

        let smem_index_update = match self.op {
            ReduceOp::ArgMax | ReduceOp::ArgMin => wgsl! {
                smem_index[index] = smem_index[index + stride];
            },
            _ => wgsl! {},
        };

        kernel_builder.write_global(wgsl! {
            fn block_reduce(index: u32, stride: u32) {
                workgroupBarrier();
                if 'stride_condition {
                    'smem_update
                    'smem_index_update
                }
            }
        });

        Ok(())
    }

    fn render_get_strided_index(
        &self,
        kernel_builder: &mut WgslKernelBuilder,
    ) -> Result<(), OperationError> {
        kernel_builder.write_global(wgsl! {
            fn get_strided_index(idx: u32, num_dims: u32, shape: vec4<u32>, stride: vec4<u32>) -> u32 {
                var strided_i: u32 = 0;
                var idx_: u32 = idx;
                for (var d: u32 = 0; d < num_dims; d++) {
                    var dim_idx: u32 = num_dims - 1 - d;
                    strided_i += (idx_ % shape[dim_idx]) * stride[dim_idx];
                    idx_ /= shape[dim_idx];
                }
                return strided_i;
            }
        });

        Ok(())
    }
}

#[derive(Debug, derive_new::new, ShaderType, WgslMetadata)]
pub struct ReduceMeta {
    src_numel: u32,
    num_dims: u32,
    num_reduce_dims: u32,
    el_to_reduce_per_block: u32,
    // Hard limit of summing along 4 dimensions; we might be able to improve this.
    shape: glam::UVec4,
    stride: glam::UVec4,
}

impl OpGuards for Reduce {
    fn check_shapes(&self) {
        let input = &self.input;
        let rank = input.dim();
        for &dim in self.reduce_dims.iter() {
            assert!(dim < rank);
        }
    }

    fn check_dtypes(&self) {
        assert!(self.reduce_dims.len() <= 4);
    }
}

impl Operation for Reduce {
    fn name(&self) -> &'static str {
        match self.op {
            ReduceOp::Sum => "Sum",
            ReduceOp::Min => "Min",
            ReduceOp::Max => "Max",
            ReduceOp::ArgMin => "Argmin",
            ReduceOp::ArgMax => "Argmax",
            ReduceOp::Norm2 => "Norm2",
        }
    }

    fn compute_view(&self) -> Result<StorageView, OperationError> {
        let mut output_shape_vec = self.reduced_shape.to_vec();

        if !self.keepdim {
            for &dim in self.reduce_dims.iter().rev() {
                output_shape_vec.remove(dim);
            }
        }

        let output_dtype = match self.op {
            ReduceOp::Sum | ReduceOp::Min | ReduceOp::Max | ReduceOp::Norm2 => DType::F32,
            ReduceOp::ArgMin | ReduceOp::ArgMax => DType::I32,
        };

        // This is a special case for the sum_all operation
        if output_shape_vec.is_empty() {
            output_shape_vec = vec![1];
        }

        let output_shape = Shape::from(output_shape_vec);
        let stride = Stride::from(&output_shape);
        Ok(StorageView::new(output_shape, output_dtype, stride))
    }

    #[inline]
    fn srcs(&self) -> RVec<&OpTensor> {
        rvec![&self.input]
    }

    fn supports_inplace(&self) -> bool {
        false
    }
}

pub enum ReduceKernels {
    Standard(Reduce),
}

impl GPUOperation for Reduce {
    type KernelEnum = ReduceKernels;

    fn select_kernel(&self) -> Self::KernelEnum {
        ReduceKernels::Standard(self.clone())
    }
}

impl Kernel for ReduceKernels {
    type Metadata = ReduceMeta;

    fn kernel_name(&self) -> String {
        match self {
            ReduceKernels::Standard(inner) => inner.op.kernel_name().to_string(),
        }
    }

    fn kernel_element(&self, _dst: &OpTensor) -> KernelElement {
        // let input = self.srcs()[0];
        // let rank = input.dim();
        // let shape = input.shape();
        // let mut min_N = 4;
        // for &dim in self.sum_dims.iter() {
        //     if dim + 1 < rank {
        //         let N = shape[dim + 1] as u32;
        //         if N % 4 == 0 {
        //             min_N = std::cmp::min(min_N, 4);
        //         } else if N % 2 == 0 {
        //             min_N = std::cmp::min(min_N, 2);
        //         } else {
        //             min_N = std::cmp::min(min_N, 1);
        //         }
        //     }
        // }
        // match min_N {
        //     4 => KernelElement::Vec4,
        //     2 => KernelElement::Vec2,
        //     _ => KernelElement::Scalar,
        // }
        KernelElement::Scalar
    }

    fn calculate_dispatch(&self, _dst: &OpTensor) -> Result<Workload, OperationError> {
        let ReduceKernels::Standard(inner) = self;
        // This one is a little tricky
        let sum_dim_size = if inner.reduce_dims.len() == inner.dims.len() {
            1
        } else {
            inner.dims[inner.dims.len() - (inner.reduce_dims.len() + 1)]
        };
        let x_count = inner.dst_numel / sum_dim_size;
        let y_count = sum_dim_size;
        Ok(Workload {
            workgroup_size: wgs![256, 1, 1],
            workgroup_count: wgc![x_count as u32, y_count as u32, 1],
        })
    }

    fn storage_bind_group_layout(
        &self,
        inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError> {
        if inplace {
            panic!("Only non-inplace sum is supported");
        }
        Ok(BindGroupLayoutDescriptor::unary())
    }

    fn metadata(&self, _: &OpTensor, _: &KernelElement) -> Result<Self::Metadata, OperationError> {
        let ReduceKernels::Standard(inner) = self;
        let mut shape = [0; 4];
        for (i, &dim) in inner.dims.iter().enumerate() {
            shape[i] = dim as u32;
        }
        let mut strides = [0; 4];
        for (i, &stride) in inner.stride.iter().enumerate() {
            strides[i] = stride as u32;
        }

        Ok(ReduceMeta {
            src_numel: inner.src_numel as u32,
            num_dims: inner.input.shape().len() as u32,
            num_reduce_dims: inner.reduce_dims.len() as u32,
            el_to_reduce_per_block: inner.el_to_reduce_per_block as u32,
            shape: shape.into(),
            stride: strides.into(),
        })
    }

    fn build_kernel(
        &self,
        inplace: bool,
        dst: &OpTensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let ReduceKernels::Standard(inner) = self;
        let kernel_element = self.kernel_element(dst);
        match (inner.input.dtype(), &kernel_element) {
            (DType::F32, KernelElement::Scalar) => {
                self.render::<Scalar<f32>>(inplace, dst, workgroup_size)
            }
            (DType::F16, KernelElement::Scalar) => {
                self.render::<Scalar<f16>>(inplace, dst, workgroup_size)
            }
            (DType::I32, KernelElement::Scalar) => {
                self.render::<Scalar<i32>>(inplace, dst, workgroup_size)
            }
            _ => Err(OperationError::CompileError(format!(
                "Unsupported dtype {:?} or kernel element {:?}",
                inner.input.dtype(),
                kernel_element
            ))),
        }
    }
}

impl KernelRenderable for ReduceKernels {
    fn register_bindings<P: WgslPrimitive>(
        &self,
        builder: &mut WgslKernelBuilder,
        _: bool,
    ) -> Result<(), OperationError> {
        let ReduceKernels::Standard(inner) = self;

        builder.register_storage("X", BindingMode::ReadOnly, Array::<P>::default());
        match inner.op {
            ReduceOp::ArgMax | ReduceOp::ArgMin => {
                builder.register_storage(
                    "Y",
                    BindingMode::ReadWrite,
                    Array::<Scalar<i32>>::default(),
                );
            }
            _ => {
                builder.register_storage("Y", BindingMode::ReadWrite, Array::<P>::default());
            }
        }
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
                BuiltIn::WorkgroupId,
                BuiltIn::LocalInvocationId,
                BuiltIn::NumWorkgroups
            ],
            device.compute_features().clone(),
        );

        self.register_bindings::<P>(&mut kernel_builder, inplace)?;
        kernel_builder.render_metadata(&self.metadata(dst, &self.kernel_element(dst))?);

        let ReduceKernels::Standard(inner) = self;

        let dtype = P::T::DT;
        let op = inner.op.kernel_name();

        kernel_builder.write_global(wgsl! {
            const BLOCK_SIZE: u32 = 256u;
            const maxFloat: f32 = 3.402823e+38f;
            var<workgroup> smem: array<'dtype, BLOCK_SIZE>; //max 16kb
        });

        match inner.op {
            ReduceOp::ArgMax | ReduceOp::ArgMin => {
                kernel_builder.write_global(wgsl! {
                    var<workgroup> smem_index: array<i32, BLOCK_SIZE>;
                });
            }
            _ => {}
        }

        inner.render_get_strided_index(&mut kernel_builder)?;
        inner.render_block_reduce(&mut kernel_builder)?;

        kernel_builder.write_main(wgsl! {
            let thread_id = local_invocation_id.x;
            let group_y = workgroup_id.y;

            let split_dim_idx = metadata.num_dims - metadata.num_reduce_dims - 1;
            let split_dim = metadata.shape[split_dim_idx];

            let split_len = num_workgroups.x;

            let destination_id = (group_y * split_len) + workgroup_id.x;
        });

        let smem_initialize = match inner.op {
            ReduceOp::Sum | ReduceOp::Norm2 => wgsl! {
                smem[thread_id] = 'dtype(0.0);
            },
            ReduceOp::Max | ReduceOp::ArgMax => wgsl! {
                smem[thread_id] = 'dtype(-maxFloat);
            },
            ReduceOp::Min | ReduceOp::ArgMin => wgsl! {
                smem[thread_id] = 'dtype(maxFloat);
            },
        };

        kernel_builder.write_main(smem_initialize);

        match inner.op {
            ReduceOp::ArgMax | ReduceOp::ArgMin => {
                kernel_builder.write_main(wgsl! {
                    smem_index[thread_id] = 0x7fffffffi;
                    var not_set = true;
                });
            }
            _ => {}
        }

        kernel_builder.write_main(wgsl! {
            let start_idx = destination_id * metadata.el_to_reduce_per_block;
            let stop_idx = min(start_idx + metadata.el_to_reduce_per_block, metadata.src_numel);
            var idx = start_idx + thread_id;
        });

        let smem_update = match inner.op {
            ReduceOp::Sum => wgsl! {
                smem[thread_id] += X[strided_i];
            },
            ReduceOp::Norm2 => wgsl! {
                smem[thread_id] += X[strided_i] * X[strided_i];
            },
            ReduceOp::Max | ReduceOp::Min => wgsl! {
                smem[thread_id] = 'op(smem[thread_id], X[strided_i]);
            },
            ReduceOp::ArgMax | ReduceOp::ArgMin => {
                let op = match inner.op {
                    ReduceOp::ArgMax => ">",
                    ReduceOp::ArgMin => "<",
                    _ => unreachable!(),
                };
                wgsl! {
                    if (not_set || X[strided_i] 'op smem[thread_id]) {
                        smem[thread_id] = X[strided_i];
                        smem_index[thread_id] = i32(idx % metadata.shape[metadata.num_dims - 1]);
                        not_set = false;
                    }
                }
            }
        };

        kernel_builder.write_main(wgsl! {
            while (idx < stop_idx) {
                let strided_i = get_strided_index(idx, metadata.num_dims, metadata.shape, metadata.stride);
                'smem_update
                idx += BLOCK_SIZE;
            }
        });

        kernel_builder.write_main(wgsl! {
            block_reduce(thread_id, 128u);
            block_reduce(thread_id, 64u);
            block_reduce(thread_id, 32u);
            block_reduce(thread_id, 16u);
            block_reduce(thread_id, 8u);
            block_reduce(thread_id, 4u);
            block_reduce(thread_id, 2u);
            block_reduce(thread_id, 1u);
        });

        let dst_update = match inner.op {
            ReduceOp::ArgMax | ReduceOp::ArgMin => wgsl! {
                Y[destination_id] = smem_index[0];
            },
            ReduceOp::Norm2 => wgsl! {
                Y[destination_id] = sqrt(smem[0]);
            },
            _ => wgsl! {
                Y[destination_id] = smem[0];
            },
        };

        kernel_builder.write_main(wgsl! {
            if thread_id == 0u {
                'dst_update
            }
        });

        Ok(kernel_builder.build()?)
    }
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use test_strategy::{proptest, Arbitrary};

    use crate::test_util::run_py_prg;
    use crate::{DType, Device, DeviceRequest, NormOrd, OpTensor};

    use super::ReduceOp;

    fn ground_truth_forward(
        a: &OpTensor,
        op: &ReduceOp,
        dim: Option<usize>,
    ) -> anyhow::Result<OpTensor> {
        let dim_str = match dim {
            Some(d) => format!(", dim={d}"),
            None => "".to_string(),
        };
        let prg = match op {
            ReduceOp::Max | ReduceOp::Min => format!(
                r#"
import torch
def reduce(a):
    return torch.{}(torch.from_numpy(a){}).values.float().numpy()
"#,
                op.kernel_name(),
                dim_str
            ),
            _ => format!(
                r#"
import torch
def reduce(a):
    return torch.{}(torch.from_numpy(a){}).float().numpy()
"#,
                op.kernel_name(),
                dim_str
            ),
        };
        // let out_dtype = match op {
        //     ReduceOp::ArgMax | ReduceOp::ArgMin => DType::I32,
        //     _ => DType::F32,
        // };
        run_py_prg(prg.to_string(), &[a], &[], DType::F32)
    }

    fn run_reduce_forward_trial(
        B: usize,
        M: usize,
        N: usize,
        op: &ReduceOp,
        dim: Option<usize>,
        device: Device,
    ) -> anyhow::Result<()> {
        let a = OpTensor::randn::<f32, _>(0., 1., (B, M, N), Device::CPU, false)?;
        let mut ground = ground_truth_forward(&a, op, dim)?;

        if dim.is_none() {
            ground = ground.view(1)?;
        }

        let a_gpu = a.to(&device)?;
        let b_gpu = match dim {
            Some(dim) => match op {
                ReduceOp::Sum => a_gpu.sum(dim),
                ReduceOp::Min => a_gpu.min(dim),
                ReduceOp::Max => a_gpu.max(dim),
                ReduceOp::ArgMin => a_gpu.argmin(dim),
                ReduceOp::ArgMax => a_gpu.argmax(dim),
                ReduceOp::Norm2 => a_gpu.norm_ord_dim(NormOrd::Frobenius, dim),
            },
            None => match op {
                ReduceOp::Sum => a_gpu.sum_all(),
                ReduceOp::Norm2 => a_gpu.norm(),
                _ => panic!("All * not supported"),
            },
        }?;

        let ours = b_gpu.cast(DType::F32)?.to(&Device::CPU)?;
        // println!("input = {:?}", a);
        // println!("input stride = {:?}", a.stride());
        // println!("ours = {:?}", ours);
        // println!("ground = {:?}", ground);
        ground.all_close(&ours, 3e-5, 1e-5)?;
        Ok(())
    }

    #[derive(Arbitrary, Debug)]
    struct ReduceProblem {
        op: ReduceOp,
        #[strategy(1..=3usize)]
        B: usize,
        #[strategy(1..=256usize)]
        M: usize,
        #[strategy(1..=256usize)]
        N: usize,
        #[strategy(0..=2usize)]
        dim: usize,
    }

    #[proptest(cases = 256)]
    fn test_reduce(prob: ReduceProblem) {
        let ReduceProblem {
            B,
            M,
            N,
            ref op,
            dim,
        } = prob;
        println!(
            "B = {}, M = {}, N = {}, op = {}, dim = {}",
            B,
            M,
            N,
            op.kernel_name(),
            dim
        );
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        run_reduce_forward_trial(B, M, N, op, Some(dim), device).unwrap();
    }

    #[derive(Arbitrary, Debug)]
    struct SumAllProblem {
        #[strategy(1..=3usize)]
        B: usize,
        #[strategy(1..=256usize)]
        M: usize,
        #[strategy(1..=256usize)]
        N: usize,
    }

    #[proptest(cases = 16)]
    fn test_sum_all(prob: SumAllProblem) {
        let SumAllProblem { B, M, N } = prob;
        println!("B = {B}, M = {M}, N = {N}");
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        run_reduce_forward_trial(B, M, N, &ReduceOp::Sum, None, device).unwrap();
    }

    #[derive(Arbitrary, Debug)]
    struct ReduceBackwardProblem {
        #[strategy(1..=3usize)]
        B: usize,
        #[strategy(1..=64usize)]
        M: usize,
        #[strategy(1..=64usize)]
        N: usize,
    }

    fn ground_truth_backward(a: &OpTensor) -> anyhow::Result<OpTensor> {
        let prg = r#"
import torch
def reduce_backward(a):
    a_tensor = torch.tensor(torch.from_numpy(a), requires_grad=True)
    result = torch.sum(a_tensor)
    result.backward(torch.ones_like(result))
    return a_tensor.grad.numpy()
    "#
        .to_string();
        run_py_prg(prg.to_string(), &[a], &[], DType::F32)
    }

    fn run_sum_backward_trial(
        problem: ReduceBackwardProblem,
        device: Device,
    ) -> anyhow::Result<()> {
        let ReduceBackwardProblem { B, M, N } = problem;
        let gpu_device = device.try_gpu()?;
        let a = OpTensor::randn::<f32, _>(0., 1., (B, M, N), Device::CPU, false)?;
        let ground = ground_truth_backward(&a)?;

        let a_gpu = a.to(&device)?.requires_grad_(true)?;
        let b_gpu = a_gpu.clone().sum_all()?;

        b_gpu.backward()?;
        gpu_device.mark_step()?;
        let a_grad = a_gpu.grad().unwrap().clone();

        let ours = a_grad.to(&Device::CPU)?;
        println!("ours = {ours:?}");
        println!("ground = {ground:?}");
        ground.all_close(&ours, 1e-5, 1e-5)?;
        Ok(())
    }

    #[proptest(cases = 8)]
    fn test_reduce_backward(prob: ReduceBackwardProblem) {
        let ReduceBackwardProblem { B, M, N } = prob;
        println!("B = {B}, M = {M}, N = {N}");
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        run_sum_backward_trial(prob, device).unwrap();
    }
}
