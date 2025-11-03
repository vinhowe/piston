use derive_new::new;
use encase::ShaderType;
use half::f16;
use piston_macros::{IrFields, WgslMetadata};

use crate::gpu::dtype::WgslDType;
use crate::{
    Array, BindingMode, BuiltIn, DType, KernelElement, KernelSource, OpGuards, OpTensor, Operation,
    OperationError, RVec, Scalar, StorageView, Stride, Vec2, Vec4, WgslKernelBuilder,
    WgslPrimitive, WorkgroupSize, Workload,
    gpu::{BindGroupLayoutDescriptor, WorkgroupCount},
    rvec, wgc, wgs,
};
use crate::{GPUOperation, Kernel, KernelRenderable};
use inline_wgsl::wgsl;

#[derive(new, Debug, Clone, IrFields)]
pub struct RoPE {
    pub(crate) input: OpTensor,
    pub(crate) dim: usize,
    pub(crate) base: f32,
    pub(crate) offset: usize,
    // Doing things this way, with a backward field in an op, is a little bit inelegant, but it's
    // so nicely consolidated
    pub(crate) backward: bool,
}

impl RoPE {
    pub fn input(&self) -> &OpTensor {
        &self.input
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn base(&self) -> f32 {
        self.base
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn backward(&self) -> bool {
        self.backward
    }
}

#[derive(Debug, derive_new::new, ShaderType, WgslMetadata)]
pub struct RoPEMeta {
    in_stride: glam::UVec3,
    out_stride: glam::UVec3,
    seq_len: u32,
    offset: u32,
    base: f32,
    scale: f32,
}

impl OpGuards for RoPE {
    fn check_shapes(&self) {
        let input = &self.input;
        //TODO: overly restrictive
        assert!(input.dim() == 4);
        assert!(input.shape()[3] >= self.dim);
        assert!(self.dim.is_multiple_of(8));
    }

    fn check_dtypes(&self) {
        let input = &self.input;
        assert!(input.dtype().is_float());
    }
}

impl Operation for RoPE {
    fn name(&self) -> &'static str {
        "RoPE"
    }

    fn compute_view(&self) -> Result<StorageView, OperationError> {
        Ok(self.input.storage_view().clone())
    }

    #[inline]
    fn srcs(&self) -> RVec<&OpTensor> {
        rvec![&self.input]
    }

    fn supports_inplace(&self) -> bool {
        true
    }
}

impl GPUOperation for RoPE {
    type KernelEnum = RoPEKernels;

    fn select_kernel(&self) -> Self::KernelEnum {
        if self.backward {
            RoPEKernels::Backward(self.clone())
        } else {
            RoPEKernels::Forward(self.clone())
        }
    }
}

pub enum RoPEKernels {
    Forward(RoPE),
    Backward(RoPE),
}

fn rope_body(is_backward: bool) -> String {
    if is_backward {
        wgsl! {
            let rx1 = x1 * costheta + x2 * sintheta;
            let rx2 = -x1 * sintheta + x2 * costheta;
        }
    } else {
        wgsl! {
            let rx1 = x1 * costheta - x2 * sintheta;
            let rx2 = x1 * sintheta + x2 * costheta;
        }
    }
}

impl KernelRenderable for RoPEKernels {
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
        dst: &OpTensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let device = dst.device().try_gpu().unwrap();
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

        let (is_backward, _inner) = match self {
            RoPEKernels::Forward(x) => (false, x),
            RoPEKernels::Backward(x) => (true, x),
        };

        kernel_builder.render_metadata(&self.metadata(dst, &self.kernel_element(dst))?);

        let body_code = rope_body(is_backward);

        let dtype = P::T::DT;
        let write_operations = if inplace {
            wgsl! {
                in[out_index_1] = rx1;
                in[out_index_2] = rx2;
            }
        } else {
            wgsl! {
                out[out_index_1] = rx1;
                out[out_index_2] = rx2;
            }
        };

        kernel_builder.write_main(wgsl! {
            if(global_invocation_id.y >= metadata.seq_len) {
              return;
            }

            let grid = vec3<u32>(num_workgroups.x * 8u, num_workgroups.y * 8u, num_workgroups.z * 1u);

            let out_index_1 = dot(global_invocation_id, vec3<u32>(metadata.out_stride[2], metadata.out_stride[1], metadata.out_stride[0]));
            let out_index_2 = out_index_1 + grid.x * metadata.out_stride[2];

            let in_index_1 = dot(global_invocation_id, vec3<u32>(metadata.in_stride[2], metadata.in_stride[1], metadata.in_stride[0]));
            let in_index_2 = in_index_1 + grid.x * metadata.in_stride[2];

            let L = metadata.scale * f32(global_invocation_id.y + metadata.offset);
            let d = f32(global_invocation_id.x) / f32(grid.x);

            let theta = L * exp2(-d * metadata.base);
            let costheta = 'dtype(cos(theta));
            let sintheta = 'dtype(sin(theta));

            let x1 = in[in_index_1];
            let x2 = in[in_index_2];

            'body_code

            'write_operations
        });

        Ok(kernel_builder.build()?)
    }
}

impl Kernel for RoPEKernels {
    type Metadata = RoPEMeta;

    fn kernel_name(&self) -> String {
        match self {
            RoPEKernels::Forward(_) => "rope_forward".to_string(),
            RoPEKernels::Backward(_) => "rope_backward".to_string(),
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

    fn metadata(
        &self,
        dst: &OpTensor,
        _: &KernelElement,
    ) -> Result<Self::Metadata, OperationError> {
        let inner = match self {
            RoPEKernels::Forward(x) => x,
            RoPEKernels::Backward(x) => x,
        };

        let mut input_shape = inner.input.shape().clone();
        let SL = input_shape[2];
        let mut out_shape = dst.shape().clone();
        input_shape.remove(0);
        out_shape.remove(0);
        let in_stride = Stride::from(&input_shape);
        let out_stride = Stride::from(&out_shape);
        Ok(RoPEMeta::new(
            (&in_stride).into(),
            (&out_stride).into(),
            SL as u32,
            inner.offset as u32,
            f32::log2(inner.base),
            1.0,
        ))
    }

    fn kernel_element(&self, _dst: &OpTensor) -> KernelElement {
        KernelElement::Scalar
    }

    fn calculate_dispatch(&self, _dst: &OpTensor) -> Result<Workload, OperationError> {
        const WGSX: usize = 8;
        const WGSY: usize = 8;
        const WGSZ: usize = 1;
        let workgroup_size = wgs![WGSX as _, WGSY as _, WGSZ as _];

        let inner = match self {
            RoPEKernels::Forward(x) => x,
            RoPEKernels::Backward(x) => x,
        };
        let [_, _, SL, HD]: [usize; 4] = inner.input.shape().try_into()?;
        let mat_size = SL * HD;

        let total_x = inner.dim / 2; // solve pairs
        let total_y = SL;
        let total_z = inner.input.shape().numel() / mat_size;

        let wgcx = WorkgroupCount::div_ceil(total_x, WGSX) as u32;
        let wgcy = WorkgroupCount::div_ceil(total_y, WGSY) as u32;
        let wgcz = WorkgroupCount::div_ceil(total_z, WGSZ) as u32;

        Ok(Workload {
            workgroup_count: wgc![wgcx, wgcy, wgcz],
            workgroup_size,
        })
    }

    fn build_kernel(
        &self,
        inplace: bool,
        dst: &OpTensor,
        workgroup_size: &WorkgroupSize,
    ) -> Result<KernelSource, OperationError> {
        let kernel_element = self.kernel_element(dst);
        let inner = match self {
            RoPEKernels::Forward(x) => x,
            RoPEKernels::Backward(x) => x,
        };
        match (inner.input.dtype(), &kernel_element) {
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
                inner.input.dtype(),
                kernel_element
            ))),
        }
    }
}

#[cfg(all(test, feature = "pyo3", target_os = "macos"))]
mod tests {
    use test_strategy::{Arbitrary, proptest};

    use crate::test_util::run_py_prg;
    use crate::{Device, DeviceRequest, Tensor, randn};

    fn ground_truth(a: &Tensor, dim: usize, offset: usize) -> anyhow::Result<Tensor> {
        let prg = r#"
import mlx.core as mx
import mlx.nn as nn
import numpy as np

def mlx_rope(input, dim, offset):
    rope = nn.RoPE(dim)
    mx_input = mx.array(input)
    y = rope(mx_input, offset)
    mx.eval(y)
    return np.array(y)
"#;
        run_py_prg(prg.to_string(), &[a], &[&dim, &offset], a.dtype())
    }

    fn run_rope_trial(problem: RoPEProblem, device: Device) {
        let RoPEProblem {
            BS,
            NH,
            SL,
            HD,
            dim,
            offset,
        } = problem;
        let shape = (BS, NH, SL, HD);
        let a = randn(shape, None, None, Default::default()).unwrap();
        let ground = ground_truth(&a, dim, offset).unwrap();

        let a = a.to(&device).unwrap();
        let b = a.rope_(dim, 10000.0, offset).unwrap();

        let ours = b.to(&Device::CPU).unwrap();
        //println!("ours = \n{:#?}\n", ours.to_ndarray_view::<f32>());
        //println!("ground = \n{:#?}", ground.to_ndarray_view::<f32>());
        //Weak tolerance because of `ffast-math`
        ground.all_close(&ours, 1e-2, 1e-2).unwrap();
    }

    #[derive(Arbitrary, Debug)]
    struct RoPEProblem {
        #[strategy(1..=2usize)]
        BS: usize,
        #[strategy(1..=64usize)]
        NH: usize,
        #[strategy(1..=256usize)]
        SL: usize,
        #[strategy(32..=128usize)]
        #[filter(#HD.is_multiple_of(16))]
        HD: usize,
        #[strategy(32..=#HD)]
        #[filter(#dim.is_multiple_of(32))]
        dim: usize,
        #[strategy(0..=#SL)]
        offset: usize,
    }

    #[proptest(cases = 16)]
    fn test_rope_gpu(prob: RoPEProblem) {
        let RoPEProblem {
            BS,
            NH,
            SL,
            HD,
            dim,
            offset,
        } = prob;
        println!("BS = {BS}, NH = {NH}, SL = {SL}, HD = {HD}, rope_dim = {dim}, offset = {offset}");

        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        run_rope_trial(prob, device);
    }

    #[proptest(cases = 16)]
    fn test_rope_cpu(prob: RoPEProblem) {
        let RoPEProblem {
            BS,
            NH,
            SL,
            HD,
            dim,
            offset,
        } = prob;
        println!("BS = {BS}, NH = {NH}, SL = {SL}, HD = {HD}, rope_dim = {dim}, offset = {offset}");

        let device = Device::request_device(DeviceRequest::CPU).unwrap();
        run_rope_trial(prob, device);
    }

    #[test]
    fn debug_rope_cpu() {
        let prob = RoPEProblem {
            BS: 1,
            NH: 5,
            SL: 180,
            HD: 112,
            dim: 96,
            offset: 141,
        };
        println!("{prob:?}");

        let device = Device::request_device(DeviceRequest::CPU).unwrap();
        run_rope_trial(prob, device);
    }
}
