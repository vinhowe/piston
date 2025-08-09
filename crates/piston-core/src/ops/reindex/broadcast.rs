use derive_new::new;
use encase::ShaderType;
use piston_macros::WgslMetadata;

use crate::{
    OpGuards, OpTensor, Operation, OperationError, RVec, Shape, StorageView, Stride, rvec,
};
use piston_macros::IrFields;

#[derive(Debug, WgslMetadata, ShaderType, derive_new::new)]
pub struct BroadcastMeta {
    src_shape: glam::UVec4,
    dst_shape: glam::UVec4,
    src_stride: glam::UVec4,
    dst_stride: glam::UVec4,
    src_numel: u32,
    dst_numel: u32,
}

#[derive(new, Debug, Clone, IrFields)]
pub struct Broadcast {
    pub src: OpTensor,
    to: Shape,
}

impl Broadcast {
    pub fn to(&self) -> &Shape {
        &self.to
    }
}

impl OpGuards for Broadcast {
    //TODO: check the broadcast is valid
    fn check_shapes(&self) {
        let src_shape = self.src.shape();
        let to_shape = &self.to;

        let sr = src_shape.dim();
        let dr = to_shape.dim();
        if sr > dr {
            panic!("Source shape cannot have more dimensions than target shape: {sr} > {dr}");
        }

        let src_iter = src_shape.iter().rev();
        let to_iter = to_shape.iter().rev();

        for (src_dim, to_dim) in src_iter.zip(to_iter) {
            if *src_dim != 1 && *src_dim != *to_dim {
                panic!(
                    "Invalid broadcast: source dimension {src_dim} cannot be broadcast to {to_dim}"
                );
            }
        }
    }

    fn check_dtypes(&self) {
        assert!(!self.src.dtype().is_quantized());
    }
}

impl Operation for Broadcast {
    fn name(&self) -> &'static str {
        "Broadcast"
    }

    // For rules, see https://numpy.org/doc/stable/user/basics.broadcasting.html
    fn compute_view(&self) -> Result<StorageView, OperationError> {
        let src_shape = self.src.shape();

        if *src_shape == self.to {
            return Ok(self.src.storage_view().clone());
        }

        let stride = Stride::from(&self.to);
        Ok(StorageView::new(self.to.clone(), self.src.dtype(), stride))
    }

    fn srcs(&self) -> RVec<&OpTensor> {
        rvec![&self.src]
    }
}

#[cfg(all(test, feature = "pyo3"))]
mod tests {
    use proptest::{
        arbitrary::Arbitrary,
        strategy::{BoxedStrategy, Just, Strategy},
    };
    use test_strategy::proptest;

    use crate::{
        Broadcast, Device, DeviceRequest, Shape, Tensor, randn, shape, test_util::run_py_prg,
    };

    impl Arbitrary for BroadcastProblem {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_args: ()) -> Self::Strategy {
            Shape::arbitrary_with(vec![1..=2, 1..=8, 1..=2, 1..=128])
                .prop_flat_map(|original_shape| {
                    let create_broadcast_range = |dim: usize| {
                        if original_shape[dim] == 1 {
                            1..=8
                        } else {
                            original_shape[dim]..=original_shape[dim]
                        }
                    };

                    let to = Shape::arbitrary_with(vec![
                        create_broadcast_range(0),
                        create_broadcast_range(1),
                        create_broadcast_range(2),
                        create_broadcast_range(3),
                    ]);
                    (Just(original_shape), to)
                })
                .prop_map(|(original_shape, to)| BroadcastProblem {
                    op: Broadcast::new(
                        randn(original_shape, None, None, Default::default())
                            .unwrap()
                            .inner_or_source()
                            .clone(),
                        to,
                    ),
                })
                .boxed()
        }
    }

    #[derive(Debug, Clone)]
    struct BroadcastProblem {
        op: Broadcast,
    }

    fn ground_truth(a: &Tensor, args: &str) -> anyhow::Result<Tensor> {
        let prg = format!(
            r#"
import torch
import numpy as np
def slice(a):
    torch_a = torch.from_numpy(a)
    return np.ascontiguousarray(torch_a.broadcast_to({args}).numpy())
"#,
        );
        run_py_prg(prg.to_string(), &[a], &[], a.dtype())
    }

    fn run_reindex_trial(prob: BroadcastProblem, device: Device) -> anyhow::Result<()> {
        println!("\n\nBroadcast problem: {prob:?}");
        let BroadcastProblem { op } = prob;
        let a = op.src.wrap();

        let a_gpu = a.to(&device)?;
        let ground = ground_truth(&a, &op.to.as_torch())?;
        let ours = a_gpu.broadcast_to(op.to.clone())?;
        let d_gpu = ours.to(&Device::CPU)?;
        ground.all_close(&d_gpu, 1e-5, 1e-5)?;
        Ok(())
    }

    #[proptest(cases = 16)]
    fn test_broadcast_gpu(prob: BroadcastProblem) {
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        run_reindex_trial(prob, device).unwrap();
    }

    #[proptest(cases = 16)]
    fn test_broadcast_cpu(prob: BroadcastProblem) {
        let device = Device::request_device(DeviceRequest::CPU).unwrap();
        run_reindex_trial(prob, device).unwrap();
    }

    #[test]
    fn debug_broadcast() {
        let device = Device::request_device(DeviceRequest::GPU).unwrap();
        let prob = BroadcastProblem {
            op: Broadcast::new(
                randn(1, None, None, Default::default())
                    .unwrap()
                    .inner_or_source()
                    .clone(),
                shape![4, 32, 128, 128],
            ),
        };
        run_reindex_trial(prob, device).unwrap();
    }
}
