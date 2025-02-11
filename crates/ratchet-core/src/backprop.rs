/// Adapted from candle:
/// https://github.com/huggingface/candle/blob/main/candle-core/src/backprop.rs
/// Methods for backpropagation of gradients.
use crate::ops::{BinaryOp, UnaryOp};
use crate::{
    rvec, Affine, Binary, Broadcast, Cmp, Concat, Conv, DType, Gather, GroupNorm, IndexAdd,
    IndexSelect, LazyOp, Matmul, Norm, NormOp, Permute, Powf, Reduce, ReduceOp, Reindex,
    ScatterAdd, Shape, Slice, Softmax, Tensor, TensorId, Unary, View, WhereCond,
};
use crate::{HashMap, Trilu};
use anyhow::Result;
use std::collections::hash_map::Entry;

// thiserror error for Tensor
#[derive(thiserror::Error, Debug)]
pub enum BackpropError {
    #[error("Tensor is not resolved")]
    BackwardNotSupported { op: &'static str },
}

// arg has been reduced to node via reduce_dims, expand it back to arg.
// This has to handle keepdims.
fn broadcast_back(arg: &Tensor, node: &Tensor, reduced_dims: &[usize]) -> Result<Tensor> {
    if arg.rank() == node.rank() {
        // keepdim = true
        node.clone().broadcast_to(arg.shape().clone())
    } else {
        // keepdim = false
        node.clone()
            .view(reduced_dims.into())?
            .broadcast_to(arg.shape().clone())
    }
}

thread_local! {
    static RATCHET_GRAD_DO_NOT_DETACH: bool = {
        match std::env::var("RATCHET_GRAD_DO_NOT_DETACH") {
            Ok(s) => {
                !s.is_empty() && s != "0"
            },
            Err(_) => false,
        }
    }
}

impl Tensor {
    /// Return all the nodes that lead to this value in a topologically sorted vec, the first
    /// elements having dependencies on the latter ones, e.g. the first element if any is the
    /// argument.
    /// This assumes that the op graph is a DAG.
    // TODO(vinhowe): This could be consolidated with execution_order and whatever caching we
    // do...
    fn sorted_nodes(&self) -> Vec<&Tensor> {
        // The vec of sorted nodes is passed as an owned value rather than a mutable reference
        // to get around some lifetime limitations.
        fn walk<'a>(
            node: &'a Tensor,
            nodes: Vec<&'a Tensor>,
            already_seen: &mut HashMap<TensorId, bool>,
        ) -> (bool, Vec<&'a Tensor>) {
            if let Some(&tg) = already_seen.get(&node.id()) {
                return (tg, nodes);
            }
            let mut track_grad = false;
            let mut nodes = if node.is_variable() {
                // Do not call recursively on the "leaf" nodes.
                track_grad = true;
                nodes
            } else if matches!(node.dt(), DType::I32 | DType::U32) {
                nodes
            } else {
                match node.op() {
                    LazyOp::IndexAdd(IndexAdd {
                        dst: t1,
                        src: t2,
                        ids: t3,
                        ..
                    })
                    | LazyOp::ScatterAdd(ScatterAdd {
                        dst: t1,
                        src: t2,
                        ids: t3,
                        ..
                    })
                    | LazyOp::WhereCond(WhereCond {
                        input: t1,
                        on_true: t2,
                        on_false: t3,
                    }) => {
                        let (tg, nodes) = walk(t1, nodes, already_seen);
                        track_grad |= tg;
                        let (tg, nodes) = walk(t2, nodes, already_seen);
                        track_grad |= tg;
                        let (tg, nodes) = walk(t3, nodes, already_seen);
                        track_grad |= tg;
                        nodes
                    }
                    LazyOp::Conv(Conv {
                        input: lhs,
                        weight: rhs,
                        ..
                    })
                    | LazyOp::Binary(Binary { lhs, rhs, .. })
                    | LazyOp::Gather(Gather {
                        src: lhs, ids: rhs, ..
                    })
                    | LazyOp::Select(IndexSelect {
                        src: lhs,
                        indices: rhs,
                        ..
                    })
                    | LazyOp::Matmul(Matmul { lhs, rhs, .. }) => {
                        let (tg, nodes) = walk(lhs, nodes, already_seen);
                        track_grad |= tg;
                        let (tg, nodes) = walk(rhs, nodes, already_seen);
                        track_grad |= tg;
                        nodes
                    }
                    LazyOp::Concat(Concat { inputs, .. }) => {
                        inputs.iter().fold(nodes, |nodes, input| {
                            let (tg, nodes) = walk(input, nodes, already_seen);
                            track_grad |= tg;
                            nodes
                        })
                    }
                    LazyOp::Affine(Affine {
                        src: input, mul, ..
                    }) => {
                        if *mul == 0. {
                            nodes
                        } else {
                            let (tg, nodes) = walk(input, nodes, already_seen);
                            track_grad |= tg;
                            nodes
                        }
                    }
                    LazyOp::Unary(Unary {
                        input: _node,
                        op: UnaryOp::Ceil,
                    })
                    | LazyOp::Unary(Unary {
                        input: _node,
                        op: UnaryOp::Floor,
                    }) => nodes,
                    LazyOp::Cmp(Cmp { lhs: node, .. })
                    | LazyOp::Unary(Unary { input: node, .. })
                    | LazyOp::Reduce(Reduce {
                        input: node,
                        op: ReduceOp::Min | ReduceOp::Sum | ReduceOp::Max,
                        ..
                    })
                    | LazyOp::Reindex(Reindex::Permute(Permute { src: node, .. }))
                    | LazyOp::Reindex(Reindex::Broadcast(Broadcast { src: node, .. }))
                    | LazyOp::Reindex(Reindex::Slice(Slice { src: node, .. }))
                    | LazyOp::Softmax(Softmax { input: node, .. })
                    | LazyOp::Powf(Powf { src: node, .. }) => {
                        let (tg, nodes) = walk(node, nodes, already_seen);
                        track_grad |= tg;
                        nodes
                    }
                    LazyOp::View(View { src: node, .. }) => {
                        let (tg, nodes) = walk(node, nodes, already_seen);
                        track_grad |= tg;
                        nodes
                    }
                    LazyOp::Norm(NormOp::RMSNorm(Norm { input: node, .. }))
                    | LazyOp::Norm(NormOp::LayerNorm(Norm { input: node, .. }))
                    | LazyOp::Norm(NormOp::GroupNorm(GroupNorm {
                        norm: Norm { input: node, .. },
                        ..
                    })) => {
                        let (tg, nodes) = walk(node, nodes, already_seen);
                        track_grad |= tg;
                        nodes
                    }
                    LazyOp::IndexWrite(_) => todo!(),
                    LazyOp::Cast(_) => todo!(),
                    LazyOp::Copy(_) => todo!(),
                    LazyOp::Detach(_)
                    | LazyOp::Const
                    | LazyOp::RoPE(_)
                    | LazyOp::Reduce(Reduce {
                        op: ReduceOp::ArgMax | ReduceOp::ArgMin,
                        ..
                    })
                    | LazyOp::FillConstant(_)
                    | LazyOp::FillRandn(_)
                    | LazyOp::Arange(_)
                    | LazyOp::Cache(_)
                    | LazyOp::Trilu(_) => nodes,
                }
            };
            already_seen.insert(node.id(), track_grad);
            if track_grad {
                nodes.push(node);
            }
            (track_grad, nodes)
        }
        let (_tg, mut nodes) = walk(self, vec![], &mut HashMap::default());
        nodes.reverse();
        nodes
    }

    pub fn backward(&self) -> Result<GradStore> {
        let sorted_nodes = self.sorted_nodes();
        let mut grads = GradStore::new();
        grads.insert(self, self.ones_like::<f32>().contiguous());
        for node in sorted_nodes.iter() {
            if node.is_variable() {
                continue;
            }
            log::debug!("Backwarding: {:?}", node.op().name());
            let grad = grads
                .remove(node)
                .expect("ratchet internal error - grad not populated");
            // From candle:
            // https://github.com/huggingface/candle/issues/1241
            // Ideally, we would make these operations in place where possible to ensure that we
            // do not have to allocate too often. Here we just call `.detach` to avoid computing
            // the backprop graph of the backprop itself. This would be an issue for second order
            // derivatives but these are out of scope at the moment.
            let do_not_detach = RATCHET_GRAD_DO_NOT_DETACH.with(|b| *b);
            let grad = if do_not_detach { grad } else { grad.detach() };
            match node.op() {
                LazyOp::Binary(Binary {
                    lhs,
                    rhs,
                    op: BinaryOp::Add,
                }) => {
                    let lhs_sum_grad = grads.or_insert(lhs.clone())?;
                    *lhs_sum_grad = lhs_sum_grad.clone().add(grad.clone())?;
                    let rhs_sum_grad = grads.or_insert(rhs.clone())?;
                    *rhs_sum_grad = rhs_sum_grad.clone().add(grad)?;
                }
                LazyOp::Binary(Binary {
                    lhs,
                    rhs,
                    op: BinaryOp::Sub,
                }) => {
                    let lhs_sum_grad = grads.or_insert(lhs.clone())?;
                    *lhs_sum_grad = lhs_sum_grad.clone().add(grad.clone())?;
                    let rhs_sum_grad = grads.or_insert(rhs.clone())?;
                    *rhs_sum_grad = rhs_sum_grad.clone().sub(grad)?;
                }
                LazyOp::Binary(Binary {
                    lhs,
                    rhs,
                    op: BinaryOp::Mul,
                }) => {
                    let lhs_grad = grad.clone().mul(rhs.clone())?;
                    let lhs_sum_grad = grads.or_insert(lhs.clone())?;
                    *lhs_sum_grad = lhs_sum_grad.clone().add(lhs_grad.clone())?;
                    let rhs_grad = grad.mul(lhs.clone())?;
                    let rhs_sum_grad = grads.or_insert(rhs.clone())?;
                    *rhs_sum_grad = rhs_sum_grad.clone().add(rhs_grad.clone())?;
                }
                LazyOp::Binary(Binary {
                    lhs,
                    rhs,
                    op: BinaryOp::Div,
                }) => {
                    let lhs_grad = grad.clone().div(rhs.clone())?;
                    let lhs_sum_grad = grads.or_insert(lhs.clone())?;
                    *lhs_sum_grad = lhs_sum_grad.clone().add(lhs_grad.clone())?;
                    let rhs_grad = grad.mul(lhs.clone())?.div(rhs.clone().square()?)?;
                    let rhs_sum_grad = grads.or_insert(rhs.clone())?;
                    *rhs_sum_grad = rhs_sum_grad.clone().sub(rhs_grad.clone())?;
                }
                LazyOp::WhereCond(WhereCond {
                    input,
                    on_true,
                    on_false,
                }) => {
                    let zeros = grad.clone().zeros_like::<f32>();
                    let t_sum_grad = grads.or_insert(on_true.clone())?;
                    let t_grad = input.clone().where_cond(grad.clone(), zeros.clone())?;
                    *t_sum_grad = t_sum_grad.clone().add(t_grad)?;
                    let f_sum_grad = grads.or_insert(on_false.clone())?;
                    let f_grad = input.clone().where_cond(zeros, grad)?;
                    *f_sum_grad = f_sum_grad.clone().add(f_grad)?;
                }
                LazyOp::Matmul(Matmul {
                    lhs,
                    rhs,
                    trans_lhs,
                    trans_rhs,
                    trans_dst,
                    bias,
                }) => {
                    let lhs_grad =
                        grad.clone()
                            .gemm(rhs.clone(), None, *trans_dst, !trans_rhs, *trans_lhs)?;
                    let lhs_sum_grad = grads.or_insert(lhs.clone())?;
                    *lhs_sum_grad = lhs_sum_grad.clone().add(lhs_grad.clone())?;

                    let rhs_grad =
                        lhs.clone()
                            .gemm(grad.clone(), None, !trans_lhs, *trans_dst, *trans_rhs)?;
                    let rhs_sum_grad = grads.or_insert(rhs.clone())?;
                    *rhs_sum_grad = rhs_sum_grad.clone().add(rhs_grad.clone())?;

                    // Calculate the gradient with respect to the bias term
                    if let Some(bias) = bias {
                        let bias_grad = grad.sum_keepdim(&[0])?; // Assuming bias is summed over the appropriate axis
                        let bias_sum_grad = grads.or_insert(bias.clone())?;
                        *bias_sum_grad = bias_sum_grad.clone().add(bias_grad.clone())?;
                    }
                }
                LazyOp::Reindex(Reindex::Broadcast(Broadcast { src, .. })) => {
                    let arg_dims = src.shape().to_vec();
                    let node_dims = node.shape().to_vec();

                    let left_dims = node_dims.len() - arg_dims.len();
                    let mut sum_dims: Vec<usize> = (0..left_dims).collect();
                    for (dim, (node_dim, arg_dim)) in node_dims[left_dims..]
                        .iter()
                        .zip(arg_dims.iter())
                        .enumerate()
                    {
                        if node_dim != arg_dim {
                            sum_dims.push(dim + left_dims);
                        }
                    }

                    let mut arg_grad = grad.sum_keepdim(sum_dims.as_slice())?;
                    for _i in 0..left_dims {
                        arg_grad = arg_grad.squeeze()?;
                    }
                    let sum_grad = grads.or_insert(src.clone())?;
                    *sum_grad = sum_grad
                        .clone()
                        .add(arg_grad.broadcast_to(sum_grad.shape().clone())?)?;
                }
                LazyOp::Reindex(Reindex::Slice(Slice { src: arg, indices })) => {
                    let arg_dims = arg.shape().to_vec();
                    let index_lens = indices.iter().map(|range| range.end - range.start);

                    // Get index of first dimension with length that doesn't match as a heuristic to make cat work
                    let first_different_index = arg_dims
                        .iter()
                        .zip(index_lens)
                        .position(|(arg_dim, slice_dim)| *arg_dim != slice_dim)
                        .unwrap();

                    let left_pad = if indices[first_different_index].start == 0 {
                        None
                    } else {
                        let mut dims = arg_dims.to_vec();
                        dims[first_different_index] = indices[first_different_index].start;
                        Some(Tensor::zeros::<f32>(&Shape::from(dims), arg.device()))
                    };

                    let right_pad =
                        if arg_dims[first_different_index] == indices[first_different_index].end {
                            None
                        } else {
                            let mut dims = arg_dims.to_vec();
                            dims[first_different_index] = arg_dims[first_different_index]
                                - indices[first_different_index].end;
                            Some(Tensor::zeros::<f32>(&Shape::from(dims), arg.device()))
                        };

                    let arg_grad = match (left_pad, right_pad) {
                        (None, None) => grad.clone(),
                        (Some(left_pad), None) => {
                            Tensor::cat(rvec![left_pad, grad], first_different_index)?
                        }
                        (None, Some(right_pad)) => {
                            Tensor::cat(rvec![grad, right_pad], first_different_index)?
                        }
                        (Some(left_pad), Some(right_pad)) => {
                            Tensor::cat(rvec![left_pad, grad, right_pad], first_different_index)?
                        }
                    };

                    let sum_grad = grads.or_insert(arg.clone())?;
                    *sum_grad = sum_grad.clone().add(arg_grad)?;
                }
                LazyOp::Reindex(Reindex::Permute(Permute { src: arg, dims })) => {
                    let mut inv_dims = vec![0; dims.len()];
                    for (i, &dim) in dims.iter().enumerate() {
                        inv_dims[dim] = i;
                    }
                    let arg_grad = grad.permute(&inv_dims)?;
                    let sum_grad = grads.or_insert(arg.clone())?;
                    *sum_grad = sum_grad.clone().add(arg_grad)?;
                }
                LazyOp::Reduce(Reduce {
                    input: arg,
                    reduced_shape,
                    op: ReduceOp::Sum,
                    ..
                }) => {
                    let grad = broadcast_back(arg, &grad, reduced_shape.inner())?;
                    let sum_grad = grads.or_insert(arg.clone())?;
                    *sum_grad = sum_grad.clone().add(grad)?;
                }
                LazyOp::Reduce(Reduce {
                    input: arg,
                    reduced_shape,
                    op: ReduceOp::Max,
                    ..
                }) => {
                    let node = broadcast_back(arg, node, reduced_shape.inner())?;
                    let grad = broadcast_back(arg, &grad, reduced_shape.inner())?;
                    let grad = node.eq(arg.clone())?.cast(grad.dt())?.mul(grad)?;
                    let sum_grad = grads.or_insert(arg.clone())?;
                    *sum_grad = sum_grad
                        .clone()
                        .add(grad.broadcast_to(sum_grad.shape().clone())?)?;
                }
                LazyOp::Reduce(Reduce {
                    input: arg,
                    reduced_shape,
                    op: ReduceOp::Min,
                    ..
                }) => {
                    let node = broadcast_back(arg, node, reduced_shape.inner())?;
                    let grad = broadcast_back(arg, &grad, reduced_shape.inner())?;
                    let grad = node.eq(arg.clone())?.cast(grad.dt())?.mul(grad)?;
                    let sum_grad = grads.or_insert(arg.clone())?;
                    *sum_grad = sum_grad
                        .clone()
                        .add(grad.broadcast_to(sum_grad.shape().clone())?)?;
                }
                LazyOp::Unary(Unary {
                    input: arg,
                    op: UnaryOp::Log,
                }) => {
                    let sum_grad = grads.or_insert(arg.clone())?;
                    *sum_grad = sum_grad.clone().add((grad / arg.clone())?)?
                }
                LazyOp::Unary(Unary {
                    input: arg,
                    op: UnaryOp::Sin,
                }) => {
                    let sum_grad = grads.or_insert(arg.clone())?;
                    *sum_grad = sum_grad.clone().add((grad * arg.clone().cos())?)?
                }
                LazyOp::Unary(Unary {
                    input: arg,
                    op: UnaryOp::Cos,
                }) => {
                    let sum_grad = grads.or_insert(arg.clone())?;
                    *sum_grad = sum_grad.clone().sub((grad * arg.clone().sin())?)?
                }
                LazyOp::Unary(Unary {
                    input: arg,
                    op: UnaryOp::Tanh,
                }) => {
                    let sum_grad = grads.or_insert(arg.clone())?;
                    let minus_dtanh = ((*node).clone().square()? - 1.)?;
                    *sum_grad = sum_grad.clone().sub((grad.clone() * minus_dtanh)?)?
                }
                LazyOp::Unary(Unary {
                    input: arg,
                    op: UnaryOp::Abs,
                }) => {
                    let sum_grad = grads.or_insert(arg.clone())?;
                    let ones = arg.ones_like::<f32>();
                    let abs_grad = arg
                        .clone()
                        .ge(arg.clone().zeros_like::<f32>())?
                        .where_cond(ones.clone(), ones.neg()?)?;
                    *sum_grad = sum_grad.clone().add((grad * abs_grad)?)?
                }
                LazyOp::Unary(Unary {
                    input: arg,
                    op: UnaryOp::Exp,
                }) => {
                    let sum_grad = grads.or_insert(arg.clone())?;
                    *sum_grad = sum_grad.clone().add((grad * (*node).clone())?)?
                }
                LazyOp::Unary(Unary {
                    input: arg,
                    op: UnaryOp::Neg,
                }) => {
                    let sum_grad = grads.or_insert(arg.clone())?;
                    *sum_grad = sum_grad.clone().sub(grad)?
                }
                LazyOp::Unary(Unary {
                    input: arg,
                    op: UnaryOp::Reciprocal,
                }) => {
                    let sum_grad = grads.or_insert(arg.clone())?;
                    let grad = (grad / arg.clone().square()?)?;
                    *sum_grad = sum_grad.clone().sub(grad)?
                }
                LazyOp::Unary(Unary {
                    input: _,
                    op: UnaryOp::Ceil,
                }) => Err(BackpropError::BackwardNotSupported { op: "ceil" })?,
                LazyOp::Unary(Unary {
                    input: arg,
                    op: UnaryOp::Gelu,
                }) => {
                    let sum_grad = grads.or_insert(arg.clone())?;
                    let cube = arg.clone().powf(3.)?;
                    let tanh = (0.0356774 * cube.clone() + (0.797885 * arg.clone())?)?.tanh()?;
                    let gelu_grad = (((0.5 * tanh.clone())?
                        + (0.0535161 * cube + (0.398942 * arg.clone())?)?
                            * (1. - tanh.clone().powf(2.)?))?
                        + 0.5)?;
                    *sum_grad = sum_grad.clone().add((grad * gelu_grad)?)?
                }
                LazyOp::Unary(Unary {
                    input: arg,
                    op: UnaryOp::Relu,
                }) => {
                    let sum_grad = grads.or_insert(arg.clone())?;
                    let relu_grad = arg.clone().affine(2.0, 0.0)?.mul(
                        arg.clone()
                            .ge(arg.clone().zeros_like::<f32>())?
                            .cast(arg.dt())?,
                    )?;
                    *sum_grad = sum_grad.clone().add((grad * relu_grad)?)?;
                }
                LazyOp::Unary(Unary {
                    input: arg,
                    op: UnaryOp::Relu2,
                }) => {
                    let sum_grad = grads.or_insert(arg.clone())?;
                    let relu_grad = arg
                        .clone()
                        .ge(arg.clone().zeros_like::<f32>())?
                        .cast(arg.dt())?;
                    *sum_grad = sum_grad.clone().add((grad * relu_grad)?)?
                }
                LazyOp::Unary(Unary {
                    input: arg,
                    op: UnaryOp::Silu,
                }) => {
                    let sum_grad = grads.or_insert(arg.clone())?;
                    // d/dx silu = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
                    let sigmoid_arg = (arg.clone().neg()?.exp()? + 1.)?.recip()?;
                    let silu_grad =
                        (sigmoid_arg.clone() * (1. + (arg.clone() * (1. - sigmoid_arg)?)?)?)?;
                    *sum_grad = sum_grad.clone().add((grad * silu_grad)?)?
                }
                LazyOp::Unary(Unary {
                    input: arg,
                    op: UnaryOp::Square,
                }) => {
                    let arg_grad = arg.clone().mul(grad.clone())?.affine(2., 0.)?;
                    let sum_grad = grads.or_insert(arg.clone())?;
                    *sum_grad = sum_grad.clone().add(arg_grad)?
                }
                LazyOp::Unary(Unary {
                    input: arg,
                    op: UnaryOp::Sqrt,
                }) => {
                    let arg_grad = grad.div((*node).clone())?.affine(0.5, 0.)?;
                    let sum_grad = grads.or_insert(arg.clone())?;
                    *sum_grad = sum_grad.clone().add(arg_grad)?
                }
                LazyOp::Detach(_) => todo!(),
                LazyOp::Unary(Unary {
                    input: _,
                    op: UnaryOp::Sigmoid,
                }) => todo!(),
                LazyOp::Unary(Unary {
                    input: _,
                    op: UnaryOp::Floor,
                })
                | LazyOp::Reduce(Reduce {
                    input: _,
                    op: ReduceOp::ArgMax,
                    ..
                })
                | LazyOp::Reduce(Reduce {
                    input: _,
                    op: ReduceOp::ArgMin,
                    ..
                })
                | LazyOp::FillConstant(_)
                | LazyOp::FillRandn(_)
                | LazyOp::Arange(_) => {}
                LazyOp::View(View { src: arg, .. }) => {
                    let arg_grad = grad.clone().view(arg.shape().clone())?;
                    let sum_grad = grads.or_insert(arg.clone())?;
                    *sum_grad = sum_grad.clone().add(arg_grad)?;
                }
                LazyOp::Select(IndexSelect {
                    src: arg,
                    indices,
                    dim,
                }) => {
                    let sum_grad = grads.or_insert(arg.clone())?;
                    *sum_grad = sum_grad
                        .clone()
                        .index_add(indices.clone(), grad.clone(), *dim)?;
                }
                LazyOp::Softmax(Softmax { input: arg, dim }) => {
                    // Get the softmax output (s)
                    let softmax_output = (*node).clone();

                    // Compute the sum of the gradients
                    let sum_grad = grad.clone().sum_keepdim(&[*dim])?;

                    // Compute the gradient with respect to the softmax input
                    let input_grad = softmax_output
                        .clone()
                        .mul(grad.clone())?
                        .sub(softmax_output.clone().mul(sum_grad)?)?;

                    // Insert the computed gradient into the grads map
                    let arg_sum_grad = grads.or_insert(arg.clone())?;
                    *arg_sum_grad = arg_sum_grad.clone().add(input_grad)?;
                }
                LazyOp::Norm(NormOp::LayerNorm(Norm {
                    input: arg,
                    scale,
                    bias,
                    eps,
                })) => {
                    let d = arg.shape()[1] as f32;

                    // Retrieve the necessary intermediate values from the forward pass
                    let mean = (arg.clone().sum(&[0])? / d)?; // Compute mean of the input
                    let mean_broadcast = mean.clone().broadcast_to(arg.shape().clone())?;

                    let var = (arg
                        .clone()
                        .sub(mean_broadcast.clone())?
                        .square()?
                        .sum(&[0])?
                        / d)?;

                    let x_normed = arg
                        .clone()
                        .sub(mean_broadcast)?
                        .div((var.clone() + *eps)?.sqrt()?)?;

                    // Compute the gradients with respect to beta and gamma
                    let grad_beta = grad.clone().sum_keepdim(&[0])?;
                    let grad_gamma = (x_normed.clone().mul(grad.clone()))?.sum_keepdim(&[0])?;

                    // Compute the gradient with respect to the normalized input
                    let grad_x_normed = grad.clone().mul(scale.clone())?;

                    // Compute the gradients with respect to mean and variance
                    let std = (var.clone() + *eps)?.sqrt()?;
                    let grad_mean =
                        (grad_x_normed.clone().sum_keepdim(&[1])?.neg())?.div(std.clone())?;
                    let grad_var = ((grad_x_normed.clone().mul(x_normed.clone()))?
                        .sum_keepdim(&[1])?
                        .neg()?
                        .div((var.clone() + *eps)?)?
                        / 2.0)?;

                    let grad_x = grad_x_normed
                        .clone()
                        .div(std.clone())?
                        .add((grad_mean.clone() / d)?)?
                        .add(
                            (x_normed
                                .clone()
                                .mul(std.clone())?
                                .mul((grad_var.clone() * 2.0)?)?
                                / d)?,
                        )?;

                    // Insert the computed gradients into the grads map
                    let sum_grad = grads.or_insert(arg.clone())?;
                    *sum_grad = sum_grad.clone().add(grad_x)?;

                    let gamma_sum_grad = grads.or_insert(scale.clone())?;
                    *gamma_sum_grad = gamma_sum_grad.clone().add(grad_gamma)?;

                    let beta_sum_grad = grads.or_insert(bias.clone().unwrap().clone())?;
                    *beta_sum_grad = beta_sum_grad.clone().add(grad_beta)?;
                }
                LazyOp::Affine(Affine { src: arg, mul, .. }) => {
                    let arg_grad = grad.affine(*mul, 0.)?;
                    let sum_grad = grads.or_insert(arg.clone())?;
                    *sum_grad = sum_grad.clone().add(arg_grad.clone())?
                }
                LazyOp::Gather(Gather { src, ids, dim, .. }) => {
                    let sum_grad = grads.or_insert(src.clone())?;
                    *sum_grad = sum_grad
                        .clone()
                        .scatter_add(ids.clone(), grad.clone(), *dim)?;
                }
                LazyOp::ScatterAdd(ScatterAdd { dst, src, ids, dim }) => {
                    let dst_sum_grad = grads.or_insert(dst.clone())?;
                    *dst_sum_grad = dst_sum_grad.clone().add(grad.clone())?;

                    let src_grad = grad.gather(ids.clone(), *dim)?;
                    let src_sum_grad = grads.or_insert(src.clone())?;
                    *src_sum_grad = src_sum_grad.clone().add(src_grad.clone())?;
                }
                LazyOp::Trilu(Trilu { src: arg, upper, k }) => {
                    let masked_grad = if *upper { grad.triu(*k) } else { grad.tril(*k) }?;

                    let sum_grad = grads.or_insert(arg.clone())?;
                    *sum_grad = sum_grad.clone().add(masked_grad)?;
                }
                LazyOp::Norm(_) => todo!(),
                LazyOp::Const => panic!("ratchet internal error - const node in backprop"),
                LazyOp::Concat(_) => todo!(),
                LazyOp::Cmp(_) => todo!(),
                LazyOp::Powf(_) => todo!(),
                LazyOp::Cast(_) => todo!(),
                LazyOp::RoPE(_) => todo!(),
                LazyOp::Conv(_) => todo!(),
                LazyOp::IndexWrite(_) => todo!(),
                LazyOp::IndexAdd(_) => todo!(),
                LazyOp::Cache(_) => todo!(),
                LazyOp::Copy(_) => todo!(),
            };
        }
        #[cfg(feature = "plotting")]
        {
            crate::plot::render_backward_to_file(&grads, "backward.svg").unwrap();
        }
        Ok(grads)
    }
}

/// A store for gradients, associating a tensor id to the corresponding gradient tensor, used for back propagation.
#[derive(Debug)]
pub struct GradStore(HashMap<TensorId, Tensor>);

impl GradStore {
    /// Create a new gradient store
    fn new() -> Self {
        GradStore(HashMap::default())
    }

    /// Get the gradient tensor corresponding to the given tensor id
    pub fn get_id(&self, id: TensorId) -> Option<&Tensor> {
        self.0.get(&id)
    }

    /// Get the gradient tensor associated with the given tensor
    pub fn get(&self, tensor: &Tensor) -> Option<&Tensor> {
        self.0.get(&tensor.id())
    }

    /// Remove the gradient tensor associated with the given tensor, returning it if it exists
    pub fn remove(&mut self, tensor: &Tensor) -> Option<Tensor> {
        self.0.remove(&tensor.id())
    }

    /// Insert a gradient tensor associated with the given tensor, returning the previous gradient tensor if it existed
    pub fn insert(&mut self, tensor: &Tensor, grad: Tensor) -> Option<Tensor> {
        self.0.insert(tensor.id(), grad)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&TensorId, &Tensor)> {
        self.0.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&TensorId, &mut Tensor)> {
        self.0.iter_mut()
    }

    /// Get the gradient tensor associated with the given tensor, or, if it does not exist,
    /// insert a tensor of zeroes, with the same shape and type as the given tensors and return it
    fn or_insert(&mut self, tensor: Tensor) -> Result<&mut Tensor> {
        let grad = match self.0.entry(tensor.id()) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let grad = tensor.clone().zeros_like::<f32>();
                entry.insert(grad)
            }
        };
        Ok(grad)
    }
}
