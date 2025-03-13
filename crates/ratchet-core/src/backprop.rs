/// Adapted from candle:
/// https://github.com/huggingface/candle/blob/main/candle-core/src/backprop.rs
/// Methods for backpropagation of gradients.
use crate::ops::{BinaryOp, UnaryOp};
use crate::{
    rvec, Affine, Alibi, Binary, Broadcast, Cast, Cmp, Concat, Conv, DType, Gather, GroupNorm,
    IndexAdd, IndexSelect, LazyOp, Matmul, Norm, NormOp, Permute, Powf, Reduce, ReduceOp, Reindex,
    RoPE, ScatterAdd, ScopePusher, Shape, Slice, Softmax, Tensor, TensorId, Unary, View, WhereCond,
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
            let mut nodes = if node.requires_grad() {
                // Do not call recursively on the "leaf" nodes.
                track_grad = true;
                nodes
            } else if matches!(node.dtype(), DType::I32 | DType::U32) {
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
                    | LazyOp::RoPE(RoPE { input: node, .. })
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
                    LazyOp::Cast(Cast { input, .. }) => {
                        if input.dtype().is_float() {
                            let (tg, nodes) = walk(input, nodes, already_seen);
                            track_grad |= tg;
                            nodes
                        } else {
                            nodes
                        }
                    }
                    LazyOp::IndexWrite(_) => todo!(),
                    LazyOp::Copy(_) => todo!(),
                    LazyOp::Detach(_)
                    | LazyOp::Const
                    | LazyOp::Alibi(_)
                    | LazyOp::Reduce(Reduce {
                        op: ReduceOp::ArgMax | ReduceOp::ArgMin,
                        ..
                    })
                    | LazyOp::FillConstant(_)
                    | LazyOp::FillRandn(_)
                    | LazyOp::Bernoulli(_)
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
        let _scope_guard = ScopePusher::new("backward");
        let sorted_nodes = self.sorted_nodes();
        let mut grads = GradStore::new();
        grads.insert(self, self.ones_like::<f32>()?.contiguous()?);
        for node in sorted_nodes.iter() {
            let _op_scope_guard = ScopePusher::new(&format!("for:{}", node.op().name()));
            if node.requires_grad() {
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
                    grads.accumulate_add(lhs, grad.clone())?;
                    grads.accumulate_add(rhs, grad)?;
                }
                LazyOp::Binary(Binary {
                    lhs,
                    rhs,
                    op: BinaryOp::Sub,
                }) => {
                    grads.accumulate_add(lhs, grad.clone())?;
                    grads.accumulate_sub(rhs, grad)?;
                }
                LazyOp::Binary(Binary {
                    lhs,
                    rhs,
                    op: BinaryOp::Mul,
                }) => {
                    let lhs_grad = grad.clone().mul(rhs.clone())?;
                    grads.accumulate_add(lhs, lhs_grad)?;
                    let rhs_grad = grad.mul(lhs.clone())?;
                    grads.accumulate_add(rhs, rhs_grad)?;
                }
                LazyOp::Binary(Binary {
                    lhs,
                    rhs,
                    op: BinaryOp::Div,
                }) => {
                    let lhs_grad = grad.clone().div(rhs.clone())?;
                    grads.accumulate_add(lhs, lhs_grad)?;
                    let rhs_grad = grad.mul(lhs.clone())?.div(rhs.clone().square()?)?;
                    grads.accumulate_sub(rhs, rhs_grad)?;
                }
                LazyOp::WhereCond(WhereCond {
                    input,
                    on_true,
                    on_false,
                }) => {
                    let zeros = grad.clone().zeros_like::<f32>()?;
                    let t_grad = input.clone().where_cond(grad.clone(), zeros.clone())?;
                    grads.accumulate_add(on_true, t_grad)?;
                    let f_grad = input.clone().where_cond(zeros, grad)?;
                    grads.accumulate_add(on_false, f_grad)?;
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
                    grads.accumulate_add(lhs, lhs_grad)?;

                    let rhs_grad =
                        lhs.clone()
                            .gemm(grad.clone(), None, !trans_lhs, *trans_dst, *trans_rhs)?;
                    grads.accumulate_add(rhs, rhs_grad)?;

                    // Calculate the gradient with respect to the bias term
                    if let Some(bias) = bias {
                        let bias_grad = grad.sum_keepdim(&[0])?; // Assuming bias is summed over the appropriate axis
                        grads.accumulate_add(bias, bias_grad)?;
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
                    grads.accumulate_add(src, arg_grad.broadcast_to(src.shape().clone())?)?;
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
                        Some(Tensor::zeros::<f32>(&Shape::from(dims), arg.device())?)
                    };

                    let right_pad =
                        if arg_dims[first_different_index] == indices[first_different_index].end {
                            None
                        } else {
                            let mut dims = arg_dims.to_vec();
                            dims[first_different_index] = arg_dims[first_different_index]
                                - indices[first_different_index].end;
                            Some(Tensor::zeros::<f32>(&Shape::from(dims), arg.device())?)
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

                    grads.accumulate_add(arg, arg_grad)?;
                }
                LazyOp::Reindex(Reindex::Permute(Permute { src: arg, dims })) => {
                    let mut inv_dims = vec![0; dims.len()];
                    for (i, &dim) in dims.iter().enumerate() {
                        inv_dims[dim] = i;
                    }
                    let arg_grad = grad.permute(&inv_dims)?;
                    grads.accumulate_add(arg, arg_grad)?;
                }
                LazyOp::Reduce(Reduce {
                    input: arg,
                    reduced_shape,
                    op: ReduceOp::Sum,
                    ..
                }) => {
                    let grad = broadcast_back(arg, &grad, reduced_shape.inner())?;
                    grads.accumulate_add(arg, grad)?;
                }
                LazyOp::Reduce(Reduce {
                    input: arg,
                    reduced_shape,
                    op: ReduceOp::Max,
                    ..
                }) => {
                    let node = broadcast_back(arg, node, reduced_shape.inner())?;
                    let grad = broadcast_back(arg, &grad, reduced_shape.inner())?;
                    let grad = node.eq(arg.clone())?.cast(grad.dtype())?.mul(grad)?;
                    grads.accumulate_add(arg, grad.broadcast_to(arg.shape().clone())?)?;
                }
                LazyOp::Reduce(Reduce {
                    input: arg,
                    reduced_shape,
                    op: ReduceOp::Min,
                    ..
                }) => {
                    let node = broadcast_back(arg, node, reduced_shape.inner())?;
                    let grad = broadcast_back(arg, &grad, reduced_shape.inner())?;
                    let grad = node.eq(arg.clone())?.cast(grad.dtype())?.mul(grad)?;
                    grads.accumulate_add(arg, grad.broadcast_to(arg.shape().clone())?)?;
                }
                LazyOp::Unary(Unary {
                    input: arg,
                    op: UnaryOp::Log,
                }) => {
                    let arg_grad = (grad / arg.clone())?;
                    grads.accumulate_add(arg, arg_grad)?;
                }
                LazyOp::Unary(Unary {
                    input: arg,
                    op: UnaryOp::Sin,
                }) => {
                    let arg_grad = (grad * arg.clone().cos())?;
                    grads.accumulate_add(arg, arg_grad)?;
                }
                LazyOp::Unary(Unary {
                    input: arg,
                    op: UnaryOp::Cos,
                }) => {
                    let arg_grad = (grad * arg.clone().sin())?;
                    grads.accumulate_sub(arg, arg_grad)?;
                }
                LazyOp::Unary(Unary {
                    input: arg,
                    op: UnaryOp::Tanh,
                }) => {
                    let minus_dtanh = ((*node).clone().square()? - 1.)?;
                    let arg_grad = (grad.clone() * minus_dtanh)?;
                    grads.accumulate_sub(arg, arg_grad)?;
                }
                LazyOp::Unary(Unary {
                    input: arg,
                    op: UnaryOp::Abs,
                }) => {
                    let ones = arg.ones_like::<f32>()?;
                    let abs_grad = arg
                        .clone()
                        .ge(arg.clone().zeros_like::<f32>()?)?
                        .where_cond(ones.clone(), ones.neg()?)?;
                    let arg_grad = (grad * abs_grad)?;
                    grads.accumulate_add(arg, arg_grad)?;
                }
                LazyOp::Unary(Unary {
                    input: arg,
                    op: UnaryOp::Exp,
                }) => {
                    let arg_grad = (grad * (*node).clone())?;
                    grads.accumulate_add(arg, arg_grad)?;
                }
                LazyOp::Unary(Unary {
                    input: arg,
                    op: UnaryOp::Neg,
                }) => {
                    grads.accumulate_sub(arg, grad)?;
                }
                LazyOp::Unary(Unary {
                    input: arg,
                    op: UnaryOp::Reciprocal,
                }) => {
                    let arg_grad = (grad / arg.clone().square()?)?;
                    grads.accumulate_sub(arg, arg_grad)?;
                }
                LazyOp::Unary(Unary {
                    input: _,
                    op: UnaryOp::Ceil,
                }) => Err(BackpropError::BackwardNotSupported { op: "ceil" })?,
                LazyOp::Unary(Unary {
                    input: arg,
                    op: UnaryOp::Gelu,
                }) => {
                    let cube = arg.clone().powf(3.)?;
                    let tanh = (0.0356774 * cube.clone() + (0.797885 * arg.clone())?)?.tanh()?;
                    let gelu_grad = (((0.5 * tanh.clone())?
                        + (0.0535161 * cube + (0.398942 * arg.clone())?)?
                            * (1. - tanh.clone().powf(2.)?))?
                        + 0.5)?;
                    let arg_grad = (grad * gelu_grad)?;
                    grads.accumulate_add(arg, arg_grad)?;
                }
                LazyOp::Unary(Unary {
                    input: arg,
                    op: UnaryOp::Relu,
                }) => {
                    let relu_grad = arg.clone().affine(2.0, 0.0)?.mul(
                        arg.clone()
                            .ge(arg.clone().zeros_like::<f32>()?)?
                            .cast(arg.dtype())?,
                    )?;
                    let arg_grad = grad.mul(relu_grad)?;
                    grads.accumulate_add(arg, arg_grad)?;
                }
                LazyOp::Unary(Unary {
                    input: arg,
                    op: UnaryOp::Relu2,
                }) => {
                    let relu_grad = arg
                        .clone()
                        .ge(arg.clone().zeros_like::<f32>()?)?
                        .cast(arg.dtype())?;
                    let arg_grad = grad.mul(relu_grad)?;
                    grads.accumulate_add(arg, arg_grad)?;
                }
                LazyOp::Unary(Unary {
                    input: arg,
                    op: UnaryOp::Silu,
                }) => {
                    let sigmoid_arg = (arg.clone().neg()?.exp()? + 1.)?.recip()?;
                    let silu_grad =
                        (sigmoid_arg.clone() * (1. + (arg.clone() * (1. - sigmoid_arg)?)?)?)?;
                    let arg_grad = grad.mul(silu_grad)?;
                    grads.accumulate_add(arg, arg_grad)?;
                }
                LazyOp::Unary(Unary {
                    input: arg,
                    op: UnaryOp::Swiglu,
                }) => {
                    // swiglu(x) = x^2 * sigma(x)
                    //
                    // By product rule:
                    // d/dx [x^2 * sigma(x)] = 2x * sigma(x) + x^2 * sigma(x)*(1 - sigma(x)).

                    // 1) Compute sigma(x) = 1 / (1 + e^-x).
                    let sigmoid_arg = (arg.clone().neg()?.exp()? + 1.)?.recip()?;

                    // By product rule:
                    // 2) term1 = 2x * sigma(x).
                    let product_term_1 = (arg.clone() * 2.)?.mul(sigmoid_arg.clone())?;

                    // 3) term2 = x^2 * sigma(x)*(1 - sigma(x)).
                    let product_term_2 = arg
                        .clone()
                        .square()?
                        .mul(sigmoid_arg.clone())?
                        .mul((1. - sigmoid_arg.clone())?)?;

                    // 4) Final derivative wrt x is term1 + term2; multiply by the chain-rule grad.
                    let swiglu_grad = product_term_1.add(product_term_2)?;
                    let arg_grad = grad.mul(swiglu_grad)?;

                    grads.accumulate_add(arg, arg_grad)?;
                }
                LazyOp::Unary(Unary {
                    input: arg,
                    op: UnaryOp::Square,
                }) => {
                    let arg_grad = arg.clone().mul(grad)?.affine(2., 0.)?;
                    grads.accumulate_add(arg, arg_grad)?;
                }
                LazyOp::Unary(Unary {
                    input: arg,
                    op: UnaryOp::Sqrt,
                }) => {
                    let arg_grad = grad.div((*node).clone())?.affine(0.5, 0.)?;
                    grads.accumulate_add(arg, arg_grad)?;
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
                | LazyOp::Bernoulli(_)
                | LazyOp::Arange(_) => {}
                LazyOp::View(View { src: arg, .. }) => {
                    let arg_grad = grad.clone().view(arg.shape().clone())?;
                    grads.accumulate_add(arg, arg_grad)?;
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

                    grads.accumulate_add(arg, input_grad)?;
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

                    grads.accumulate_add(arg, grad_x)?;
                    grads.accumulate_add(scale, grad_gamma)?;
                    grads.accumulate_add(&bias.clone().unwrap(), grad_beta)?;
                }
                LazyOp::Affine(Affine { src: arg, mul, .. }) => {
                    let arg_grad = grad.affine(*mul, 0.)?;
                    grads.accumulate_add(arg, arg_grad)?;
                }
                LazyOp::Gather(Gather { src, ids, dim, .. }) => {
                    let sum_grad = grads.or_insert(src.clone())?;
                    *sum_grad = sum_grad
                        .clone()
                        .scatter_add(ids.clone(), grad.clone(), *dim)?;
                }
                LazyOp::ScatterAdd(ScatterAdd { dst, src, ids, dim }) => {
                    grads.accumulate_add(dst, grad.clone())?;
                    let src_grad = grad.gather(ids.clone(), *dim)?;
                    grads.accumulate_add(src, src_grad)?;
                }
                LazyOp::Trilu(Trilu { src: arg, upper, k }) => {
                    let masked_grad = if *upper {
                        grad.triu(*k)?
                    } else {
                        grad.tril(*k)?
                    };
                    grads.accumulate_add(arg, masked_grad)?;
                }
                LazyOp::Alibi(Alibi { input, .. }) => {
                    grads.accumulate_add(input, grad)?;
                }
                LazyOp::Cast(Cast {
                    input,
                    dst_dtype: _,
                }) => {
                    grads.accumulate_add(input, grad.cast(input.dtype())?)?;
                }
                LazyOp::Norm(_) => todo!(),
                LazyOp::Const => panic!("ratchet internal error - const node in backprop"),
                LazyOp::Concat(_) => todo!(),
                LazyOp::Cmp(_) => todo!(),
                LazyOp::Powf(_) => todo!(),
                LazyOp::RoPE(RoPE {
                    input: arg,
                    dim,
                    base,
                    offset,
                    ..
                }) => {
                    let arg_grad = grad.rope_backward(*dim, *base, *offset)?;
                    grads.accumulate_add(arg, arg_grad)?;
                }
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
                let grad = tensor.clone().zeros_like::<f32>()?;
                entry.insert(grad)
            }
        };
        Ok(grad)
    }

    /// If there's an existing gradient for `tensor`, add `grad` to it.
    /// Otherwise, just store `grad` as-is (no need to create zeros and then add).
    fn accumulate_add(&mut self, tensor: &Tensor, grad: Tensor) -> Result<()> {
        use std::collections::hash_map::Entry;
        match self.0.entry(tensor.id()) {
            Entry::Occupied(mut entry) => {
                let existing = entry.get_mut();
                *existing = existing.clone().add(grad)?;
            }
            Entry::Vacant(entry) => {
                // TODO(vinhowe): This is a hack to avoid creating zeros and then adding; it does
                // increase perf.
                // It's not great; we should do a tensor copy or something.
                entry.insert(grad.affine(1., 0.)?);
            }
        }
        Ok(())
    }

    fn accumulate_sub(&mut self, tensor: &Tensor, grad: Tensor) -> Result<()> {
        use std::collections::hash_map::Entry;
        match self.0.entry(tensor.id()) {
            Entry::Occupied(mut entry) => {
                let existing = entry.get_mut();
                *existing = existing.clone().sub(grad)?;
            }
            Entry::Vacant(entry) => {
                entry.insert(grad.neg()?);
            }
        }
        Ok(())
    }
}
