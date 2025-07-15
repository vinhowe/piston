/// Adapted from candle:
/// https://github.com/huggingface/candle/blob/main/candle-core/src/backprop.rs
/// Methods for backpropagation of gradients.
use crate::ops::{BinaryOp, TernaryOp, UnaryOp};
use crate::{
    rvec, Affine, Alibi, Binary, Broadcast, Cast, Cmp, Concat, Conv, DType, Gather, GroupNorm,
    IndexAdd, IndexSelect, LazyOp, Matmul, Norm, NormOp, OpTensor, Permute, Powf, Reduce, ReduceOp,
    Reindex, RoPE, ScatterAdd, ScopePusher, Slice, Softmax, Tensor, TensorId, Ternary, Unary, View,
    WhereCond,
};
use crate::{HashMap, HashSet, Trilu};
use anyhow::Result;

#[derive(thiserror::Error, Debug)]
pub enum BackpropError {
    #[error("Tensor is not resolved")]
    BackwardNotSupported { op: &'static str },
}

// arg has been reduced to node via reduce_dims, expand it back to arg.
// This has to handle keepdims.
fn broadcast_back(arg: &OpTensor, node: &OpTensor, reduced_dims: &[usize]) -> Result<OpTensor> {
    if arg.dim() == node.dim() {
        // keepdim = true
        node.clone().broadcast_to(arg.shape().clone())
    } else {
        // keepdim = false
        node.clone()
            .view(reduced_dims)?
            .broadcast_to(arg.shape().clone())
    }
}

/// Get the gradient tensor associated with the given tensor, or, if it does not exist,
/// insert a tensor of zeroes, with the same shape and type as the given tensors and return it
fn or_insert(tensor: &OpTensor) -> Result<OpTensor> {
    let grad = match tensor.grad() {
        Some(grad) => grad,
        None => {
            let grad = tensor.clone().zeros_like::<f32>(None, false)?;
            tensor.set_grad(Some(grad.clone()));
            grad
        }
    };
    Ok(grad)
}

/// Context for gradient accumulation during backpropagation.
/// Tracks which tensors should receive gradients and provides methods for accumulation.
struct GradAccumContext {
    tracked: HashSet<TensorId>,
}

impl GradAccumContext {
    fn new(tracked_nodes: &[&OpTensor]) -> Self {
        let tracked = tracked_nodes.iter().map(|node| node.id()).collect();
        Self { tracked }
    }

    /// Add gradient to tensor if it's being tracked
    fn add(&self, tensor: &OpTensor, grad: OpTensor) -> Result<()> {
        if !self.tracked.contains(&tensor.id()) {
            return Ok(());
        }
        match tensor.grad() {
            Some(existing_grad) => {
                tensor.set_grad(Some(existing_grad.clone().add(grad)?));
            }
            None => {
                // TODO(vinhowe): This is a hack to avoid creating zeros and then adding; it does
                // increase perf.
                // It's not great; we should do a tensor copy or something.
                tensor.set_grad(Some(grad.affine(1., 0.)?));
            }
        }
        Ok(())
    }

    /// Subtract gradient from tensor if it's being tracked
    fn sub(&self, tensor: &OpTensor, grad: OpTensor) -> Result<()> {
        if !self.tracked.contains(&tensor.id()) {
            return Ok(());
        }
        match tensor.grad() {
            Some(existing_grad) => {
                tensor.set_grad(Some(existing_grad.clone().sub(grad)?));
            }
            None => {
                // TODO(vinhowe): This is a hack to avoid creating zeros and then adding; it does
                // increase perf.
                // It's not great; we should do a tensor copy or something.
                tensor.set_grad(Some(grad.neg()?));
            }
        }
        Ok(())
    }
}

thread_local! {
    static PISTON_GRAD_DO_NOT_DETACH: bool = {
        match std::env::var("PISTON_GRAD_DO_NOT_DETACH") {
            Ok(s) => {
                !s.is_empty() && s != "0"
            },
            Err(_) => false,
        }
    }
}

impl OpTensor {
    /// Return all the nodes that lead to this value in a topologically sorted vec, the first
    /// elements having dependencies on the latter ones, e.g. the first element if any is the
    /// argument.
    /// This assumes that the op graph is a DAG.
    // TODO(vinhowe): This could be consolidated with execution_order and whatever caching we
    // do...
    fn sorted_nodes(&self) -> Vec<&OpTensor> {
        // The vec of sorted nodes is passed as an owned value rather than a mutable reference
        // to get around some lifetime limitations.
        fn walk<'a>(
            node: &'a OpTensor,
            nodes: Vec<&'a OpTensor>,
            already_seen: &mut HashMap<TensorId, bool>,
        ) -> (bool, Vec<&'a OpTensor>) {
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
                    })
                    | LazyOp::Ternary(Ternary {
                        input: t1,
                        tensor1: t2,
                        tensor2: t3,
                        ..
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
                log::trace!("Tracking grad for node {:?}", node.id());
            } else {
                log::trace!("Not tracking grad for node {:?}", node.id());
            }
            (track_grad, nodes)
        }
        let (_tg, mut nodes) = walk(self, vec![], &mut HashMap::default());
        nodes.reverse();
        nodes
    }

    pub fn backward(&self) -> Result<()> {
        let _scope_guard = ScopePusher::new("backward");
        let sorted_nodes = self.sorted_nodes();

        // Create gradient context for tracked tensors
        let ctx = GradAccumContext::new(&sorted_nodes);

        self.set_grad(Some(self.ones_like::<f32>(None, false)?.contiguous()?));
        for node in sorted_nodes.iter() {
            let _op_scope_guard = ScopePusher::new(&format!("for:{}", node.op().name()));
            if node.requires_grad() {
                continue;
            }
            log::trace!("Backwarding {:?}", node.id());
            // This just says that we don't track intermediate gradients.
            let grad = node
                .take_grad()
                .expect("piston internal error - grad not populated");
            // From candle:
            // https://github.com/huggingface/candle/issues/1241
            // Ideally, we would make these operations in place where possible to ensure that we
            // do not have to allocate too often. Here we just call `.detach` to avoid computing
            // the backprop graph of the backprop itself. This would be an issue for second order
            // derivatives but these are out of scope at the moment.
            let do_not_detach = PISTON_GRAD_DO_NOT_DETACH.with(|b| *b);
            let grad = if do_not_detach { grad } else { grad.detach() };
            match node.op() {
                LazyOp::Binary(Binary {
                    lhs,
                    rhs,
                    op: BinaryOp::Add,
                }) => {
                    ctx.add(lhs, grad.clone())?;
                    ctx.add(rhs, grad)?;
                }
                LazyOp::Binary(Binary {
                    lhs,
                    rhs,
                    op: BinaryOp::Sub,
                }) => {
                    ctx.add(lhs, grad.clone())?;
                    ctx.sub(rhs, grad)?;
                }
                LazyOp::Binary(Binary {
                    lhs,
                    rhs,
                    op: BinaryOp::Mul,
                }) => {
                    let lhs_grad = grad.clone().mul(rhs.clone())?;
                    ctx.add(lhs, lhs_grad)?;
                    let rhs_grad = grad.mul(lhs.clone())?;
                    ctx.add(rhs, rhs_grad)?;
                }
                LazyOp::Binary(Binary {
                    lhs,
                    rhs,
                    op: BinaryOp::Div,
                }) => {
                    let lhs_grad = grad.clone().div(rhs.clone())?;
                    ctx.add(lhs, lhs_grad)?;
                    let rhs_grad = grad.mul(lhs.clone())?.div(rhs.clone().square()?)?;
                    ctx.sub(rhs, rhs_grad)?;
                }
                LazyOp::Binary(Binary {
                    lhs,
                    rhs,
                    op: BinaryOp::Maximum,
                })
                | LazyOp::Binary(Binary {
                    lhs,
                    rhs,
                    op: BinaryOp::Minimum,
                }) => {
                    let mask_lhs = (*node).clone().eq(lhs.clone())?.cast(grad.dtype())?;
                    let mask_rhs = (*node).clone().eq(rhs.clone())?.cast(grad.dtype())?;

                    // If both masks are 1 one the same point, we want to scale the
                    // gradient by 0.5 rather than 1.
                    let lhs_grad = mask_lhs
                        .clone()
                        .mul(grad.clone())?
                        .div((mask_rhs.clone() + 1.)?)?;
                    ctx.add(lhs, lhs_grad)?;

                    let rhs_grad = mask_rhs.mul(grad)?.div((mask_lhs + 1.)?)?;
                    ctx.add(rhs, rhs_grad)?;
                }
                LazyOp::Ternary(Ternary {
                    input,
                    tensor1,
                    tensor2,
                    value,
                    op: TernaryOp::Addcdiv,
                }) => {
                    // addcdiv: input + value * (tensor1 / tensor2)
                    // Gradient for input is simply grad
                    ctx.add(input, grad.clone())?;

                    // Gradient for tensor1 is grad * value / tensor2
                    let tensor1_grad = grad
                        .clone()
                        .mul(tensor2.clone().recip()?)?
                        .affine(*value, 0.)?;
                    ctx.add(tensor1, tensor1_grad)?;

                    // Gradient for tensor2 is -grad * value * tensor1 / tensor2^2
                    let tensor2_grad = grad
                        .mul(tensor1.clone())?
                        .div(tensor2.clone().square()?)?
                        .affine(-*value, 0.)?;
                    ctx.add(tensor2, tensor2_grad)?;
                }
                LazyOp::Ternary(Ternary {
                    input,
                    tensor1,
                    tensor2,
                    value,
                    op: TernaryOp::Addcmul,
                }) => {
                    // addcmul: input + value * (tensor1 * tensor2)
                    // Gradient for input is simply grad
                    ctx.add(input, grad.clone())?;

                    // Gradient for tensor1 is grad * value * tensor2
                    let tensor1_grad = grad.clone().mul(tensor2.clone())?.affine(*value, 0.)?;
                    ctx.add(tensor1, tensor1_grad)?;

                    // Gradient for tensor2 is grad * value * tensor1
                    let tensor2_grad = grad.mul(tensor1.clone())?.affine(*value, 0.)?;
                    ctx.add(tensor2, tensor2_grad)?;
                }
                LazyOp::WhereCond(WhereCond {
                    input,
                    on_true,
                    on_false,
                }) => {
                    let zeros = grad.clone().zeros_like::<f32>(None, false)?;
                    let t_grad = grad.clone().where_cond(input.clone(), zeros.clone())?;
                    ctx.add(on_true, t_grad)?;
                    let f_grad = zeros.clone().where_cond(input.clone(), grad)?;
                    ctx.add(on_false, f_grad)?;
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
                    ctx.add(lhs, lhs_grad)?;

                    let rhs_grad =
                        lhs.clone()
                            .gemm(grad.clone(), None, !trans_lhs, *trans_dst, *trans_rhs)?;
                    ctx.add(rhs, rhs_grad)?;

                    // Calculate the gradient with respect to the bias term
                    if let Some(bias) = bias {
                        let bias_grad = grad.sum_keepdim(1)?; // Assuming bias is summed over the appropriate axis
                        ctx.add(bias, bias_grad)?;
                    }
                }
                LazyOp::Reindex(Reindex::Broadcast(Broadcast { src, .. })) => {
                    let arg_dims = src.shape().inner();
                    let node_dims = node.shape().inner();

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
                        arg_grad = arg_grad.squeeze_all()?;
                    }
                    ctx.add(src, arg_grad.broadcast_to(src.shape().clone())?)?;
                }
                LazyOp::Reindex(Reindex::Slice(Slice { src: arg, indices })) => {
                    let arg_dims = arg.shape().inner();
                    let index_lens = indices.iter().map(|range| range.end - range.start);

                    // Get index of first dimension with length that doesn't match as a heuristic
                    // to make cat work
                    let first_different_index = arg_dims
                        .iter()
                        .zip(index_lens)
                        .position(|(arg_dim, slice_dim)| *arg_dim != slice_dim)
                        .unwrap();

                    let left_pad = if indices[first_different_index].start == 0 {
                        None
                    } else {
                        let mut dims = arg_dims.clone();
                        dims[first_different_index] = indices[first_different_index].start;
                        Some(OpTensor::zeros::<f32, _>(dims, arg.device(), false)?)
                    };

                    let right_pad =
                        if arg_dims[first_different_index] == indices[first_different_index].end {
                            None
                        } else {
                            let mut dims = arg_dims.clone();
                            dims[first_different_index] = arg_dims[first_different_index]
                                - indices[first_different_index].end;
                            Some(OpTensor::zeros::<f32, _>(dims, arg.device(), false)?)
                        };

                    let arg_grad = match (left_pad, right_pad) {
                        (None, None) => grad.clone(),
                        (Some(left_pad), None) => {
                            OpTensor::cat(rvec![left_pad, grad], first_different_index)?
                        }
                        (None, Some(right_pad)) => {
                            OpTensor::cat(rvec![grad, right_pad], first_different_index)?
                        }
                        (Some(left_pad), Some(right_pad)) => {
                            OpTensor::cat(rvec![left_pad, grad, right_pad], first_different_index)?
                        }
                    };

                    ctx.add(arg, arg_grad)?;
                }
                LazyOp::Reindex(Reindex::Permute(Permute { src: arg, dims })) => {
                    let mut inv_dims = rvec![0; dims.len()];
                    for (i, &dim) in dims.iter().enumerate() {
                        inv_dims[dim] = i;
                    }
                    let arg_grad = grad.permute(inv_dims)?;
                    ctx.add(arg, arg_grad)?;
                }
                LazyOp::Reduce(Reduce {
                    input: arg,
                    reduced_shape,
                    op: ReduceOp::Sum,
                    ..
                }) => {
                    let grad = broadcast_back(arg, &grad, reduced_shape.inner())?;
                    ctx.add(arg, grad)?;
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
                    ctx.add(arg, grad.broadcast_to(arg.shape().clone())?)?;
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
                    ctx.add(arg, grad.broadcast_to(arg.shape().clone())?)?;
                }
                LazyOp::Unary(Unary {
                    input: arg,
                    op: UnaryOp::Log,
                }) => {
                    let arg_grad = (grad / arg.clone())?;
                    ctx.add(arg, arg_grad)?;
                }
                LazyOp::Unary(Unary {
                    input: arg,
                    op: UnaryOp::Sin,
                }) => {
                    let arg_grad = (grad * arg.clone().cos())?;
                    ctx.add(arg, arg_grad)?;
                }
                LazyOp::Unary(Unary {
                    input: arg,
                    op: UnaryOp::Cos,
                }) => {
                    let arg_grad = (grad * arg.clone().sin())?;
                    ctx.sub(arg, arg_grad)?;
                }
                LazyOp::Unary(Unary {
                    input: arg,
                    op: UnaryOp::Tanh,
                }) => {
                    let minus_dtanh = ((*node).clone().square()? - 1.)?;
                    let arg_grad = (grad.clone() * minus_dtanh)?;
                    ctx.sub(arg, arg_grad)?;
                }
                LazyOp::Unary(Unary {
                    input: arg,
                    op: UnaryOp::Abs,
                }) => {
                    let ones = arg.ones_like::<f32>(None, false)?;
                    let abs_grad = arg
                        .clone()
                        .ge(arg.clone().zeros_like::<f32>(None, false)?)?
                        .where_cond(ones.clone(), ones.neg()?)?;
                    let arg_grad = (grad * abs_grad)?;
                    ctx.add(arg, arg_grad)?;
                }
                LazyOp::Unary(Unary {
                    input: arg,
                    op: UnaryOp::Exp,
                }) => {
                    let arg_grad = (grad * (*node).clone())?;
                    ctx.add(arg, arg_grad)?;
                }
                LazyOp::Unary(Unary {
                    input: arg,
                    op: UnaryOp::Neg,
                }) => {
                    ctx.sub(arg, grad)?;
                }
                LazyOp::Unary(Unary {
                    input: arg,
                    op: UnaryOp::Reciprocal,
                }) => {
                    let arg_grad = (grad / arg.clone().square()?)?;
                    ctx.sub(arg, arg_grad)?;
                }
                LazyOp::Unary(Unary {
                    input: _,
                    op: UnaryOp::Ceil,
                }) => Err(BackpropError::BackwardNotSupported { op: "ceil" })?,
                LazyOp::Unary(Unary {
                    input: arg,
                    op: UnaryOp::Gelu,
                }) => {
                    let cube = arg.clone().pow(3.)?;
                    let tanh = (0.0356774 * cube.clone() + (0.797885 * arg.clone())?)?.tanh()?;
                    let gelu_grad = (((0.5 * tanh.clone())?
                        + (0.0535161 * cube + (0.398942 * arg.clone())?)?
                            * (1. - tanh.clone().pow(2.)?))?
                        + 0.5)?;
                    let arg_grad = (grad * gelu_grad)?;
                    ctx.add(arg, arg_grad)?;
                }
                LazyOp::Unary(Unary {
                    input: arg,
                    op: UnaryOp::Relu,
                }) => {
                    let relu_grad = arg
                        .clone()
                        .ge(arg.clone().zeros_like::<f32>(None, false)?)?
                        .cast(arg.dtype())?;
                    let arg_grad = grad.mul(relu_grad)?;
                    ctx.add(arg, arg_grad)?;
                }
                LazyOp::Unary(Unary {
                    input: arg,
                    op: UnaryOp::Relu2,
                }) => {
                    let relu_grad = arg.clone().affine(2.0, 0.0)?.mul(
                        arg.clone()
                            .ge(arg.clone().zeros_like::<f32>(None, false)?)?
                            .cast(arg.dtype())?,
                    )?;
                    let arg_grad = grad.mul(relu_grad)?;
                    ctx.add(arg, arg_grad)?;
                }
                LazyOp::Unary(Unary {
                    input: arg,
                    op: UnaryOp::Silu,
                }) => {
                    let sigmoid_arg = (arg.clone().neg()?.exp()? + 1.)?.recip()?;
                    let silu_grad =
                        (sigmoid_arg.clone() * (1. + (arg.clone() * (1. - sigmoid_arg)?)?)?)?;
                    let arg_grad = grad.mul(silu_grad)?;
                    ctx.add(arg, arg_grad)?;
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

                    ctx.add(arg, arg_grad)?;
                }
                LazyOp::Unary(Unary {
                    input: arg,
                    op: UnaryOp::Square,
                }) => {
                    let arg_grad = arg.clone().mul(grad)?.affine(2., 0.)?;
                    ctx.add(arg, arg_grad)?;
                }
                LazyOp::Unary(Unary {
                    input: arg,
                    op: UnaryOp::Sqrt,
                }) => {
                    let arg_grad = grad.div((*node).clone())?.affine(0.5, 0.)?;
                    ctx.add(arg, arg_grad)?;
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
                    ctx.add(arg, arg_grad)?;
                }
                LazyOp::Select(IndexSelect {
                    src: arg,
                    indices,
                    dim,
                }) => {
                    if ctx.tracked.contains(&arg.id()) {
                        let sum_grad = or_insert(arg)?;
                        arg.set_grad(Some(sum_grad.clone().index_add(
                            indices.clone(),
                            grad.clone(),
                            *dim,
                        )?));
                    }
                }
                LazyOp::Softmax(Softmax { input: arg, dim }) => {
                    // Get the softmax output (s)
                    let softmax_output = (*node).clone();

                    // Compute the sum of the gradients
                    let sum_grad = grad.clone().sum_keepdim(*dim)?;

                    // Compute the gradient with respect to the softmax input
                    let input_grad = softmax_output
                        .clone()
                        .mul(grad.clone())?
                        .sub(softmax_output.clone().mul(sum_grad)?)?;

                    ctx.add(arg, input_grad)?;
                }
                LazyOp::Norm(NormOp::LayerNorm(Norm {
                    input: arg,
                    scale,
                    bias,
                    eps,
                })) => {
                    // TODO(vinhowe): The following is an AI-generated celebration of laziness,
                    // and it requires many, many backward ops for a single forward op. This should
                    // instead be implemented in a single backward kernel, and its implementation
                    // should be understood by its author (the human, not Gemini 2.5 Pro).
                    let rank = arg.dim();
                    let norm_axis = rank - 1;
                    let d = arg.shape()[norm_axis] as f32;

                    // Determine axes to reduce over for gamma/beta grads (all except norm_axis)
                    let mut sum_axes: Vec<usize> = (0..rank).collect();
                    sum_axes.remove(norm_axis);

                    // Recompute intermediate values using the correct normalization axis
                    // Ideally, these should be cached from the forward pass
                    let mean = arg
                        .clone()
                        // Keepdim for broadcasting
                        .sum_keepdim(norm_axis)?
                        .affine(1. / d, 0.)?;
                    let var = arg
                        .clone()
                        .sub(mean.clone())?
                        .square()?
                        .sum_keepdim(norm_axis)?
                        .affine(1. / d, 0.)?; // Keepdim for broadcasting

                    let std = (var.clone() + *eps)?.sqrt()?;
                    let x_normed = arg.clone().sub(mean)?.div(std.clone())?;

                    // Compute gradients w.r.t scale (gamma) and bias (beta)
                    let grad_gamma = x_normed
                        .clone()
                        .mul(grad.clone())?
                        .sum_keepdim(sum_axes.as_slice())?;
                    ctx.add(scale, grad_gamma.squeeze_all()?)?;

                    if let Some(bias) = bias {
                        let grad_beta = grad.clone().sum_keepdim(sum_axes.as_slice())?;
                        ctx.add(bias, grad_beta.squeeze_all()?)?;
                    }

                    // Compute gradient w.r.t normalized input
                    let grad_x_normed = grad.clone().mul(scale.clone())?;

                    // Compute gradients w.r.t mean and variance (using correct reduction axis)
                    // dL/dmu = sum(dL/dx_normed * (-1/std)) over norm_axis
                    let grad_mean = grad_x_normed
                        .clone()
                        .sum_keepdim(norm_axis)?
                        .neg()?
                        .div(std.clone())?;

                    // dL/dVar = sum(dL/dx_normed * (-x_normed)) / (2 * std^2) over norm_axis
                    let grad_var = grad_x_normed
                        .clone()
                        .mul(x_normed.clone())?
                        .sum_keepdim(norm_axis)?
                        .neg()?
                        .div(var.clone().affine(1., *eps)?)? // std^2 = var + eps
                        .affine(0.5, 0.)?;

                    // Compute gradient w.r.t input x using the chain rule:
                    // dL/dx = (dL/dx_normed / std) + (dL/dvar * dvar/dx) + (dL/dmu * dmu/dx)
                    // dvar/dx = (2/N) * (x - mu)
                    // dmu/dx = 1/N
                    let grad_x = grad_x_normed
                        .div(std.clone())? // (dL/dx_normed / std)
                        .add(
                            grad_var
                                .mul(x_normed.clone().mul(std)?)?
                                .affine(2. / d, 0.)?,
                        )?
                        // dL/dvar * (2/N) * (x - mu) = dL/dvar * (2/N) * x_normed * std
                        .add(grad_mean.affine(1. / d, 0.)?)?; // dL/dmu * (1/N)

                    ctx.add(arg, grad_x)?;
                }
                LazyOp::Affine(Affine { src: arg, mul, .. }) => {
                    let arg_grad = grad.affine(*mul, 0.)?;
                    ctx.add(arg, arg_grad)?;
                }
                LazyOp::Gather(Gather { src, ids, dim, .. }) => {
                    // We can't use or_insert here because we need to scatter into a zero tensor.
                    let sum_grad = src.zeros_like::<f32>(None, false)?;
                    let src_grad = sum_grad.scatter_add(ids.clone(), grad.clone(), *dim)?;
                    ctx.add(src, src_grad)?;
                }
                LazyOp::ScatterAdd(ScatterAdd { dst, src, ids, dim }) => {
                    ctx.add(dst, grad.clone())?;
                    let src_grad = grad.gather(ids.clone(), *dim)?;
                    ctx.add(src, src_grad)?;
                }
                LazyOp::Trilu(Trilu { src: arg, upper, k }) => {
                    let masked_grad = if *upper {
                        grad.triu(*k)?
                    } else {
                        grad.tril(*k)?
                    };
                    ctx.add(arg, masked_grad)?;
                }
                LazyOp::Alibi(Alibi { input, .. }) => {
                    ctx.add(input, grad)?;
                }
                LazyOp::Cast(Cast {
                    input,
                    dst_dtype: _,
                }) => {
                    ctx.add(input, grad.cast(input.dtype())?)?;
                }
                LazyOp::Norm(_) => todo!(),
                LazyOp::Const => panic!("piston internal error - const node in backprop"),
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
                    ctx.add(arg, arg_grad)?;
                }
                LazyOp::Conv(_) => todo!(),
                LazyOp::IndexWrite(_) => todo!(),
                LazyOp::IndexAdd(_) => todo!(),
                LazyOp::Cache(_) => todo!(),
                LazyOp::Copy(_) => todo!(),
            };
        }
        Ok(())
    }
}

impl Tensor {
    pub fn backward(&self) -> Result<()> {
        self.inner().read().backward()
    }
}
