// TODO(vinhowe): This cursed approach to reference handling partly reflects the tension between
// Rust's convenient reference counting approach to memory management and JS's automatic garbage
// collection, but there is probably a more thoughtful approach to this. Hard to justify the energy
// right now, though.

use crate::error::IntoJsError;
use crate::js_util::downcast_from_ptr;
use crate::shape::FromJsDim;
use half::f16;
use js_sys::{Array, Function, Object, Reflect};
use parking_lot::RwLock;
use piston::{
    AllDims, DType, Dim, IrScalarValue, IrValue, LazyOp, NormOrd, Storage, Tensor, TensorId,
    TensorTypeOrScalar, TensorTypeOrScalarEnum,
};
use piston_macros::js_tensor_web_op;
use std::cell::RefCell;
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::{Arc, Weak};
use wasm_bindgen::prelude::*;
use wasm_bindgen::{JsError, JsValue};

use crate::{device::JsDevice, dtype::JsDType};

enum MaybeStrong<T> {
    Strong(Arc<T>),
    Weak(Weak<T>),
}

impl<T> MaybeStrong<T> {
    pub fn downgrade(&self) -> Weak<T> {
        match self {
            MaybeStrong::Strong(a) => Arc::downgrade(a),
            MaybeStrong::Weak(w) => w.clone(),
        }
    }

    pub fn upgrade(&self) -> Option<Arc<T>> {
        match self {
            MaybeStrong::Strong(a) => Some(a.clone()),
            MaybeStrong::Weak(w) => w.upgrade(),
        }
    }
}

#[wasm_bindgen(js_name = Tensor)]
pub struct JsTensor {
    inner: MaybeStrong<(RwLock<Option<Tensor>>, StrongJsTensorId)>,
}

impl JsTensor {
    fn new_impl(inner: MaybeStrong<(RwLock<Option<Tensor>>, StrongJsTensorId)>) -> Self {
        if let MaybeStrong::Strong(ref inner) = inner {
            register_active_tensor(JsTensor::new_impl(MaybeStrong::Weak(Arc::downgrade(inner))));
        }
        Self { inner }
    }

    pub fn new(inner: Tensor) -> Self {
        Self::new_impl(MaybeStrong::Strong(Arc::new((
            RwLock::new(Some(inner)),
            StrongJsTensorId::new(),
        ))))
    }

    pub fn new_weak(inner: Weak<(RwLock<Option<Tensor>>, StrongJsTensorId)>) -> Self {
        Self::new_impl(MaybeStrong::Weak(inner))
    }

    pub(crate) fn inner(&self) -> Tensor {
        match self.inner {
            MaybeStrong::Strong(ref inner) => inner
                .0
                .read()
                .as_ref()
                .expect("Tried to use a dropped Tensor; strong inner value taken")
                .clone(),
            MaybeStrong::Weak(ref inner) => inner
                .upgrade()
                .as_ref()
                .expect("Tried to use a dropped Tensor; ref dropped")
                .0
                .read()
                .as_ref()
                .expect("Tried to use a dropped Tensor; weak inner value taken")
                .clone(),
        }
    }

    pub(crate) fn weak(&self) -> JsTensor {
        JsTensor::new_weak(self.inner.downgrade())
    }

    fn js_value(&self) -> JsValue {
        self.weak().into()
    }
}

#[wasm_bindgen(js_class = Tensor)]
impl JsTensor {
    // Marker function for downcasting from a JS object
    #[wasm_bindgen(unchecked_prelude = "__wbg_piston_tensor(): void;")]
    pub fn __wbg_piston_tensor() {}

    #[wasm_bindgen(js_name = new, unchecked_return_type = "Tensor")]
    pub async fn new_js(
        #[wasm_bindgen(
            unchecked_param_type = "Float32Array | Float64Array | Int32Array | Uint32Array | number[]"
        )]
        data: &JsValue,
        #[wasm_bindgen(unchecked_param_type = "Uint32Array | number[]")] value: &JsValue,
        #[wasm_bindgen(unchecked_param_type = "FromDataOptions")] options: &JsValue,
    ) -> Result<JsValue, JsError> {
        fromData(data, value, options)
    }

    #[wasm_bindgen(js_name = __pistonDrop)]
    pub fn __piston_drop(&self) {
        if let Some(inner) = self.inner.upgrade() {
            inner.0.write().take();
        }
    }

    #[wasm_bindgen(getter = __pistonHasValue)]
    pub fn __piston_has_value(&self) -> bool {
        self.inner
            .upgrade()
            .map(|inner| inner.0.read().is_some())
            .unwrap_or(false)
    }

    #[wasm_bindgen(getter = __pistonStrongId)]
    pub fn __piston_strong_id(&self) -> usize {
        self.inner.upgrade().map(|inner| inner.1.0).unwrap_or(0)
    }

    #[wasm_bindgen(getter)]
    pub fn id(&self) -> usize {
        self.inner().id().0
    }

    #[wasm_bindgen(getter = __pistonStrongCount)]
    pub fn __piston_strong_count(&self) -> usize {
        self.inner().strong_count()
    }

    // - Skipping storage_view

    pub fn dim(&self) -> usize {
        self.inner().dim()
    }

    #[wasm_bindgen(getter)]
    pub fn ndim(&self) -> usize {
        self.inner().dim()
    }

    #[wasm_bindgen(getter)]
    pub fn dtype(&self) -> JsDType {
        JsDType {
            dtype: self.inner().dtype(),
        }
    }

    #[wasm_bindgen(
        unchecked_return_type = "number[] | number",
        unchecked_prelude = "size(): number[];\nsize(dim: number): number;"
    )]
    pub fn size(&self, dim: Option<isize>) -> JsValue {
        let inner = self.inner();
        if let Some(dim) = dim {
            let dim = FromJsDim(dim);
            let size = dim.to_index(&inner.shape(), "size").unwrap();
            JsValue::from_f64(inner.shape()[size] as f64)
        } else {
            let shape = inner.shape().to_vec();
            let array = js_sys::Array::new();
            for dim in shape {
                array.push(&JsValue::from_f64(dim as f64));
            }
            array.into()
        }
    }

    #[wasm_bindgen(getter, unchecked_return_type = "number[]")]
    pub fn shape(&self) -> JsValue {
        let shape = self.inner().shape().to_vec();
        let array = js_sys::Array::new();
        for dim in shape {
            array.push(&JsValue::from_f64(dim as f64));
        }
        array.into()
    }

    #[wasm_bindgen(
        unchecked_return_type = "number[] | number",
        unchecked_prelude = "stride(): number[];\nstride(dim: number): number;"
    )]
    pub fn stride(&self, dim: Option<isize>) -> JsValue {
        let inner = self.inner();
        if let Some(dim) = dim {
            let dim = FromJsDim(dim);
            let stride = dim.to_index(&inner.shape(), "stride").unwrap();
            JsValue::from_f64(inner.stride()[stride] as f64)
        } else {
            let stride = inner.stride().to_vec();
            let array = js_sys::Array::new();
            for dim in stride {
                array.push(&JsValue::from_f64(dim as f64));
            }
            array.into()
        }
    }

    #[wasm_bindgen(getter = nbytes)]
    pub fn num_bytes(&self) -> usize {
        self.inner().num_bytes()
    }

    #[wasm_bindgen(getter)]
    pub fn device(&self) -> JsDevice {
        JsDevice {
            inner: self.inner().device(),
        }
    }

    // - Skipping storage

    pub fn resolved(&self) -> bool {
        self.inner().resolved()
    }

    pub fn op(&self) -> JsValue {
        let op = self.inner().op();
        let ir = op.ir();
        let name = ir.name().to_string();

        let obj = js_sys::Object::new();
        js_sys::Reflect::set(&obj, &"name".into(), &JsValue::from_str(&name)).unwrap();

        if let Some(ir_fields) = ir.fields() {
            js_sys::Reflect::set(
                &obj,
                &"fields".into(),
                &convert_ir_fields_to_js(ir_fields, &op),
            )
            .unwrap();
        }

        obj.into()
    }

    // Convenient for building graphs
    #[wasm_bindgen(unchecked_return_type = "number[]", js_name = srcIds)]
    pub fn src_ids(&self) -> JsValue {
        let src_ids = self
            .inner()
            .op()
            .srcs()
            .iter()
            .map(|s| s.id().0)
            .collect::<Vec<_>>();
        let array = js_sys::Array::new();
        for id in src_ids {
            array.push(&JsValue::from_f64(id as f64));
        }
        array.into()
    }

    pub fn scope(&self) -> Option<String> {
        self.inner().scope().clone()
    }

    #[wasm_bindgen(js_name = isScalar)]
    pub fn is_scalar(&self) -> bool {
        self.inner().is_scalar()
    }

    #[wasm_bindgen(getter = requiresGrad)]
    pub fn requires_grad(&self) -> bool {
        self.inner().requires_grad()
    }

    #[wasm_bindgen(getter = isLeaf)]
    pub fn is_leaf(&self) -> bool {
        self.inner().is_leaf()
    }

    #[wasm_bindgen(getter = retainsGrad)]
    pub fn retains_grad(&self) -> bool {
        self.inner().retains_grad()
    }

    #[wasm_bindgen(js_name = retainGrad)]
    pub fn retain_grad(&self) -> Result<(), JsError> {
        self.inner().retain_grad().map_err(|e| e.into_js_error())
    }

    pub fn is_contiguous(&self) -> bool {
        self.inner().is_contiguous()
    }
}

macro_rules! impl_binary_op {
    ($op:ident, $Name:ident) => {
        #[js_tensor_web_op(name = $Name, variants = [method, method_inplace, function])]
        pub fn $op(input: Tensor, other: TensorOrScalar) -> anyhow::Result<Tensor> {}
    };
}

macro_rules! impl_binary_op_tensor_only {
    ($op:ident, $Name:ident) => {
        #[js_tensor_web_op(name = $Name, variants = [method, method_inplace, function])]
        pub fn $op(input: Tensor, other: Tensor) -> anyhow::Result<Tensor> {}
    };
}

macro_rules! impl_ternary_op {
    ($op:ident, $Name:ident) => {
        #[js_tensor_web_op(name = $Name, variants = [method, method_inplace, function])]
        pub fn $op(
            input: Tensor,
            tensor1: Tensor,
            tensor2: Tensor,
            value: f32,
        ) -> anyhow::Result<Tensor> {
        }
    };
}

macro_rules! impl_cmp_op {
    ($op:ident, $Name:ident) => {
        #[js_tensor_web_op(name = $Name, variants = [method, method_inplace, function])]
        pub fn $op(input: Tensor, other: TensorOrScalar) -> anyhow::Result<Tensor> {}
    };
}

macro_rules! impl_unary_op {
    ($op:ident, $Name:ident) => {
        #[js_tensor_web_op(name = $Name, variants = [method, method_inplace, function])]
        pub fn $op(input: Tensor) -> anyhow::Result<Tensor> {}
    };
}

impl_binary_op!(add, Add);
impl_binary_op!(sub, Sub);
impl_binary_op!(mul, Mul);
impl_binary_op!(div, Div);
impl_binary_op!(pow, Pow);
impl_binary_op_tensor_only!(minimum, Minimum);
impl_binary_op_tensor_only!(maximum, Maximum);

impl_ternary_op!(addcdiv, Addcdiv);
impl_ternary_op!(addcmul, Addcmul);

impl_cmp_op!(eq, Eq);
impl_cmp_op!(ne, Ne);
impl_cmp_op!(gt, Gt);
impl_cmp_op!(ge, Ge);
impl_cmp_op!(lt, Lt);
impl_cmp_op!(le, Le);
impl_cmp_op!(logical_and, LogicalAnd);
impl_cmp_op!(logical_or, LogicalOr);
impl_cmp_op!(logical_xor, LogicalXor);

impl_unary_op!(gelu, Gelu);
impl_unary_op!(tanh, Tanh);
impl_unary_op!(exp, Exp);
impl_unary_op!(log, Log);
impl_unary_op!(sin, Sin);
impl_unary_op!(cos, Cos);
impl_unary_op!(abs, Abs);
impl_unary_op!(sqrt, Sqrt);
impl_unary_op!(relu, Relu);
impl_unary_op!(relu2, Relu2);
impl_unary_op!(floor, Floor);
impl_unary_op!(ceil, Ceil);
impl_unary_op!(neg, Neg);
impl_unary_op!(sigmoid, Sigmoid);
impl_unary_op!(swiglu, Swiglu);
impl_unary_op!(silu, Silu);
impl_unary_op!(square, Square);
impl_unary_op!(recip, Recip);
impl_unary_op!(logical_not, LogicalNot);
impl_unary_op!(isnan, IsNan);
impl_unary_op!(isinf, IsInf);

#[js_tensor_web_op(name = Full, variants = [function])]
pub fn full(shape: Shape, value: f32, options: TensorOptions) -> JsTensorResult {
    piston::full(shape, value, options)
}

#[js_tensor_web_op(name = Cast, variants = [method])]
pub fn cast(input: Tensor, dtype: DType) -> JsTensorResult {
    piston::cast(input, dtype)
}

#[js_tensor_web_op(name = Float, variants = [method])]
pub fn float(input: Tensor) -> JsTensorResult {
    piston::cast(input, DType::F32)
}

#[js_tensor_web_op(name = Half, variants = [method])]
pub fn half(input: Tensor) -> JsTensorResult {
    piston::cast(input, DType::F16)
}

#[js_tensor_web_op(name = GroupNorm, variants = [method])]
pub fn group_norm(
    input: Tensor,
    num_groups: usize,
    weight: Option<Tensor>,
    bias: Option<Tensor>,
    eps: f32,
) -> JsTensorResult {
}

#[js_tensor_web_op(name = LayerNorm, variants = [method])]
pub fn layer_norm(
    input: Tensor,
    weight: Option<Tensor>,
    bias: Option<Tensor>,
    eps: f32,
) -> JsTensorResult {
}

#[js_tensor_web_op(name = RmsNorm, variants = [method])]
pub fn rms_norm(
    input: Tensor,
    weight: Option<Tensor>,
    #[op(default = 1e-5)] eps: f32,
) -> JsTensorResult {
}

#[js_tensor_web_op(name = Conv1d, variants = [method])]
pub fn conv1d(
    input: Tensor,
    weight: Tensor,
    bias: Option<Tensor>,
    #[op(default = 1)] stride: usize,
    #[op(default = 0)] padding: usize,
) -> JsTensorResult {
}

#[js_tensor_web_op(name = Softmax, variants = [method])]
pub fn softmax(input: Tensor, dim: Dim) -> JsTensorResult {}

#[js_tensor_web_op(name = Rope, variants = [method_inplace])]
pub fn rope(input: Tensor, dim: usize, base: f32, offset: usize) -> JsTensorResult {}

#[js_tensor_web_op(name = Alibi, variants = [method])]
pub fn alibi(input: Tensor, #[op(default = 8.0)] max_bias: f32) -> JsTensorResult {}

#[js_tensor_web_op(name = Matmul, variants = [method])]
pub fn matmul(
    input: Tensor,
    rhs: Tensor,
    #[op(default = false)] trans_lhs: bool,
    #[op(default = false)] trans_rhs: bool,
) -> JsTensorResult {
}

#[js_tensor_web_op(name = Gemm, variants = [method])]
pub fn gemm(
    input: Tensor,
    rhs: Tensor,
    bias: Option<Tensor>,
    #[op(default = false)] trans_lhs: bool,
    #[op(default = false)] trans_rhs: bool,
    #[op(default = false)] trans_out: bool,
) -> JsTensorResult {
}

#[js_tensor_web_op(name = Affine, variants = [method])]
pub fn affine(
    input: Tensor,
    #[op(default = 1.0)] mul: f32,
    #[op(default = 0.0)] add: f32,
) -> JsTensorResult {
}

#[js_tensor_web_op(name = Sum, variants = [method])]
pub fn sum(
    input: Tensor,
    dim: Option<Dims>,
    #[op(default = false)] keepdim: bool,
) -> JsTensorResult {
    if let Some(sum_dims) = dim {
        input.sum(sum_dims, keepdim)
    } else {
        input.sum(AllDims, keepdim)
    }
}

#[js_tensor_web_op(name = Mean, variants = [method])]
pub fn mean(
    input: Tensor,
    dim: Option<Dims>,
    #[op(default = false)] keepdim: bool,
) -> JsTensorResult {
    if let Some(mean_dims) = dim {
        input.mean(mean_dims, keepdim)
    } else {
        input.mean(AllDims, keepdim)
    }
}

#[js_tensor_web_op(name = Var, variants = [method])]
pub fn var(
    input: Tensor,
    dim: Option<Dims>,
    #[op(default = false)] keepdim: bool,
) -> JsTensorResult {
    if let Some(var_dims) = dim {
        input.var(var_dims, keepdim)
    } else {
        input.var(AllDims, keepdim)
    }
}

#[js_tensor_web_op(name = Max, variants = [method])]
pub fn max(
    input: Tensor,
    dim: Option<Dims>,
    #[op(default = false)] keepdim: bool,
) -> JsTensorResult {
    if let Some(max_dims) = dim {
        input.max(max_dims, keepdim)
    } else {
        input.max(AllDims, keepdim)
    }
}

#[js_tensor_web_op(name = Min, variants = [method])]
pub fn min(
    input: Tensor,
    dim: Option<Dims>,
    #[op(default = false)] keepdim: bool,
) -> JsTensorResult {
    if let Some(min_dims) = dim {
        input.min(min_dims, keepdim)
    } else {
        input.min(AllDims, keepdim)
    }
}

#[js_tensor_web_op(name = Argmax, variants = [method])]
pub fn argmax(
    input: Tensor,
    dim: Option<Dims>,
    #[op(default = false)] keepdim: bool,
) -> JsTensorResult {
    if let Some(argmax_dims) = dim {
        input.argmax(argmax_dims, keepdim)
    } else {
        input.argmax(AllDims, keepdim)
    }
}

#[js_tensor_web_op(name = Argmin, variants = [method])]
pub fn argmin(
    input: Tensor,
    dim: Option<Dims>,
    #[op(default = false)] keepdim: bool,
) -> JsTensorResult {
    if let Some(argmin_dims) = dim {
        input.argmin(argmin_dims, keepdim)
    } else {
        input.argmin(AllDims, keepdim)
    }
}

#[js_tensor_web_op(name = Norm, variants = [method])]
pub fn norm(
    input: Tensor,
    ord: Option<NormOrd>,
    dim: Option<Dims>,
    #[op(default = false)] keepdim: bool,
) -> JsTensorResult {
    if let Some(norm_dims) = dim {
        input.norm(ord, norm_dims, keepdim)
    } else {
        input.norm(ord, AllDims, keepdim)
    }
}

#[js_tensor_web_op(name = Flatten, variants = [method, function])]
pub fn flatten(
    input: Tensor,
    #[op(default = 0)] start_dim: Dim,
    #[op(default = -1)] end_dim: Dim,
) -> JsTensorResult {
}

#[js_tensor_web_op(name = Slice, variants = [method])]
pub fn slice(
    input: Tensor,
    #[op(unchecked_type = "[start: number, end: number][]")] ranges: JsValue,
) -> JsTensorResult {
    let ranges = ranges
        .dyn_into::<js_sys::Array>()
        .expect("Ranges must be an array")
        .iter()
        .map(|r| {
            let range: js_sys::Array = r.clone().into();
            let start = range.get(0).as_f64().map(|v| v as usize);
            let end = range.get(1).as_f64().map(|v| v as usize);
            if let Some(end) = end {
                start.expect("Invalid range")..end
            } else {
                0usize..start.expect("Invalid range")
            }
        })
        .collect::<Vec<_>>();
    input.slice(&ranges)
}

#[js_tensor_web_op(name = Split, variants = [method, function])]
pub fn split(
    input: Tensor,
    #[op(unchecked_type = "number | number[]", name = "splitSizeOrSections")]
    split_size_or_sections: JsValue,
    #[op(default = 0)] dim: Dim,
) -> Result<Vec<Tensor>, JsError> {
    let arg = if let Some(n) = split_size_or_sections.as_f64() {
        piston::SplitArg::SplitSize(n as usize)
    } else if split_size_or_sections.is_object() && split_size_or_sections.is_array() {
        let arr = split_size_or_sections
            .dyn_into::<js_sys::Array>()
            .map_err(|_| JsError::new("split sections must be an array"))?;
        let sizes = arr
            .iter()
            .map(|v| {
                v.as_f64()
                    .map(|f| f as usize)
                    .ok_or_else(|| JsError::new("sections must be numbers"))
            })
            .collect::<Result<Vec<_>, JsError>>()?;
        piston::SplitArg::Sizes(sizes.into())
    } else {
        return Err(JsError::new(
            "split requires a number or an array of numbers",
        ));
    };
    let parts = piston::split(input, arg, dim).map_err(|e| e.into_js_error())?;
    Ok::<_, JsError>(parts.into_iter().collect::<Vec<_>>())
}

#[js_tensor_web_op(name = Chunk, variants = [method, function])]
pub fn chunk(
    input: Tensor,
    chunks: usize,
    #[op(default = 0)] dim: Dim,
) -> Result<Vec<Tensor>, JsError> {
    let parts = piston::chunk(input, chunks, dim).map_err(|e| e.into_js_error())?;
    Ok::<_, JsError>(parts.into_iter().collect::<Vec<_>>())
}

#[js_tensor_web_op(name = View, variants = [method])]
pub fn view(input: Tensor, shape: ShapeWithOneHole) -> JsTensorResult {}

#[js_tensor_web_op(name = Unsqueeze, variants = [method, function])]
pub fn unsqueeze(input: Tensor, dim: Dim) -> JsTensorResult {}

#[js_tensor_web_op(name = Squeeze, variants = [method, function])]
pub fn squeeze(input: Tensor, dim: Option<Dims>) -> JsTensorResult {
    match dim {
        Some(dims) => input.squeeze(dims),
        None => input.squeeze(()),
    }
}

#[js_tensor_web_op(name = Permute, variants = [method, function])]
pub fn permute(input: Tensor, dims: Dims) -> JsTensorResult {}

#[js_tensor_web_op(name = Flip, variants = [method, function])]
pub fn flip(input: Tensor, dims: Dims) -> JsTensorResult {}

#[js_tensor_web_op(name = Transpose, variants = [method, function])]
pub fn transpose(input: Tensor, dim0: Dim, dim1: Dim) -> JsTensorResult {}

#[js_tensor_web_op(name = t, variants = [method, function])]
pub fn t(input: Tensor) -> JsTensorResult {}

#[js_tensor_web_op(getter, name = TUpper, variants = [method], target = T)]
#[allow(non_snake_case)]
pub fn T_upper(input: Tensor) -> JsTensorResult {}

#[js_tensor_web_op(getter, name = mT, variants = [method])]
#[allow(non_snake_case)]
pub fn mT(input: Tensor) -> JsTensorResult {}

#[js_tensor_web_op(name = Cache, variants = [method])]
pub fn cache(input: Tensor, source: Tensor, dim: Dim, offset: usize) -> JsTensorResult {}

#[js_tensor_web_op(name = BroadcastLeft, variants = [method])]
pub fn broadcast_left(input: Tensor, left_shape: Shape) -> JsTensorResult {}

#[js_tensor_web_op(name = BroadcastTo, variants = [method])]
pub fn broadcast_to(input: Tensor, shape: Shape) -> JsTensorResult {}

#[js_tensor_web_op(name = IndexSelect, variants = [method])]
pub fn index_select(input: Tensor, indices: Tensor, dim: Dim) -> JsTensorResult {}

#[js_tensor_web_op(name = IndexWrite, variants = [method])]
pub fn index_write(input: Tensor, src: Tensor, write_start: Dims) -> JsTensorResult {}

#[js_tensor_web_op(name = Where, variants = [method, function], js_name = "where")]
pub fn where_cond(input: Tensor, condition: Tensor, on_false: TensorOrScalar) -> JsTensorResult {}

#[js_tensor_web_op(name = Clamp, variants = [method, method_inplace, function])]
pub fn clamp(
    input: Tensor,
    min: Option<TensorOrScalar>,
    max: Option<TensorOrScalar>,
) -> JsTensorResult {
}

#[js_tensor_web_op(name = ScatterAdd, variants = [method, function])]
pub fn scatter_add(input: Tensor, indices: Tensor, source: Tensor, dim: Dim) -> JsTensorResult {}

#[js_tensor_web_op(name = IndexAdd, variants = [method_inplace, function])]
pub fn index_add(input: Tensor, indices: Tensor, source: Tensor, dim: Dim) -> JsTensorResult {}

#[js_tensor_web_op(name = Gather, variants = [method, function])]
pub fn gather(input: Tensor, dim: Dim, index: Tensor) -> JsTensorResult {}

#[js_tensor_web_op(name = Triu, variants = [method, method_inplace, function])]
pub fn triu(input: Tensor, k: Option<i32>) -> JsTensorResult {}

#[js_tensor_web_op(name = Tril, variants = [method, method_inplace, function])]
pub fn tril(input: Tensor, k: Option<i32>) -> JsTensorResult {}

#[js_tensor_web_op(name = Lerp, variants = [method, method_inplace, function])]
pub fn lerp(input: Tensor, end: Tensor, weight: TensorOrScalar) -> JsTensorResult {}

#[js_tensor_web_op(name = Bernoulli, variants = [function, method, method_inplace])]
pub fn bernoulli(input: Tensor) -> JsTensorResult {}

#[js_tensor_web_op(name = Multinomial, variants = [method, function])]
pub fn multinomial(
    input: Tensor,
    #[op(name = "numSamples")] num_samples: usize,
    #[op(default = false)] replacement: bool,
) -> JsTensorResult {
}

#[js_tensor_web_op(name = Topk, variants = [method, function])]
pub fn topk(
    input: Tensor,
    k: usize,
    #[op(default = -1)] dim: Dim,
    #[op(default = true)] largest: bool,
    #[op(default = false)] sorted: bool,
) -> Result<Vec<Tensor>, JsError> {
    if sorted {
        return Err(JsError::new("topk: sorted=true not implemented"));
    }
    let result = piston::topk(input, k, Some(dim), Some(largest), Some(sorted))
        .map_err(|e| e.into_js_error())?;
    Ok::<_, JsError>(result.into_iter().collect::<Vec<_>>())
}

#[js_tensor_web_op(name = Zero, variants = [method_inplace])]
pub fn zero(input: Tensor) -> JsTensorResult {}

#[js_tensor_web_op(name = ZerosLike, variants = [function])]
pub fn zeros_like(input: Tensor, options: TensorOptions) -> JsTensorResult {}

#[js_tensor_web_op(name = OnesLike, variants = [function])]
pub fn ones_like(input: Tensor, options: TensorOptions) -> JsTensorResult {}

#[js_tensor_web_op(name = Contiguous, variants = [method])]
pub fn contiguous(input: Tensor) -> JsTensorResult {}

#[js_tensor_web_op(name = RequiresGrad, variants = [method_inplace])]
pub fn requires_grad(input: Tensor, requires_grad: bool) -> JsTensorResult {}

#[js_tensor_web_op(name = FromData, variants = [function])]
pub fn from_data(
    #[op(
        unchecked_type = "Float32Array | Float16Array | Float64Array | Int32Array | Uint32Array | Uint8Array | number[]"
    )]
    data: JsValue,
    shape: Shape,
    options: TensorOptions,
) -> JsTensorResult {
    if data.is_array() {
        let array = data
            .dyn_into::<js_sys::Array>()
            .map_err(|_| JsError::new("Failed to convert data to array"))?;

        let data = array
            .iter()
            .map(|v| {
                v.as_f64()
                    .map(|f| f as f32)
                    .ok_or_else(|| JsError::new("Array contains non-numeric values"))
            })
            .collect::<Result<Vec<f32>, JsError>>()?;

        Tensor::from_data(data, shape, options)
    } else if js_sys::Float32Array::instanceof(&data) {
        let array = js_sys::Float32Array::from(data);
        let data = array.to_vec().into_iter().collect::<Vec<_>>();
        Tensor::from_data(data, shape, options)
    } else if js_sys::Float64Array::instanceof(&data) {
        let array = js_sys::Float64Array::from(data);
        let data = array
            .to_vec()
            .into_iter()
            .map(|v| v as f32)
            .collect::<Vec<_>>();
        Tensor::from_data(data, shape, options)
    } else if js_sys::Int32Array::instanceof(&data) {
        let array = js_sys::Int32Array::from(data);
        let data = array.to_vec().into_iter().collect::<Vec<_>>();
        Tensor::from_data(data, shape, options)
    } else if js_sys::Uint32Array::instanceof(&data) {
        let array = js_sys::Uint32Array::from(data);
        let data = array.to_vec().into_iter().collect::<Vec<_>>();
        Tensor::from_data(data, shape, options)
    } else {
        return Err(JsError::new("Unsupported data type"));
    }
}

#[js_tensor_web_op(name = To, variants = [method])]
pub async fn to(input: Tensor, device: Device) -> Result<JsTensor, JsError> {
    Ok::<_, JsError>(input.to(&device).await?)
}

#[js_tensor_web_op(name = Grad, variants = [method], getter)]
pub fn grad(input: Tensor) -> Result<Option<JsTensor>, JsError> {
    Ok::<_, JsError>(input.grad().map(JsTensor::new))
}

#[cfg(not(feature = "debug"))]
#[js_tensor_web_op(name = DebugTensor, variants = [method], getter)]
pub fn debug_tensor(input: Tensor) -> Result<JsTensor, JsError> {
    input.get_or_create_debug_tensor()
}

#[wasm_bindgen(js_name = promoteTypes)]
pub fn promote_types(dtype1: JsDType, dtype2: JsDType) -> Result<JsDType, JsError> {
    piston::promote_types(dtype1.dtype, dtype2.dtype)
        .map(|dtype| JsDType { dtype })
        .map_err(|e| e.into_js_error())
}

#[wasm_bindgen(js_class = Tensor)]
impl JsTensor {
    // Skipping copy; the api is not great right now

    // Skipping into_bytes; probably won't work great with reference counting

    // Skipping from_quantized; we don't really have well-rounded quantized support right now

    // Skipping from_disk; not clear what it would represent in the JS API

    pub fn detach(&self) -> Result<JsTensor, JsError> {
        Ok(JsTensor::new(
            self.inner().detach().map_err(|e| e.into_js_error())?,
        ))
    }

    pub fn detach_(&self) -> Result<JsTensor, JsError> {
        Ok(JsTensor::new(
            self.inner().detach_().map_err(|e| e.into_js_error())?,
        ))
    }

    #[wasm_bindgen(js_name = gpuBuffer, unchecked_return_type = "GPUBuffer|null")]
    pub fn gpu_buffer(&self) -> Result<JsValue, JsError> {
        let inner = self.inner();
        let inner_source = inner.inner_or_source();
        match inner_source.storage().as_ref() {
            Some(Storage::GPU(g)) => Ok(g.inner().as_webgpu_buffer().into()),
            _ => Err(JsError::new("Tensor is not on GPU")),
        }
    }

    #[wasm_bindgen(js_name = toVec, unchecked_return_type = "Float32Array | Int32Array | Uint32Array")]
    pub async fn to_vec(&self, dtype: Option<JsDType>) -> Result<JsValue, JsError> {
        let dtype = dtype.map(|d| d.dtype).unwrap_or(self.inner().dtype());
        let inner = self.inner();
        match dtype {
            DType::F32 => {
                let result = inner.to_vec::<f32>().await.map_err(|e| e.into_js_error())?;
                let array = js_sys::Float32Array::new_with_length(result.len() as u32);
                array.copy_from(&result);
                Ok(array.into())
            }
            DType::F16 => {
                let result = inner.to_vec::<f16>().await.map_err(|e| e.into_js_error())?;
                let f32_vec: Vec<f32> = result.iter().map(|&x| f16::to_f32(x)).collect();
                let array = js_sys::Float32Array::new_with_length(f32_vec.len() as u32);
                array.copy_from(&f32_vec);
                Ok(array.into())
            }
            DType::I32 => {
                let result = inner.to_vec::<i32>().await.map_err(|e| e.into_js_error())?;
                let array = js_sys::Int32Array::new_with_length(result.len() as u32);
                array.copy_from(&result);
                Ok(array.into())
            }
            DType::U32 => {
                let result = inner.to_vec::<u32>().await.map_err(|e| e.into_js_error())?;
                let array = js_sys::Uint32Array::new_with_length(result.len() as u32);
                array.copy_from(&result);
                Ok(array.into())
            }
            _ => {
                panic!("Unsupported dtype");
            }
        }
    }

    #[wasm_bindgen(unchecked_return_type = "number")]
    pub async fn item(&self, dtype: Option<JsDType>) -> Result<JsValue, JsError> {
        let dtype = dtype.map(|d| d.dtype).unwrap_or(self.inner().dtype());
        let inner = self.inner();
        match dtype {
            DType::F32 => Ok(JsValue::from_f64(inner.item::<f32>().await.into())),
            DType::F16 => Ok(JsValue::from_f64(
                f16::to_f32(inner.item::<f16>().await).into(),
            )),
            DType::I32 => Ok(JsValue::from_f64(inner.item::<i32>().await.into())),
            DType::U32 => Ok(JsValue::from_f64(inner.item::<u32>().await.into())),
            _ => panic!("Unsupported dtype"),
        }
    }

    #[wasm_bindgen(js_name = hasNaN)]
    pub fn has_nan(&self, dtype: Option<JsDType>) -> bool {
        let dtype = dtype.map(|d| d.dtype).unwrap_or(self.inner().dtype());
        match dtype {
            DType::F32 => self.inner().has_nan::<f32>(),
            DType::F16 => self.inner().has_nan::<f16>(),
            _ => panic!("Unsupported dtype"),
        }
    }

    #[wasm_bindgen(setter = grad)]
    pub fn set_grad(&self, grad: Option<JsTensor>) {
        self.inner().set_grad(grad.map(|g| g.inner().clone()));
    }

    pub fn backward(&self) -> Result<(), JsError> {
        self.inner().backward().map_err(|e| e.into_js_error())
    }
}

#[js_tensor_web_op(name = "Cat", variants = [function])]
pub fn cat(tensors: Vec<Tensor>, #[op(default = 0)] dim: Dim) -> anyhow::Result<Tensor> {
    piston::cat(tensors.into(), dim)
}

#[js_tensor_web_op(name = "Stack", variants = [function])]
pub fn stack(tensors: Vec<Tensor>, #[op(default = 0)] dim: Dim) -> anyhow::Result<Tensor> {
    piston::stack(tensors.into(), dim)
}

#[js_tensor_web_op(name = "Arange", variants = [function])]
pub fn arange(
    #[op(keyword)] end: f32,
    #[op(default = 0.0)] start: f32,
    #[op(default = 1.0)] step: f32,
    options: TensorOptions,
) -> anyhow::Result<Tensor> {
    piston::arange(Some(start), end, Some(step), options)
}

#[js_tensor_web_op(name = "Randint", variants = [function])]
pub fn randint(
    high: i32,
    #[op(keyword)] shape: Shape,
    #[op(default = 0)] low: i32,
    options: TensorOptions,
) -> anyhow::Result<Tensor> {
    piston::randint(low, high, shape, options)
}

#[js_tensor_web_op(name = "Randn", variants = [function])]
pub fn randn(
    shape: Shape,
    mean: Option<f32>,
    std: Option<f32>,
    options: TensorOptions,
) -> anyhow::Result<Tensor> {
    piston::randn(shape, mean, std, options)
}

#[js_tensor_web_op(name = "Rand", variants = [function])]
pub fn rand(
    shape: Shape,
    lo: Option<f32>,
    up: Option<f32>,
    options: TensorOptions,
) -> anyhow::Result<Tensor> {
    piston::rand(shape, lo, up, options)
}

#[js_tensor_web_op(name = "Eye", variants = [function])]
pub fn eye(n: usize, m: Option<usize>, options: TensorOptions) -> anyhow::Result<Tensor> {
    piston::eye(n, m, options)
}

#[js_tensor_web_op(name = OneHot, variants = [function, method])]
pub fn one_hot(input: Tensor, #[op(name = "numClasses")] num_classes: usize) -> JsTensorResult {}

#[js_tensor_web_op(name = "Zeros", variants = [function])]
pub fn zeros(shape: Shape, options: TensorOptions) -> anyhow::Result<Tensor> {
    piston::zeros(shape, options)
}

#[js_tensor_web_op(name = "Ones", variants = [function])]
pub fn ones(shape: Shape, options: TensorOptions) -> anyhow::Result<Tensor> {
    piston::ones(shape, options)
}

#[wasm_bindgen(js_class = Tensor)]
impl JsTensor {
    #[wasm_bindgen(js_name = _clone)]
    pub fn _clone_js(&self) -> JsTensor {
        JsTensor::new(self.inner().clone())
    }

    #[wasm_bindgen(js_name = _cloneWeak)]
    pub fn _clone_weak_js(&self) -> JsTensor {
        JsTensor::new_weak(self.inner.downgrade())
    }
}

#[derive(Clone)]
#[wasm_bindgen(js_name = TensorOrScalar)]
struct JsTensorOrScalar {
    inner: JsValue,
}

impl TensorTypeOrScalar<Tensor> for JsTensorOrScalar {
    fn tensor_or_scalar(&self) -> anyhow::Result<TensorTypeOrScalarEnum<Tensor>> {
        if let Ok(other) = JsTensor::try_from(self.inner.clone()) {
            Ok(TensorTypeOrScalarEnum::Tensor(other.inner()))
        } else {
            let other: f32 = self
                .inner
                .as_f64()
                .map(|f| f as f32)
                .ok_or_else(|| anyhow::anyhow!("Failed to convert JsValue to f32"))?;
            Ok(TensorTypeOrScalarEnum::Scalar(other))
        }
    }
}

impl TryFrom<JsValue> for JsTensorOrScalar {
    type Error = JsError;
    fn try_from(value: JsValue) -> Result<Self, Self::Error> {
        if let Some(scalar) = value.as_f64() {
            Ok(JsTensorOrScalar {
                inner: JsValue::from_f64(scalar),
            })
        } else {
            // We don't do any extra validation here; just hope it'll work
            Ok(JsTensorOrScalar {
                inner: value.clone(),
            })
        }
    }
}

fn js_value_to_norm_ord(value: JsValue) -> Result<Option<NormOrd>, JsError> {
    if value.is_undefined() || value.is_null() {
        // Handle undefined or null values
        Ok(None)
    } else if let Some(num) = value.as_f64() {
        // Handle special numeric cases
        if num == 0.0 {
            Ok(Some(NormOrd::Zero))
        } else if num == 1.0 {
            Ok(Some(NormOrd::One))
        } else if num == -1.0 {
            Ok(Some(NormOrd::NegOne))
        } else if num == f64::INFINITY {
            Ok(Some(NormOrd::Inf))
        } else if num == f64::NEG_INFINITY {
            Ok(Some(NormOrd::NegInf))
        } else {
            // For other numbers, use P norm
            Ok(Some(NormOrd::P(num as f32)))
        }
    } else if let Some(string) = value.as_string() {
        // Handle string values using from_str
        Ok(Some(NormOrd::from_str(&string).map_err(|e| {
            JsError::new(&format!("Invalid norm order: {e}"))
        })?))
    } else {
        Err(JsError::new("Norm order must be a number or string"))
    }
}

fn convert_ir_fields_to_js(
    fields: &HashMap<String, IrValue, impl std::hash::BuildHasher>,
    op: &LazyOp,
) -> JsValue {
    let obj = js_sys::Object::new();

    for (key, value) in fields {
        js_sys::Reflect::set(&obj, &key.into(), &convert_ir_value_to_js(value, op)).unwrap();
    }

    obj.into()
}

// Helper function to convert individual IrValue to JsValue
fn convert_ir_value_to_js(value: &IrValue, op: &LazyOp) -> JsValue {
    match value {
        IrValue::Tensor(tensor_value) => {
            // Use existing JS tensors from the active map to avoid creating strong allocations
            if let Some(list) = active_tensors().get(&tensor_value.id)
                && let Some(first) = list.first()
            {
                return first.js_value();
            }
            // No suitable existing JS tensor found :(
            JsValue::NULL
        }
        // Handle scalar values
        IrValue::Scalar(scalar) => match scalar {
            IrScalarValue::F32(val) => JsValue::from_f64(*val as f64),
            IrScalarValue::I32(val) => JsValue::from_f64(*val as f64),
            IrScalarValue::U32(val) => JsValue::from_f64(*val as f64),
            IrScalarValue::Bool(val) => JsValue::from_bool(*val),
            IrScalarValue::String(val) => JsValue::from_str(val),
            IrScalarValue::Vec4U32(val) => {
                let array = js_sys::Array::new();
                array.push(&JsValue::from_f64(val.x as f64));
                array.push(&JsValue::from_f64(val.y as f64));
                array.push(&JsValue::from_f64(val.z as f64));
                array.push(&JsValue::from_f64(val.w as f64));
                array.into()
            }
        },
        // Handle nested IR
        IrValue::Ir(nested_ir) => {
            let obj = js_sys::Object::new();
            js_sys::Reflect::set(&obj, &"name".into(), &JsValue::from_str(nested_ir.name()))
                .unwrap();

            if let Some(nested_fields) = nested_ir.fields() {
                js_sys::Reflect::set(
                    &obj,
                    &"fields".into(),
                    &convert_ir_fields_to_js(nested_fields, op),
                )
                .unwrap();
            }

            obj.into()
        }
        // Handle Fields (should not happen in finalized op)
        IrValue::Fields(_) => JsValue::NULL,
        // Handle Vectors of IrValues
        IrValue::Vec(vec_values) => {
            let array = js_sys::Array::new();
            for value in vec_values.iter() {
                array.push(&convert_ir_value_to_js(value, op));
            }
            array.into()
        }
        // Handle None
        IrValue::None => JsValue::NULL,
    }
}

impl TryFrom<JsValue> for JsTensor {
    type Error = JsError;
    fn try_from(value: JsValue) -> Result<Self, Self::Error> {
        downcast_from_ptr::<JsTensor>(&value, "__wbg_piston_tensor", true)
            .ok_or_else(|| JsError::new("Failed to downcast Tensor from JS value"))
    }
}

thread_local! {
    pub(crate) static ACTIVE_TENSORS: RefCell<Vec<JsTensor>> = const { RefCell::new(Vec::new()) };
}

fn register_active_tensor(tensor: JsTensor) {
    ACTIVE_TENSORS
        .try_with(|cell| {
            cell.borrow_mut()
                .push(JsTensor::new_weak(tensor.inner.downgrade()));
        })
        .ok();
}

fn active_tensors() -> HashMap<TensorId, Vec<JsTensor>> {
    ACTIVE_TENSORS.with(|cell| {
        let mut list = cell.borrow_mut();
        // Clean out broken references
        list.retain(|t| t.inner.upgrade().filter(|t| t.0.read().is_some()).is_some());
        // Group by TensorId
        let mut map: HashMap<TensorId, Vec<JsTensor>> = HashMap::new();
        for t in list.iter() {
            let id = t.inner().id();
            map.entry(id).or_default().push(t._clone_weak_js());
        }
        map
    })
}

#[wasm_bindgen(js_name = __pistonActiveTensors, unchecked_return_type = "Map<number, Tensor[]>")]
pub fn active_tensors_js() -> JsValue {
    let map_js = js_sys::Map::new();
    for (id, list) in active_tensors() {
        let arr = Array::new();
        for js_tensor in list {
            arr.push(&JsValue::from(js_tensor));
        }
        map_js.set(&JsValue::from_f64(id.0 as f64), &arr.into());
    }
    map_js.into()
}

/// Unique identifier for tensors.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[wasm_bindgen]
pub struct StrongJsTensorId(pub usize);

impl std::fmt::Debug for StrongJsTensorId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "T{}", self.0)
    }
}

impl Ord for StrongJsTensorId {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl PartialOrd for StrongJsTensorId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl StrongJsTensorId {
    pub(crate) fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}
