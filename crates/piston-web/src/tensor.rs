use crate::error::IntoJsError;
use crate::js_util::downcast_from_ptr;
use half::f16;
use js_sys::{Array, Function, Object, Reflect};
use paste::paste;
use piston::{
    AllDims, DType, Device, Dim, IrScalarValue, IrValue, LazyOp, NormOrd, Shape, Tensor,
    TensorOptions as CoreTensorOptions,
};
use piston::{TensorTypeOrScalar, TensorTypeOrScalarEnum};
use piston_macros::js_tensor_operations;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashMap;
use std::str::FromStr;
use tsify::Tsify;
use wasm_bindgen::prelude::*;
use wasm_bindgen::{JsError, JsValue};

use crate::{
    device::{GPU_DEVICE, JsDevice, gpu_sync},
    dtype::JsDType,
    shape::{FromJsDim, FromJsVecISize},
};

#[derive(Tsify, Serialize, Deserialize)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct TensorOptions {
    #[serde(default, with = "crate::js_util::try_from_js_value_preserve")]
    #[tsify(optional)]
    pub dtype: Option<JsDType>,
    #[serde(default, with = "crate::js_util::try_from_js_value_preserve")]
    #[tsify(optional)]
    pub device: Option<JsDevice>,
    #[serde(default, rename = "requiresGrad")]
    #[tsify(optional)]
    pub requires_grad: Option<bool>,
}

impl From<TensorOptions> for CoreTensorOptions {
    fn from(options: TensorOptions) -> Self {
        CoreTensorOptions {
            dtype: options.dtype.map(|d| d.dtype),
            device: options.device.map(|d| d.inner),
            requires_grad: options.requires_grad,
        }
    }
}

#[wasm_bindgen(js_name = Tensor)]
pub struct JsTensor {
    pub(crate) inner: Tensor,
}

type JsTensorResult = Result<JsTensor, JsError>;

#[wasm_bindgen(js_class = Tensor)]
impl JsTensor {
    // Marker function for downcasting from a JS object
    pub fn __wbg_piston_tensor() {}

    #[wasm_bindgen(getter)]
    pub fn id(&self) -> usize {
        self.inner.id().0
    }

    // - Skipping storage_view

    pub fn dim(&self) -> usize {
        self.inner.dim()
    }

    #[wasm_bindgen(getter)]
    pub fn ndim(&self) -> usize {
        self.inner.dim()
    }

    #[wasm_bindgen(getter)]
    pub fn dtype(&self) -> JsDType {
        JsDType {
            dtype: self.inner.dtype(),
        }
    }

    #[wasm_bindgen(unchecked_return_type = "number[] | number")]
    pub fn size(&self, dim: Option<isize>) -> JsValue {
        if let Some(dim) = dim {
            let dim = FromJsDim(dim);
            let size = dim.to_index(&self.inner.shape(), "size").unwrap();
            JsValue::from_f64(self.inner.shape()[size] as f64)
        } else {
            let shape = self.inner.shape().to_vec();
            let array = js_sys::Array::new();
            for dim in shape {
                array.push(&JsValue::from_f64(dim as f64));
            }
            array.into()
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    #[wasm_bindgen(unchecked_return_type = "number[] | number")]
    pub fn stride(&self, dim: Option<isize>) -> JsValue {
        if let Some(dim) = dim {
            let dim = FromJsDim(dim);
            let stride = dim.to_index(&self.inner.shape(), "stride").unwrap();
            JsValue::from_f64(self.inner.stride()[stride] as f64)
        } else {
            let stride = self.inner.stride().to_vec();
            let array = js_sys::Array::new();
            for dim in stride {
                array.push(&JsValue::from_f64(dim as f64));
            }
            array.into()
        }
    }

    #[wasm_bindgen(getter = nbytes)]
    pub fn num_bytes(&self) -> usize {
        self.inner.num_bytes()
    }

    pub fn device(&self) -> String {
        match self.inner.device() {
            Device::CPU => "cpu".to_string(),
            Device::GPU(_) => "webgpu".to_string(),
        }
    }

    // - Skipping storage

    pub fn resolved(&self) -> bool {
        self.inner.resolved()
    }

    pub fn op(&self) -> JsValue {
        let op = self.inner.op();
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
    #[wasm_bindgen(js_name = srcIds)]
    pub fn src_ids(&self) -> Vec<usize> {
        self.inner.op().srcs().iter().map(|s| s.id().0).collect()
    }

    pub fn scope(&self) -> Option<String> {
        self.inner.scope().clone()
    }

    #[wasm_bindgen(js_name = isScalar)]
    pub fn is_scalar(&self) -> bool {
        self.inner.is_scalar()
    }

    #[wasm_bindgen(getter = requiresGrad)]
    pub fn requires_grad(&self) -> bool {
        self.inner.requires_grad()
    }

    #[wasm_bindgen(js_name = storageId)]
    pub fn storage_id(&self) -> Option<usize> {
        self.inner.storage_id()
    }
}

macro_rules! impl_binary_op {
    ($op:ident) => {
        #[wasm_bindgen(js_class = Tensor)]
        #[js_tensor_operations]
        impl JsTensor {
            pub fn $op(&self, other: TensorOrScalar) -> JsTensorResult {}
        }
    };
}

macro_rules! impl_binary_op_tensor_only {
    ($op:ident) => {
        paste! {
            #[wasm_bindgen(js_class = Tensor)]
            #[js_tensor_operations]
            impl JsTensor {
                pub fn $op(&self, other: Tensor) -> JsTensorResult {}
            }
        }
    };
}

macro_rules! impl_ternary_op {
    ($op:ident) => {
        paste! {
            #[wasm_bindgen(js_class = Tensor)]
            #[js_tensor_operations]
            impl JsTensor {
                pub fn $op(&self, tensor1: Tensor, tensor2: Tensor, value: f32) -> JsTensorResult {}
                pub fn [<$op _>](&self, tensor1: Tensor, tensor2: Tensor, value: f32) -> JsTensorResult {}
            }
        }
    };
}

macro_rules! impl_cmp_op {
    ($op:ident) => {
        paste! {
            #[wasm_bindgen(js_class = Tensor)]
            #[js_tensor_operations]
            impl JsTensor {
                pub fn $op(&self, other: TensorOrScalar) -> JsTensorResult {}
                pub fn [<$op _>](&self, other: TensorOrScalar) -> JsTensorResult {}
            }
        }
    };
}

macro_rules! impl_unary_op {
    ($op:ident) => {
        paste! {
            #[wasm_bindgen(js_class = Tensor)]
            #[js_tensor_operations]
            impl JsTensor {
                pub fn $op(&self) -> JsTensorResult {}
                pub fn [<$op _>](&self) -> JsTensorResult {}
            }
        }
    };
}

impl_binary_op!(add);
impl_binary_op!(add_);
impl_binary_op!(sub);
impl_binary_op!(sub_);
impl_binary_op!(mul);
impl_binary_op!(mul_);
impl_binary_op!(div);
impl_binary_op!(div_);
impl_binary_op!(pow);
impl_binary_op!(pow_);
impl_binary_op_tensor_only!(minimum);
impl_binary_op_tensor_only!(maximum);
impl_binary_op_tensor_only!(minimum_);
impl_binary_op_tensor_only!(maximum_);

impl_ternary_op!(addcdiv);
impl_ternary_op!(addcmul);

impl_cmp_op!(eq);
impl_cmp_op!(ne);
impl_cmp_op!(gt);
impl_cmp_op!(ge);
impl_cmp_op!(lt);
impl_cmp_op!(le);

impl_unary_op!(gelu);
impl_unary_op!(tanh);
impl_unary_op!(exp);
impl_unary_op!(log);
impl_unary_op!(sin);
impl_unary_op!(cos);
impl_unary_op!(abs);
impl_unary_op!(sqrt);
impl_unary_op!(relu);
impl_unary_op!(relu2);
impl_unary_op!(floor);
impl_unary_op!(ceil);
impl_unary_op!(neg);
impl_unary_op!(sigmoid);
impl_unary_op!(swiglu);
impl_unary_op!(silu);
impl_unary_op!(square);
impl_unary_op!(recip);

#[wasm_bindgen(js_class = Tensor)]
#[js_tensor_operations]
impl JsTensor {
    #[dtype_generic]
    pub fn full(
        shape: Shape,
        value: f32,
        dtype: DType,
        device: &Device,
        requires_grad: bool,
    ) -> JsTensorResult {
    }

    pub fn cast(&self, dst_dtype: DType) -> JsTensorResult {}

    pub fn float(&self) -> JsTensorResult {}

    pub fn half(&self) -> JsTensorResult {}

    pub fn group_norm(
        &self,
        num_groups: usize,
        weight: Option<Tensor>,
        bias: Option<Tensor>,
        eps: f32,
    ) -> JsTensorResult {
    }

    pub fn layer_norm(
        &self,
        weight: Option<Tensor>,
        bias: Option<Tensor>,
        eps: f32,
    ) -> JsTensorResult {
    }

    pub fn rms_norm(&self, weight: Option<Tensor>, eps: f32) -> JsTensorResult {}

    pub fn conv1d(
        &self,
        weight: Tensor,
        bias: Option<Tensor>,
        stride: usize,
        padding: usize,
    ) -> JsTensorResult {
    }

    pub fn softmax(&self, dim: Dim) -> JsTensorResult {}

    pub fn rope_(&self, dim: usize, base: f32, offset: usize) -> JsTensorResult {}

    pub fn alibi(&self, max_bias: f32) -> JsTensorResult {}

    pub fn matmul(&self, rhs: Tensor, trans_lhs: bool, trans_rhs: bool) -> JsTensorResult {}

    pub fn gemm(
        &self,
        rhs: Tensor,
        bias: Option<Tensor>,
        trans_lhs: bool,
        trans_rhs: bool,
        trans_out: bool,
    ) -> JsTensorResult {
    }

    pub fn affine(&self, mul: f32, add: f32) -> JsTensorResult {}

    pub fn sum(self, dim: Option<Dims>, keepdim: bool) -> JsTensorResult {
        let tensor = if let Some(sum_dims) = dim {
            self.inner.sum(sum_dims, keepdim)
        } else {
            self.inner.sum(AllDims, keepdim)
        }
        .map_err(|e| e.into_js_error())?;
        Ok(JsTensor { inner: tensor })
    }

    pub fn mean(self, dim: Option<Dims>, keepdim: bool) -> JsTensorResult {
        let tensor = if let Some(mean_dims) = dim {
            self.inner.mean(mean_dims, keepdim)
        } else {
            self.inner.mean(AllDims, keepdim)
        }
        .map_err(|e| e.into_js_error())?;
        Ok(JsTensor { inner: tensor })
    }

    pub fn var(self, dim: Option<Dims>, keepdim: bool) -> JsTensorResult {
        let tensor = if let Some(var_dims) = dim {
            self.inner.var(var_dims, keepdim)
        } else {
            self.inner.var(AllDims, keepdim)
        }
        .map_err(|e| e.into_js_error())?;
        Ok(JsTensor { inner: tensor })
    }

    pub fn max(self, dim: Dim, keepdim: bool) -> JsTensorResult {
        let tensor = self
            .inner
            .max(dim, keepdim)
            .map_err(|e| e.into_js_error())?;
        Ok(JsTensor { inner: tensor })
    }

    pub fn min(self, dim: Dim, keepdim: bool) -> JsTensorResult {
        let tensor = self
            .inner
            .min(dim, keepdim)
            .map_err(|e| e.into_js_error())?;
        Ok(JsTensor { inner: tensor })
    }

    pub fn argmax(self, dim: Dim, keepdim: bool) -> JsTensorResult {
        let tensor = self
            .inner
            .argmax(dim, keepdim)
            .map_err(|e| e.into_js_error())?;
        Ok(JsTensor { inner: tensor })
    }

    pub fn argmin(self, dim: Dim, keepdim: bool) -> JsTensorResult {
        let tensor = self
            .inner
            .argmin(dim, keepdim)
            .map_err(|e| e.into_js_error())?;
        Ok(JsTensor { inner: tensor })
    }

    pub fn norm(self, ord: JsValue, dim: Option<Dims>, keepdim: bool) -> JsTensorResult {
        let norm_ord = js_value_to_norm_ord(ord)?;
        let tensor = if let Some(dim) = dim {
            self.inner.norm(norm_ord, dim, keepdim)
        } else {
            self.inner.norm(norm_ord, AllDims, keepdim)
        }
        .map_err(|e| e.into_js_error())?;
        Ok(JsTensor { inner: tensor })
    }

    pub fn flatten(self, start_dim: Option<Dim>, end_dim: Option<Dim>) -> JsTensorResult {
        let start_dim = start_dim.unwrap_or(FromJsDim(0));
        let end_dim = end_dim.unwrap_or(FromJsDim(-1));
        let tensor = self
            .inner
            .flatten(start_dim, end_dim)
            .map_err(|e| e.into_js_error())?;
        Ok(JsTensor { inner: tensor })
    }

    pub fn slice(self, ranges: JsValue) -> JsTensorResult {
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
        let result = self.inner.slice(&ranges).map_err(|e| e.into_js_error())?;
        Ok(JsTensor { inner: result })
    }

    pub fn view(self, shape: ShapeWithOneHole) -> JsTensorResult {}

    pub fn unsqueeze(self, dim: Dim) -> JsTensorResult {}

    pub fn squeeze(self, dims: Option<Dims>) -> JsTensorResult {
        let inner = self.inner.clone();
        let result = match dims {
            Some(dims) => inner.squeeze(dims),
            None => inner.squeeze(()),
        }
        .map_err(|e| e.into_js_error())?;
        Ok(Self { inner: result })
    }

    pub fn permute(self, dims: Dims) -> JsTensorResult {}

    pub fn transpose(self, dim0: Dim, dim1: Dim) -> JsTensorResult {}

    pub fn t(self) -> JsTensorResult {}

    #[allow(non_snake_case)]
    #[wasm_bindgen(getter)]
    pub fn mT(self) -> JsTensorResult {}

    pub fn cache(self, source: Tensor, dim: Dim, offset: usize) -> JsTensorResult {}

    /// Returns a new tensor duplicating data from the original tensor. New dimensions are inserted
    /// on the left.
    pub fn broadcast_left(self, left_shape: Shape) -> JsTensorResult {}

    pub fn broadcast_to(self, shape: Shape) -> JsTensorResult {}

    pub fn index_select(self, indices: Tensor, dim: Dim) -> JsTensorResult {}

    pub fn index_write(self, src: Tensor, write_start: Dims) -> JsTensorResult {}

    #[wasm_bindgen(js_name = where)]
    pub fn where_cond(self, condition: Tensor, on_false: TensorOrScalar) -> JsTensorResult {}

    pub fn scatter_add(self, indices: Tensor, source: Tensor, dim: Dim) -> JsTensorResult {}

    pub fn index_add_(self, indices: Tensor, source: Tensor, dim: Dim) -> JsTensorResult {}

    pub fn gather(self, indices: Tensor, dim: Dim) -> JsTensorResult {}

    pub fn triu(self, k: Option<i32>) -> JsTensorResult {}

    pub fn triu_(self, k: Option<i32>) -> JsTensorResult {}

    pub fn tril(self, k: Option<i32>) -> JsTensorResult {}

    pub fn tril_(self, k: Option<i32>) -> JsTensorResult {}

    pub fn lerp(&self, end: Tensor, weight: TensorOrScalar) -> JsTensorResult {}

    pub fn lerp_(&self, end: Tensor, weight: TensorOrScalar) -> JsTensorResult {}

    // We use the dtype of the tensor
    pub fn bernoulli(self) -> JsTensorResult {}

    pub fn bernoulli_(self) -> JsTensorResult {}

    pub fn zero_(self) -> JsTensorResult {}

    pub fn zeros_like(self, options: TensorOptions) -> JsTensorResult {}

    pub fn ones_like(self, options: TensorOptions) -> JsTensorResult {}

    pub fn is_contiguous(&self) -> bool {
        self.inner.is_contiguous()
    }

    pub fn contiguous(self) -> JsTensorResult {}

    pub fn from_data(
        data: JsValue,
        shape: Shape,
        device: &Device,
        requires_grad: bool,
    ) -> JsTensorResult {
        let tensor = if data.is_array() {
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

            Tensor::from_data(data, shape, device.clone(), requires_grad)
        } else if js_sys::Float32Array::instanceof(&data) {
            let array = js_sys::Float32Array::from(data);
            let data = array.to_vec().into_iter().collect::<Vec<_>>();
            Tensor::from_data(data, shape, device.clone(), requires_grad)
        } else if js_sys::Float64Array::instanceof(&data) {
            let array = js_sys::Float64Array::from(data);
            let data = array
                .to_vec()
                .into_iter()
                .map(|v| v as f32)
                .collect::<Vec<_>>();
            Tensor::from_data(data, shape, device.clone(), requires_grad)
        } else if js_sys::Int32Array::instanceof(&data) {
            let array = js_sys::Int32Array::from(data);
            let data = array.to_vec().into_iter().collect::<Vec<_>>();
            Tensor::from_data(data, shape, device.clone(), requires_grad)
        } else if js_sys::Uint32Array::instanceof(&data) {
            let array = js_sys::Uint32Array::from(data);
            let data = array.to_vec().into_iter().collect::<Vec<_>>();
            Tensor::from_data(data, shape, device.clone(), requires_grad)
        } else {
            return Err(JsError::new("Unsupported data type"));
        };
        Ok(JsTensor { inner: tensor })
    }

    pub fn from_bytes(
        data: &[u8],
        dtype: DType,
        shape: Shape,
        device: Device,
        requires_grad: bool,
    ) -> JsTensorResult {
    }

    pub fn detach(&self) -> JsTensor {
        JsTensor {
            inner: self.inner.detach(),
        }
    }

    pub fn detach_(&self) -> JsTensor {
        JsTensor {
            inner: self.inner.detach_(),
        }
    }

    #[wasm_bindgen(js_name = requiresGrad_)]
    pub fn requires_grad_(&self, requires_grad: bool) -> JsTensorResult {}

    // Skipping copy; the api is not great right now

    // Skipping into_bytes; probably won't work great with reference counting

    // Skipping from_quantized; we don't really have well-rounded quantized support right now

    // Skipping from_disk; not clear what it would represent in the JS API
}

#[wasm_bindgen(js_class = Tensor)]
impl JsTensor {
    #[wasm_bindgen(js_name = toVec, unchecked_return_type = "Float32Array | Int32Array | Uint32Array")]
    pub async fn to_vec(&self, dtype: Option<JsDType>) -> Result<JsValue, JsError> {
        let dtype = dtype.map(|d| d.dtype).unwrap_or(self.inner.dtype());
        match dtype {
            DType::F32 => {
                let result = self
                    .inner
                    .to_vec::<f32>()
                    .await
                    .map_err(|e| e.into_js_error())?;
                let array = js_sys::Float32Array::new_with_length(result.len() as u32);
                array.copy_from(&result);
                Ok(array.into())
            }
            DType::F16 => {
                let result = self
                    .inner
                    .to_vec::<f16>()
                    .await
                    .map_err(|e| e.into_js_error())?;
                let f32_vec: Vec<f32> = result.iter().map(|&x| f16::to_f32(x)).collect();
                let array = js_sys::Float32Array::new_with_length(f32_vec.len() as u32);
                array.copy_from(&f32_vec);
                Ok(array.into())
            }
            DType::I32 => {
                let result = self
                    .inner
                    .to_vec::<i32>()
                    .await
                    .map_err(|e| e.into_js_error())?;
                let array = js_sys::Int32Array::new_with_length(result.len() as u32);
                array.copy_from(&result);
                Ok(array.into())
            }
            DType::U32 => {
                let result = self
                    .inner
                    .to_vec::<u32>()
                    .await
                    .map_err(|e| e.into_js_error())?;
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
        let dtype = dtype.map(|d| d.dtype).unwrap_or(self.inner.dtype());
        match dtype {
            DType::F32 => Ok(JsValue::from_f64(self.inner.item::<f32>().await.into())),
            DType::F16 => Ok(JsValue::from_f64(
                f16::to_f32(self.inner.item::<f16>().await).into(),
            )),
            DType::I32 => Ok(JsValue::from_f64(self.inner.item::<i32>().await.into())),
            DType::U32 => Ok(JsValue::from_f64(self.inner.item::<u32>().await.into())),
            _ => panic!("Unsupported dtype"),
        }
    }

    #[wasm_bindgen(getter, js_name = T)]
    #[allow(non_snake_case)]
    pub fn T_upper(self) -> JsTensorResult {
        // We can't generate this one automatically because wasm bindgen lowercases all function
        // names internally, so t() and T conflict
        let inner = self.inner.T().map_err(|e| e.into_js_error())?;
        Ok(JsTensor { inner })
    }

    pub async fn to(&self, device: String) -> Result<JsTensor, JsError> {
        let device = match device.as_str() {
            "cpu" => Device::CPU,
            "gpu" | "webgpu" => {
                GPU_DEVICE.with(|device| device.borrow().clone().unwrap().inner.clone())
            }
            _ => return Err(JsError::new("Unsupported device")),
        };
        let inner = self.inner.to(&device).await?;
        Ok(JsTensor { inner })
    }

    #[wasm_bindgen(js_name = hasNaN)]
    pub fn has_nan(&self, dtype: Option<JsDType>) -> bool {
        let dtype = dtype.map(|d| d.dtype).unwrap_or(self.inner.dtype());
        match dtype {
            DType::F32 => self.inner.has_nan::<f32>(),
            DType::F16 => self.inner.has_nan::<f16>(),
            _ => panic!("Unsupported dtype"),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn grad(&self) -> Option<JsTensor> {
        self.inner.grad().map(|grad| JsTensor { inner: grad })
    }

    #[wasm_bindgen(setter = grad)]
    pub fn set_grad(&self, grad: Option<JsTensor>) {
        self.inner.set_grad(grad.map(|g| g.inner));
    }

    #[cfg(not(feature = "debug"))]
    #[wasm_bindgen(js_name = debugTensor)]
    pub fn debug_tensor(&self) -> Result<JsTensor, JsError> {
        let tensor = self.inner.get_or_create_debug_tensor()?;
        Ok(JsTensor { inner: tensor })
    }

    pub fn backward(&self) -> Result<(), JsError> {
        self.inner.backward().map_err(|e| e.into_js_error())
    }

    pub fn invalidate(&mut self) -> Result<(), JsError> {
        self.inner.invalidate().map_err(|e| e.into())
    }
}

#[wasm_bindgen(js_name = cat)]
pub fn cat(tensors: Vec<JsTensor>, dim: isize) -> JsTensorResult {
    let tensors = tensors.into_iter().map(|t| t.inner).collect();
    let dim = FromJsDim(dim);
    let tensor = piston::cat(tensors, dim).map_err(|e| e.into_js_error())?;
    Ok(JsTensor { inner: tensor })
}

#[wasm_bindgen(js_name = stack)]
pub fn stack(tensors: Vec<JsTensor>, dim: isize) -> JsTensorResult {
    let tensors = tensors.into_iter().map(|t| t.inner).collect();
    let dim = FromJsDim(dim);
    let tensor = piston::stack(tensors, dim).map_err(|e| e.into_js_error())?;
    Ok(JsTensor { inner: tensor })
}

#[derive(Tsify, Serialize, Deserialize, Default)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct ArangeNamedArgs {
    #[tsify(optional)]
    pub start: Option<f32>,
    pub end: f32,
    #[tsify(optional)]
    pub step: Option<f32>,
}

#[wasm_bindgen(js_name = arange)]
pub fn arange(kwargs: Option<ArangeNamedArgs>, options: Option<TensorOptions>) -> JsTensorResult {
    let kwargs = kwargs.unwrap_or_default();
    let options = options.map(|o| o.into()).unwrap_or_default();
    let start = kwargs.start.unwrap_or(0.0);
    let step = kwargs.step.unwrap_or(1.0);
    let tensor = piston::arange(Some(start), kwargs.end, Some(step), options)
        .map_err(|e| e.into_js_error())?;
    Ok(JsTensor { inner: tensor })
}

#[derive(Tsify, Serialize, Deserialize, Default)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct RandintNamedArgs {
    #[tsify(optional)]
    pub low: Option<i32>,
}

#[wasm_bindgen(js_name = randint)]
pub fn randint(
    high: i32,
    #[wasm_bindgen(unchecked_param_type = "Uint32Array | number[]")] shape: Vec<usize>,
    kwargs: Option<RandintNamedArgs>,
    options: Option<TensorOptions>,
) -> JsTensorResult {
    let kwargs = kwargs.unwrap_or_default();
    let low = kwargs.low.unwrap_or(0);
    let options = options.map(|o| o.into()).unwrap_or_default();
    let tensor = piston::randint(low, high, shape, options).map_err(|e| e.into_js_error())?;
    Ok(JsTensor { inner: tensor })
}

#[derive(Tsify, Serialize, Deserialize, Default)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct RandnNamedArgs {
    #[tsify(optional)]
    pub mean: Option<f32>,
    #[tsify(optional)]
    pub std: Option<f32>,
}

#[wasm_bindgen(js_name = randn)]
pub fn randn(
    #[wasm_bindgen(unchecked_param_type = "Uint32Array | number[]")] shape: Vec<usize>,
    kwargs: Option<RandnNamedArgs>,
    options: Option<TensorOptions>,
) -> JsTensorResult {
    let kwargs = kwargs.unwrap_or_default();
    let options = options.map(|o| o.into()).unwrap_or_default();
    let tensor =
        piston::randn(shape, kwargs.mean, kwargs.std, options).map_err(|e| e.into_js_error())?;
    Ok(JsTensor { inner: tensor })
}

#[derive(Tsify, Serialize, Deserialize, Default)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct RandNamedArgs {
    #[tsify(optional)]
    pub lo: Option<f32>,
    #[tsify(optional)]
    pub up: Option<f32>,
}

#[wasm_bindgen(js_name = rand)]
pub fn rand(
    #[wasm_bindgen(unchecked_param_type = "Uint32Array | number[]")] shape: Vec<usize>,
    kwargs: Option<RandNamedArgs>,
    options: Option<TensorOptions>,
) -> JsTensorResult {
    let kwargs = kwargs.unwrap_or_default();
    let options = options.map(|o| o.into()).unwrap_or_default();
    let tensor =
        piston::rand(shape, kwargs.lo, kwargs.up, options).map_err(|e| e.into_js_error())?;
    Ok(JsTensor { inner: tensor })
}

#[wasm_bindgen(js_name = zeros)]
pub fn zeros(
    #[wasm_bindgen(unchecked_param_type = "Uint32Array | number[]")] shape: Vec<usize>,
    options: Option<TensorOptions>,
) -> JsTensorResult {
    let options = options.map(|o| o.into()).unwrap_or_default();
    let tensor = piston::zeros(shape, options).map_err(|e| e.into_js_error())?;
    Ok(JsTensor { inner: tensor })
}

#[wasm_bindgen(js_name = ones)]
pub fn ones(
    #[wasm_bindgen(unchecked_param_type = "Uint32Array | number[]")] shape: Vec<usize>,
    options: Option<TensorOptions>,
) -> JsTensorResult {
    let options = options.map(|o| o.into()).unwrap_or_default();
    let tensor = piston::ones(shape, options).map_err(|e| e.into_js_error())?;
    Ok(JsTensor { inner: tensor })
}

#[wasm_bindgen(js_class = Tensor)]
impl JsTensor {
    #[wasm_bindgen]
    pub fn _clone(&self) -> JsTensor {
        JsTensor {
            inner: self.inner.clone(),
        }
    }
}

#[derive(Clone)]
struct JsTensorOrScalar {
    inner: JsValue,
}

impl TensorTypeOrScalar<Tensor> for JsTensorOrScalar {
    fn tensor_or_scalar(&self) -> anyhow::Result<TensorTypeOrScalarEnum<Tensor>> {
        if let Ok(other) = JsTensor::try_from(self.inner.clone()) {
            Ok(TensorTypeOrScalarEnum::Tensor(other.inner))
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
            // Get the tensor from the operation's sources
            if let Some(tensor) = op.srcs().iter().find(|t| t.id() == tensor_value.id) {
                let js_tensor = JsTensor {
                    inner: Tensor::wrap((*tensor).clone()),
                };
                JsValue::from(js_tensor)
            } else {
                JsValue::NULL
            }
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
        downcast_from_ptr(&value, "__wbg_piston_tensor")
            .ok_or_else(|| JsError::new("Failed to downcast from JS value"))
    }
}
