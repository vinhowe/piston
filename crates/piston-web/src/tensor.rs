use crate::error::IntoJsError;
use half::f16;
use paste::paste;
use piston::{DType, Device, Dim, IrScalarValue, IrValue, LazyOp, RVec, Shape, Tensor};
use piston::{TensorTypeOrScalar, TensorTypeOrScalarEnum};
use piston_macros::js_tensor_operations;
use std::collections::HashMap;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsError;

use crate::{
    device::{gpu_sync, JsDevice, GPU_DEVICE},
    dtype::JsDType,
    shape::{FromJsDim, FromJsVecISize},
};

#[wasm_bindgen(js_name = Tensor)]
pub struct JsTensor {
    pub(crate) inner: Tensor,
}

type JsTensorResult = Result<JsTensor, JsError>;

#[wasm_bindgen(js_class = Tensor)]
impl JsTensor {
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

    pub fn rope(&self, dim: usize, base: f32, offset: usize) -> JsTensorResult {}

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
            if keepdim {
                self.inner.sum_keepdim(sum_dims)
            } else {
                self.inner.sum(sum_dims)
            }
        } else {
            self.inner.sum_all()
        }
        .map_err(|e| e.into_js_error())?;
        Ok(JsTensor { inner: tensor })
    }

    pub fn mean(self, dim: Option<Dims>, keepdim: bool) -> JsTensorResult {
        let tensor = if let Some(mean_dims) = dim {
            if keepdim {
                self.inner.mean_keepdim(mean_dims)
            } else {
                self.inner.mean(mean_dims)
            }
        } else {
            self.inner.mean_all()
        }
        .map_err(|e| e.into_js_error())?;
        Ok(JsTensor { inner: tensor })
    }

    pub fn var(self, dim: Option<Dims>, keepdim: bool) -> JsTensorResult {
        let tensor = if let Some(var_dims) = dim {
            if keepdim {
                self.inner.var_keepdim(var_dims)
            } else {
                self.inner.var(var_dims)
            }
        } else {
            self.inner.var_all()
        }
        .map_err(|e| e.into_js_error())?;
        Ok(JsTensor { inner: tensor })
    }

    pub fn max(self, dim: Dim, keepdim: bool) -> JsTensorResult {
        let tensor = if keepdim {
            self.inner.max_keepdim(dim)
        } else {
            self.inner.max(dim)
        }
        .map_err(|e| e.into_js_error())?;
        Ok(JsTensor { inner: tensor })
    }

    pub fn min(self, dim: Dim, keepdim: bool) -> JsTensorResult {
        let tensor = if keepdim {
            self.inner.min_keepdim(dim)
        } else {
            self.inner.min(dim)
        }
        .map_err(|e| e.into_js_error())?;
        Ok(JsTensor { inner: tensor })
    }

    pub fn argmax(self, dim: Dim, keepdim: bool) -> JsTensorResult {
        let tensor = if keepdim {
            self.inner.argmax_keepdim(dim)
        } else {
            self.inner.argmax(dim)
        }
        .map_err(|e| e.into_js_error())?;
        Ok(JsTensor { inner: tensor })
    }

    pub fn argmin(self, dim: Dim, keepdim: bool) -> JsTensorResult {
        let tensor = if keepdim {
            self.inner.argmin_keepdim(dim)
        } else {
            self.inner.argmin(dim)
        }
        .map_err(|e| e.into_js_error())?;
        Ok(JsTensor { inner: tensor })
    }

    pub fn norm(self) -> JsTensorResult {}

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
            None => inner.squeeze_all(),
        }
        .map_err(|e| e.into_js_error())?;
        Ok(Self { inner: result })
    }

    pub fn cat(tensors: RVec<Tensor>, dim: Dim) -> JsTensorResult {}

    pub fn stack(tensors: RVec<Tensor>, dim: Dim) -> JsTensorResult {}

    pub fn permute(self, dims: Dims) -> JsTensorResult {}

    pub fn transpose(self, dim0: Dim, dim1: Dim) -> JsTensorResult {}

    pub fn t(self) -> JsTensorResult {}

    pub fn cache(self, source: Tensor, dim: Dim, offset: usize) -> JsTensorResult {}

    /// Returns a new tensor duplicating data from the original tensor. New dimensions are inserted
    /// on the left.
    pub fn broadcast_left(self, left_shape: Shape) -> JsTensorResult {}

    pub fn broadcast_to(self, shape: Shape) -> JsTensorResult {}

    pub fn index_select(self, indices: Tensor, dim: Dim) -> JsTensorResult {}

    pub fn index_write(self, src: Tensor, write_start: Dims) -> JsTensorResult {}

    pub fn where_cond(self, condition: Tensor, on_false: TensorOrScalar) -> JsTensorResult {}

    pub fn scatter_add(self, indices: Tensor, source: Tensor, dim: Dim) -> JsTensorResult {}

    pub fn index_add_(self, indices: Tensor, source: Tensor, dim: Dim) -> JsTensorResult {}

    pub fn gather(self, indices: Tensor, dim: Dim) -> JsTensorResult {}

    pub fn triu(self, k: Option<i32>) -> JsTensorResult {}

    pub fn triu_(self, k: Option<i32>) -> JsTensorResult {}

    pub fn tril(self, k: Option<i32>) -> JsTensorResult {}

    pub fn tril_(self, k: Option<i32>) -> JsTensorResult {}

    #[dtype_generic(f32, f16)]
    pub fn arange(
        start: f32,
        end: f32,
        dtype: DType,
        device: &Device,
        requires_grad: bool,
    ) -> JsTensorResult {
    }

    /// Creates a new 1D tensor with values from the interval `[start, end)` taken with a common
    /// difference `step` from `start`.
    #[dtype_generic]
    pub fn arange_step(
        start: f32,
        end: f32,
        step: f32,
        dtype: DType,
        device: &Device,
        requires_grad: bool,
    ) -> JsTensorResult {
    }

    #[dtype_generic(f32, i32, u32)]
    pub fn randint(
        low: f32,
        high: f32,
        shape: Shape,
        dtype: DType,
        device: Device,
        requires_grad: bool,
    ) -> JsTensorResult {
    }

    #[dtype_generic(f32, f16)]
    pub fn randn(
        mean: f32,
        std: f32,
        shape: Shape,
        dtype: DType,
        device: Device,
        requires_grad: bool,
    ) -> JsTensorResult {
    }

    #[dtype_generic(f32, f16)]
    pub fn rand(
        lo: f32,
        up: f32,
        shape: Shape,
        dtype: DType,
        device: Device,
        requires_grad: bool,
    ) -> JsTensorResult {
    }

    // We use the dtype of the tensor
    pub fn bernoulli(self) -> JsTensorResult {}

    pub fn bernoulli_(self) -> JsTensorResult {}

    pub fn zero_(self) -> JsTensorResult {}

    #[dtype_generic]
    pub fn zeros(
        shape: Shape,
        dtype: DType,
        device: &Device,
        requires_grad: bool,
    ) -> JsTensorResult {
    }

    #[dtype_generic]
    pub fn zeros_like(
        self,
        dtype: DType,
        device: Option<&Device>,
        requires_grad: bool,
    ) -> JsTensorResult {
    }

    #[dtype_generic]
    pub fn ones(
        shape: Shape,
        dtype: DType,
        device: &Device,
        requires_grad: bool,
    ) -> JsTensorResult {
    }

    #[dtype_generic]
    pub fn ones_like(
        self,
        dtype: DType,
        device: Option<&Device>,
        requires_grad: bool,
    ) -> JsTensorResult {
    }

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

#[wasm_bindgen(module = "/cast.js")]
extern "C" {
    #[wasm_bindgen(catch, js_name = cast)]
    fn try_into_tensor(from: &JsValue) -> Result<JsTensor, JsValue>;
}

impl TryFrom<JsValue> for JsTensor {
    type Error = JsValue;
    fn try_from(value: JsValue) -> Result<Self, Self::Error> {
        try_into_tensor(&value)
    }
}
