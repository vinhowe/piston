use piston::DType;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = DType)]
pub struct JsDType {
    #[wasm_bindgen(skip)]
    pub(crate) dtype: DType,
}

#[wasm_bindgen(js_class = DType)]
impl JsDType {
    #[wasm_bindgen]
    pub fn _clone(&self) -> JsDType {
        JsDType { dtype: self.dtype }
    }

    #[wasm_bindgen(getter, js_name = isFloatingPoint)]
    pub fn is_floating_point(&self) -> bool {
        self.dtype.is_float()
    }

    #[wasm_bindgen(getter, js_name = isSigned)]
    pub fn is_signed(&self) -> bool {
        self.dtype.is_signed()
    }

    #[wasm_bindgen(getter)]
    pub fn itemsize(&self) -> usize {
        self.dtype.size_of()
    }
}

#[wasm_bindgen(getter)]
pub fn float32() -> JsDType {
    JsDType { dtype: DType::F32 }
}

#[wasm_bindgen(getter)]
pub fn float16() -> JsDType {
    JsDType { dtype: DType::F16 }
}

#[wasm_bindgen(getter)]
pub fn int32() -> JsDType {
    JsDType { dtype: DType::I32 }
}

#[wasm_bindgen(getter)]
pub fn uint32() -> JsDType {
    JsDType { dtype: DType::U32 }
}
