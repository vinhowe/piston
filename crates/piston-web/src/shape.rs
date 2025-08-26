use piston::{D, Dim, Dims, RVec, Shape, ShapeWithOneHole};
use wasm_bindgen::{JsCast, JsError, JsValue};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FromJsDim(pub(crate) isize);

impl Dim for FromJsDim {
    fn to_index(&self, shape: &Shape, op: &'static str) -> anyhow::Result<usize> {
        let dim = self.0;
        let dim: Box<dyn Dim> = match dim {
            dim if dim < 0 => Box::new(D::Minus(dim.unsigned_abs())),
            _ => Box::new(dim as usize),
        };
        dim.to_index(shape, op)
    }

    fn to_index_plus_one(&self, shape: &piston::Shape, op: &'static str) -> anyhow::Result<usize> {
        let dim = self.0;
        let dim: Box<dyn Dim> = match dim {
            dim if dim < 0 => Box::new(D::Minus(dim.unsigned_abs())),
            _ => Box::new(dim as usize),
        };
        dim.to_index_plus_one(shape, op)
    }
}

pub struct FromJsVecISize(pub(crate) Vec<isize>);

impl Dims for FromJsVecISize {
    fn to_indexes_internal(self, shape: &Shape, op: &'static str) -> anyhow::Result<RVec<usize>> {
        let dims = self
            .0
            .iter()
            .map(|d| FromJsDim(*d).to_index(shape, op))
            .collect::<Result<RVec<_>, _>>()?;
        Ok(dims)
    }
}

impl FromJsVecISize {
    // Create a function instead of implementing TryFrom for Option<FromJsDims>
    pub fn from_js_value(value: JsValue) -> Result<Option<Self>, JsError> {
        if value.is_null() || value.is_undefined() {
            return Ok(None);
        }

        if let Some(num) = value.as_f64() {
            // If it's a single number
            Ok(Some(FromJsVecISize(vec![num as isize])))
        } else if value.is_array() {
            // If it's a JavaScript array
            let array = value.dyn_into::<js_sys::Array>().unwrap();
            let dims = array
                .iter()
                .filter_map(|v| v.as_f64().map(|f| f as isize))
                .collect::<Vec<_>>();
            Ok(Some(FromJsVecISize(dims)))
        } else if js_sys::Int32Array::instanceof(&value) {
            // If it's an Int32Array
            let array = js_sys::Int32Array::from(value);
            let dims = array
                .to_vec()
                .into_iter()
                .map(|v| v as isize)
                .collect::<Vec<_>>();
            Ok(Some(FromJsVecISize(dims)))
        } else {
            Err(JsError::new("Expected a number, array, or Int32Array"))
        }
    }
}

impl ShapeWithOneHole for FromJsVecISize {
    fn into_shape(self, el_count: usize) -> anyhow::Result<Shape> {
        let mut hole_count = 0;
        let mut hole_index = None;
        let mut product = 1;

        for (i, &dim) in self.0.iter().enumerate() {
            if dim == -1 {
                hole_count += 1;
                hole_index = Some(i);
                if hole_count > 1 {
                    anyhow::bail!("at most one dimension can be -1, got shape {:?}", self.0);
                }
            } else if dim <= 0 {
                anyhow::bail!(
                    "dimensions must be positive except for at most one -1, got shape {:?}",
                    self.0
                );
            } else {
                product *= dim as usize;
            }
        }

        let mut shape_vec = RVec::with_capacity(self.0.len());

        if product == 0 {
            anyhow::bail!(
                "cannot reshape tensor of {el_count} elements with a product of 0, got shape {:?}",
                self.0
            );
        }
        if !el_count.is_multiple_of(product) {
            anyhow::bail!(
                "ShapeWithOneHole.into_shape: cannot reshape tensor with {el_count} elements to shape with -1, got shape {:?}",
                self.0
            );
        }
        let inferred_dim = el_count / product;

        for (i, &dim) in self.0.iter().enumerate() {
            if hole_index == Some(i) {
                shape_vec.push(inferred_dim);
            } else {
                shape_vec.push(dim as usize);
            }
        }

        Ok(Shape::from(shape_vec))
    }
}

pub struct FromJsVecUsize(pub(crate) Vec<usize>);

impl FromJsVecUsize {
    pub fn from_js_value(value: JsValue) -> Result<Option<Self>, JsError> {
        if value.is_null() || value.is_undefined() {
            return Ok(None);
        }

        if let Some(num) = value.as_f64() {
            let f = num;
            if !f.is_finite() {
                return Err(JsError::new(
                    "Expected a finite non-negative integer for dimension",
                ));
            }
            if f < 0.0 {
                return Err(JsError::new(&format!(
                    "Expected non-negative dimension, got {f}"
                )));
            }
            if f.fract() != 0.0 {
                return Err(JsError::new(&format!(
                    "Expected integer dimension, got {f}"
                )));
            }
            if f > (usize::MAX as f64) {
                return Err(JsError::new(&format!(
                    "Dimension exceeds usize::MAX, got {f}"
                )));
            }
            Ok(Some(FromJsVecUsize(vec![f as usize])))
        } else if value.is_array() {
            let array = value.dyn_into::<js_sys::Array>().unwrap();
            let mut dims: Vec<usize> = Vec::with_capacity(array.length() as usize);
            for v in array.iter() {
                let Some(f) = v.as_f64() else {
                    return Err(JsError::new("All dimensions in the array must be numbers"));
                };
                if !f.is_finite() {
                    return Err(JsError::new(
                        "All dimensions must be finite non-negative integers",
                    ));
                }
                if f < 0.0 {
                    return Err(JsError::new(&format!(
                        "Dimensions must be non-negative, got {f}"
                    )));
                }
                if f.fract() != 0.0 {
                    return Err(JsError::new(&format!(
                        "Dimensions must be integers, got {f}"
                    )));
                }
                if f > (usize::MAX as f64) {
                    return Err(JsError::new(&format!(
                        "Dimension exceeds usize::MAX, got {f}"
                    )));
                }
                dims.push(f as usize);
            }
            Ok(Some(FromJsVecUsize(dims)))
        } else if js_sys::Int32Array::instanceof(&value) {
            let array = js_sys::Int32Array::from(value);
            let mut dims: Vec<usize> = Vec::with_capacity(array.length() as usize);
            for v in array.to_vec() {
                if v < 0 {
                    return Err(JsError::new(&format!(
                        "Dimensions must be non-negative, got {v}"
                    )));
                }
                dims.push(v as usize);
            }
            Ok(Some(FromJsVecUsize(dims)))
        } else if js_sys::Uint32Array::instanceof(&value) {
            let array = js_sys::Uint32Array::from(value);
            let dims = array
                .to_vec()
                .into_iter()
                .map(|v| v as usize)
                .collect::<Vec<_>>();
            Ok(Some(FromJsVecUsize(dims)))
        } else {
            Err(JsError::new(
                "Expected a non-negative integer, an array of non-negative integers, Int32Array, or Uint32Array",
            ))
        }
    }
}

impl From<FromJsVecUsize> for Shape {
    fn from(value: FromJsVecUsize) -> Self {
        Shape::from(value.0)
    }
}
