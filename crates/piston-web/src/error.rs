use piston::TensorError;
use wasm_bindgen::{JsError, JsValue};

/// Helper trait to convert anyhow::Error to JsError while preserving error information
pub trait IntoJsError {
    fn into_js_error(self) -> JsError;
}

impl IntoJsError for anyhow::Error {
    fn into_js_error(self) -> JsError {
        // Create a JavaScript Error object with the full error chain
        let mut message = self.to_string();

        // Add the error chain if available
        let mut current = self.source();
        while let Some(err) = current {
            message.push_str(&format!("\nCaused by: {err}"));
            current = err.source();
        }

        JsError::new(&message)
    }
}

impl IntoJsError for TensorError {
    fn into_js_error(self) -> JsError {
        self.into()
    }
}

impl IntoJsError for JsValue {
    fn into_js_error(self) -> JsError {
        let message = self
            .as_string()
            .unwrap_or_else(|| format!("JavaScript error: {self:?}"));
        JsError::new(&message)
    }
}

impl IntoJsError for JsError {
    fn into_js_error(self) -> JsError {
        self
    }
}
