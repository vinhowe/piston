use piston::TensorError;
use wasm_bindgen::JsError;

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
            message.push_str(&format!("\nCaused by: {}", err));
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
