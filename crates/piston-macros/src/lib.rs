mod ir_fields;
mod js_tensor_web_op;
mod ops;
mod scoped_module;
mod wgsl_metadata;

use proc_macro::TokenStream;
use syn::parse_macro_input;

/// Derives the `OpMetadata` trait implementation for a struct.
///
/// Generates a `.render()` method that converts a Rust struct into a WGSL struct.
#[proc_macro_derive(WgslMetadata, attributes(builder))]
pub fn derive_wgsl_metadata(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input);
    wgsl_metadata::derive(input).into()
}

/// Derives the `IrFields` trait implementation for a struct.
///
/// Generates a `.ir_fields()` method we use for hashing compute graphs.
#[proc_macro_derive(IrFields, attributes(builder))]
pub fn derive_ir_fields(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input);
    ir_fields::derive(input).into()
}

/// Derives the `ScopedModule` trait implementation for a struct.
///
/// Automatically adds scoping to Module implementations
#[proc_macro_attribute]
pub fn scoped_module(_attr: TokenStream, item: TokenStream) -> TokenStream {
    scoped_module::scoped_module(item)
}

/// Generates tensor operation variants from an OpTensor kernel function.
///
/// This macro takes a function that operates on OpTensor and generates:
/// - A kernel function with OT generics for Into<OpTensor> parameters
/// - Optional function, method, and method_inplace variants
///
/// # Example
/// ```rust
/// #[tensor_op(variants = [function, method, method_inplace])]
/// fn add<T: TensorTypeOrScalar<OpTensor>>(lhs: OpTensor, rhs: T) -> Result<OpTensor> {
///     // implementation
/// }
/// ```
#[proc_macro_attribute]
pub fn tensor_op(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr = parse_macro_input!(attr as ops::TensorOpAttr);
    let item = parse_macro_input!(item as syn::ItemFn);

    match ops::process_tensor_op(attr, item) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// Generates wasm-bindgen-exposed functions for Tensor ops in the web crate.
///
/// See `docs/js_tensor_web_op.md` for detailed behavior. This macro is attached to a
/// typed Rust function and generates free-function exports: function/method/method_inplace.
#[proc_macro_attribute]
pub fn js_tensor_web_op(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attr = parse_macro_input!(attr as js_tensor_web_op::JsTensorWebOpAttr);
    let item = parse_macro_input!(item as syn::ItemFn);

    match js_tensor_web_op::process_js_tensor_web_op(attr, item) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}
