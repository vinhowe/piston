mod ir_fields;
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
