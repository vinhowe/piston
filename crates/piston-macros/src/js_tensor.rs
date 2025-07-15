// Slop
use heck::ToLowerCamelCase;
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{parse_macro_input, FnArg, ImplItem, ItemImpl, Pat, PatIdent, Type};

/// Converts a string to lower camel case, preserving leading and trailing underscores.
fn to_lower_camel_case_with_underscore(ident: &str) -> String {
    let leading_count = ident.chars().take_while(|&c| c == '_').count();
    let trailing_count = ident.chars().rev().take_while(|&c| c == '_').count();

    if leading_count + trailing_count >= ident.len() {
        // In the case that it's all underscores
        return ident.to_string();
    }

    format!(
        "{}{}{}",
        &ident[..leading_count],
        // Convert middle part to camel case
        &ident[leading_count..ident.len() - trailing_count].to_lower_camel_case(),
        &ident[ident.len() - trailing_count..]
    )
}

/// Core logic for js_tensor_operations using proc_macro2 types
pub fn process_js_tensor_operations(mut item_impl: ItemImpl) -> TokenStream2 {
    // Extract the name of the struct being implemented (e.g., "JsTensor")
    let js_struct_name = match &*item_impl.self_ty {
        Type::Path(type_path) => type_path.path.segments.last().unwrap().ident.to_string(),
        _ => panic!("Expected a path type for the self_ty"),
    };

    // Remove the "Js" prefix to get the underlying type (e.g., "Tensor")
    let underlying_type = if let Some(underlying_type) = js_struct_name.strip_prefix("Js") {
        underlying_type.to_string()
    } else {
        panic!("Expected a Js prefix for the self_ty");
    };

    // Create an identifier for the underlying type
    let underlying_type_ident = syn::Ident::new(&underlying_type, proc_macro2::Span::call_site());

    for impl_item in &mut item_impl.items {
        if let ImplItem::Fn(ref mut input_fn) = impl_item {
            // Look for our marker attribute #[js_tensor_operation].
            let mut has_js_tensor_skip = false;
            let is_async = input_fn.sig.asyncness.is_some();

            // Remove the skip attribute.
            input_fn.attrs.retain(|attr| {
                if attr.path().is_ident("skip") {
                    has_js_tensor_skip = true;
                    false
                } else {
                    true
                }
            });

            if has_js_tensor_skip {
                continue;
            }

            let mut has_dtype_generic = false;
            let mut has_shape_generic = false;

            let mut specified_dtypes: Option<Vec<syn::Ident>> = None;
            input_fn.attrs.retain(|attr| {
                if attr.path().is_ident("dtype_generic") {
                    has_dtype_generic = true;
                    // Try to parse the tokens as a comma-separated list of idents.
                    let dtypes = match attr.parse_args_with(
                        syn::punctuated::Punctuated::<syn::Ident, syn::Token![,]>::parse_terminated,
                    ) {
                        Ok(parsed_dtypes) => Some(parsed_dtypes.into_iter().collect()),
                        Err(_) => None,
                    };
                    specified_dtypes = dtypes;
                    false
                } else {
                    true
                }
            });

            // Determine the underlying core method name.
            let method_name = input_fn.sig.ident.clone();
            let js_method_name =
                to_lower_camel_case_with_underscore(&input_fn.sig.ident.to_string());
            input_fn
                .attrs
                .push(syn::parse_quote!(#[wasm_bindgen(js_name = #js_method_name)]));

            // Process parameters (e.g. for setting js names).
            for input in &mut input_fn.sig.inputs {
                if let FnArg::Typed(pat_type) = input {
                    if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                        let param_name = pat_ident.ident.to_string();
                        let param_js_name = to_lower_camel_case_with_underscore(&param_name);
                        if is_type(&pat_type.ty, "Shape")
                            || is_type_ref(&pat_type.ty, "Shape")
                            || is_type(&pat_type.ty, "ShapeWithOneHole")
                            || is_type_ref(&pat_type.ty, "ShapeWithOneHole")
                            || is_type(&pat_type.ty, "Dims")
                            || is_type(&pat_type.ty, "TensorOrScalar")
                            || is_optional_of_type(&pat_type.ty, "Shape")
                            || is_optional_of_type(&pat_type.ty, "Dims")
                        {
                            let mut unchecked_param_type = if is_type(&pat_type.ty, "Dims")
                                || is_optional_of_type(&pat_type.ty, "Dims")
                            {
                                "Int32Array | number[] | number".to_string()
                            } else if is_type(&pat_type.ty, "ShapeWithOneHole")
                                || is_optional_of_type(&pat_type.ty, "ShapeWithOneHole")
                            {
                                "Int32Array | number[]".to_string()
                            } else if is_type(&pat_type.ty, "TensorOrScalar") {
                                "Tensor | number".to_string()
                            } else {
                                "Uint32Array | number[]".to_string()
                            };
                            if is_type(&pat_type.ty, "Shape")
                                || is_type_ref(&pat_type.ty, "Shape")
                                || is_type(&pat_type.ty, "ShapeWithOneHole")
                                || is_type_ref(&pat_type.ty, "ShapeWithOneHole")
                            {
                                has_shape_generic = true;
                            }
                            if is_optional(&pat_type.ty) {
                                unchecked_param_type =
                                    format!("{} | null | undefined", unchecked_param_type);
                            }
                            pat_type
                                .attrs
                                // We do this because we technically accept both and we want to make that clear to typescript
                                .push(syn::parse_quote!(
                                    #[wasm_bindgen(js_name = #param_js_name, unchecked_param_type = #unchecked_param_type)]
                                ));
                        } else {
                            pat_type
                                .attrs
                                .push(syn::parse_quote!(#[wasm_bindgen(js_name = #param_js_name)]));
                        }
                    }
                }
            }

            // Generate conversion code for each parameter.
            let mut prelude_stmts = Vec::new();
            let mut call_args = Vec::new();
            let mut has_self_param = false;
            let mut to_dtype_cast_params = Vec::new();

            // Iterate over each argument in the function signature.
            // We'll store parameters that should be retained
            let mut retained_inputs = Vec::new();

            for input in &mut input_fn.sig.inputs {
                match input {
                    FnArg::Receiver(_) => {
                        has_self_param = true;
                        retained_inputs.push(input.clone());
                    }
                    FnArg::Typed(ref mut pat_type) => {
                        let pat = &*pat_type.pat;
                        let ty = &*pat_type.ty;
                        let mut is_dtype_param = false;
                        let is_reference = matches!(ty, Type::Reference(_));
                        let new_ident = if let Pat::Ident(PatIdent { ref ident, .. }) = pat {
                            // Check if this parameter is named "dtype"
                            if ident == "dtype" {
                                is_dtype_param = true;
                            }
                            syn::Ident::new(&ident.to_string(), ident.span())
                        } else {
                            syn::Ident::new("unknown", proc_macro2::Span::call_site())
                        };
                        to_dtype_cast_params.push(
                            is_type(ty, "f32")
                                || is_type(ty, "f16")
                                || is_type(ty, "i32")
                                || is_type(ty, "u32"),
                        );

                        let ident = quote! { #pat };
                        // let (new_type, prelude_type, call_expr, conversion, _is_ref) =
                        let TypeConversion {
                            signature_type: new_type,
                            prelude_type,
                            call_expr,
                            inner_conversion,
                            is_ref: _,
                            is_result: _,
                        } = generate_type_and_conversion(
                            ident.clone(),
                            ident,
                            ty,
                            None,
                            has_self_param,
                        );
                        if let Some(conv) = inner_conversion {
                            prelude_stmts.push(quote! {
                                let #new_ident: #prelude_type = #conv;
                            });
                        }
                        *pat_type.ty = new_type;
                        let ref_expr = if is_reference {
                            quote! { & }
                        } else {
                            quote! {}
                        };
                        if !(has_dtype_generic && is_dtype_param) {
                            call_args.push(quote! { #ref_expr #call_expr });
                        }
                        // Keep non-device parameters
                        retained_inputs.push(input.clone());
                    }
                }
            }

            // Replace the function inputs with the filtered list (no device param)
            input_fn.sig.inputs = syn::punctuated::Punctuated::from_iter(retained_inputs);

            let inner_call = if has_self_param {
                quote! { self.inner.clone().#method_name }
            } else {
                quote! { #underlying_type_ident::#method_name }
            };

            let result_call = if is_async {
                quote! {
                    .await.map_err(|e| e.into_js_error())?
                }
            } else {
                quote! { .map_err(|e| e.into_js_error())? }
            };

            let gen_body = if input_fn.block.stmts.is_empty() {
                if has_dtype_generic {
                    // Special handling for methods with dtype parameter.
                    let f32_args = call_args
                        .iter()
                        .zip(&to_dtype_cast_params)
                        .map(|(arg, &needs_cast)| {
                            if needs_cast {
                                quote! { #arg as f32 }
                            } else {
                                quote! { #arg }
                            }
                        })
                        .collect::<Vec<_>>();
                    let f16_args = call_args
                        .iter()
                        .zip(&to_dtype_cast_params)
                        .map(|(arg, &needs_cast)| {
                            if needs_cast {
                                quote! { f16::from_f32(#arg) }
                            } else {
                                quote! { #arg }
                            }
                        })
                        .collect::<Vec<_>>();
                    let i32_args = call_args
                        .iter()
                        .zip(&to_dtype_cast_params)
                        .map(|(arg, &needs_cast)| {
                            if needs_cast {
                                quote! { #arg as i32 }
                            } else {
                                quote! { #arg }
                            }
                        })
                        .collect::<Vec<_>>();
                    let u32_args = call_args
                        .iter()
                        .zip(&to_dtype_cast_params)
                        .map(|(arg, &needs_cast)| {
                            if needs_cast {
                                quote! { #arg as u32 }
                            } else {
                                quote! { #arg }
                            }
                        })
                        .collect::<Vec<_>>();

                    let mut arms = Vec::new();

                    let includes_f32 = specified_dtypes
                        .as_ref()
                        .is_none_or(|ds| ds.iter().any(|dt| dt == "f32"));
                    let includes_f16 = specified_dtypes
                        .as_ref()
                        .is_none_or(|ds| ds.iter().any(|dt| dt == "f16"));
                    let includes_i32 = specified_dtypes
                        .as_ref()
                        .is_none_or(|ds| ds.iter().any(|dt| dt == "i32"));
                    let includes_u32 = specified_dtypes
                        .as_ref()
                        .is_none_or(|ds| ds.iter().any(|dt| dt == "u32"));

                    let shape_generic_part = if has_shape_generic {
                        quote! { , _ }
                    } else {
                        quote! {}
                    };

                    if includes_f32 {
                        arms.push(quote! {
                            DType::F32 => #inner_call::<f32 #shape_generic_part>( #(#f32_args),* ),
                        });
                    }
                    if includes_f16 {
                        arms.push(quote! {
                            DType::F16 => #inner_call::<f16 #shape_generic_part>( #(#f16_args),* ),
                        });
                    }
                    if includes_i32 {
                        arms.push(quote! {
                            DType::I32 => #inner_call::<i32 #shape_generic_part>( #(#i32_args),* ),
                        });
                    }
                    if includes_u32 {
                        arms.push(quote! {
                            DType::U32 => #inner_call::<u32 #shape_generic_part>( #(#u32_args),* ),
                        });
                    }

                    quote! {
                        {
                            #(#prelude_stmts)*
                            let result = (match dtype {
                                #(#arms)*
                                _ => return Err(JsError::new("Unsupported dtype")),
                            }) #result_call;
                            Ok(Self { inner: result })
                        }
                    }
                } else {
                    // Standard handling for methods without a dtype parameter.
                    quote! {
                        {
                            #(#prelude_stmts)*
                            let result = #inner_call( #(#call_args),* )
                                #result_call;
                            Ok(Self { inner: result })
                        }
                    }
                }
            } else {
                let user_block = &input_fn.block;
                quote! {
                    {
                        #(#prelude_stmts)*
                        #user_block
                    }
                }
            };

            // Replace the original (empty) function body with the generated one.
            input_fn.block = syn::parse2(gen_body).expect("Failed to parse generated body");
        }
    }

    quote! {
        #item_impl
    }
}

/// Proc macro wrapper for js_tensor_operations
pub fn js_tensor_operations(item: TokenStream) -> TokenStream {
    // Parse the attribute arguments.
    let item_impl = parse_macro_input!(item as ItemImpl);

    // Process with core logic
    let result = process_js_tensor_operations(item_impl);

    TokenStream::from(result)
}

struct TypeConversion {
    signature_type: Type,
    prelude_type: proc_macro2::TokenStream,
    call_expr: proc_macro2::TokenStream,
    inner_conversion: Option<proc_macro2::TokenStream>,
    is_ref: bool,
    is_result: bool,
}

impl TypeConversion {
    fn new(
        signature_type: Type,
        prelude_type: proc_macro2::TokenStream,
        call_expr: proc_macro2::TokenStream,
        inner_conversion: Option<proc_macro2::TokenStream>,
        is_ref: bool,
        is_result: bool,
    ) -> Self {
        Self {
            signature_type,
            prelude_type,
            call_expr,
            inner_conversion,
            is_ref,
            is_result,
        }
    }
}

/// Recursively generate conversion code for a given expression and type.
/// Third return value is whether the outer type should become a reference.
fn generate_type_and_conversion(
    expr: proc_macro2::TokenStream,
    ident: proc_macro2::TokenStream,
    ty: &Type,
    outer_ty: Option<&Type>,
    has_self: bool,
) -> TypeConversion {
    if is_type(ty, "Tensor") {
        TypeConversion::new(
            syn::parse_quote!(JsTensor),
            quote! { #ty },
            ident,
            Some(quote! { #expr.inner }),
            false,
            false,
        )
    } else if is_type(ty, "TensorOrScalar") {
        TypeConversion::new(
            syn::parse_quote!(JsValue),
            quote! { JsTensorOrScalar },
            ident,
            Some(quote! { JsTensorOrScalar { inner: #expr } }),
            false,
            false,
        )
    } else if is_type(ty, "Parameter") {
        TypeConversion::new(
            syn::parse_quote!(JsParameter),
            quote! { #ty },
            ident,
            Some(quote! { #expr.inner }),
            false,
            false,
        )
    } else if is_type(ty, "Shape") {
        TypeConversion::new(
            syn::parse_quote!(Vec<usize>),
            quote! { #ty },
            ident,
            Some(quote! { Shape::from(#expr) }),
            false,
            false,
        )
    } else if is_type_ref(ty, "Shape") {
        TypeConversion::new(
            syn::parse_quote!(Vec<usize>),
            quote! { #ty },
            ident,
            Some(quote! { &Shape::from(#expr) }),
            true,
            false,
        )
    } else if is_type(ty, "ShapeWithOneHole") {
        let is_option = if let Some(outer_ty) = outer_ty {
            is_type(outer_ty, "Option")
        } else {
            false
        };
        let result_expr = if is_option {
            quote! {?}
        } else {
            quote! {?.ok_or(JsError::new("Missing required dims"))?}
        };
        TypeConversion::new(
            syn::parse_quote!(JsValue),
            quote! { FromJsVecISize },
            ident,
            Some(quote! { FromJsVecISize::from_js_value(#expr) #result_expr }),
            false,
            true,
        )
    } else if is_type(ty, "Dim") {
        TypeConversion::new(
            syn::parse_quote!(isize),
            quote! { FromJsDim },
            ident,
            Some(quote! { FromJsDim(#expr) }),
            true,
            false,
        )
    } else if is_type(ty, "Dims") {
        let is_option = if let Some(outer_ty) = outer_ty {
            is_type(outer_ty, "Option")
        } else {
            false
        };
        let result_expr = if is_option {
            quote! {?}
        } else {
            quote! {?.ok_or(JsError::new("Missing required dims"))?}
        };
        TypeConversion::new(
            syn::parse_quote!(JsValue),
            quote! { FromJsVecISize },
            ident,
            Some(quote! { FromJsVecISize::from_js_value(#expr) #result_expr }),
            false,
            true,
        )
    } else if is_type(ty, "DType") {
        TypeConversion::new(
            syn::parse_quote!(Option<JsDType>),
            quote! { #ty },
            ident,
            if has_self {
                Some(quote! { #expr.map(|d| d.dtype).unwrap_or(self.inner.dtype()) })
            } else {
                Some(quote! { #expr.map(|d| d.dtype).unwrap_or(DType::F32) })
            },
            false,
            false,
        )
    } else if is_type(ty, "Device") || is_type_ref(ty, "Device") {
        let is_option = if let Some(outer_ty) = outer_ty {
            is_type(outer_ty, "Option")
        } else {
            false
        };

        let is_ref = is_type_ref(ty, "Device");
        let ref_expr = if is_ref && !is_option {
            quote! { & }
        } else {
            quote! {}
        };

        let (inner_expr, unwrap_or_expr) = if is_option {
            (quote! {#expr.inner.clone()}, quote! {})
        } else {
            (
                quote! {#expr.map(|d| d.inner)},
                if has_self {
                    quote! { .unwrap_or(self.inner.device()) }
                } else {
                    quote! { .unwrap_or(gpu_sync()?.inner.clone()) }
                },
            )
        };

        let (ident_expr, prelude_type) = if is_ref && is_option {
            (quote! { #ident.as_ref() }, quote! { Device })
        } else {
            (quote! { #ident }, quote! { #ty })
        };

        TypeConversion::new(
            if is_option {
                syn::parse_quote!(JsDevice)
            } else {
                syn::parse_quote!(Option<JsDevice>)
            },
            prelude_type,
            ident_expr,
            Some(quote! { #ref_expr #inner_expr #unwrap_or_expr }),
            is_ref,
            false,
        )
    } else if let Some((inner_ty, outer_ty)) = get_inner_type(ty, "Option") {
        // let (new_inner_type, prelude_type, ident, inner_conversion, is_ref) =
        //     generate_type_and_conversion(quote! { x }, ident, inner_ty, Some(outer_ty), has_self);
        let TypeConversion {
            signature_type,
            prelude_type,
            call_expr,
            inner_conversion,
            is_ref: _,
            is_result,
        } = generate_type_and_conversion(
            ident.clone(),
            ident.clone(),
            inner_ty,
            Some(outer_ty),
            has_self,
        );
        let result_expr = if is_result {
            quote! { .transpose()? }
        } else {
            quote! {}
        };
        let (signature_type, inner_conversion) = if is_type(&signature_type, "JsValue") {
            // We can't wrap JsValue in an Option for whatever reason.
            (signature_type, inner_conversion)
        } else {
            (
                syn::parse_quote!(Option<#signature_type>),
                inner_conversion.map(|c| quote! { #expr.map(|#ident| { #c }) #result_expr  }),
            )
        };
        TypeConversion::new(
            signature_type,
            quote! { Option<#prelude_type> },
            call_expr,
            inner_conversion,
            false,
            false,
        )
    } else if let Some((inner_ty, outer_ty)) = get_inner_type(ty, "RVec") {
        // let (new_inner_type, prelude_type, ident, inner_conversion, _is_ref) =
        //     generate_type_and_conversion(quote! { x }, ident, inner_ty, Some(outer_ty), has_self);
        let TypeConversion {
            signature_type: new_inner_type,
            prelude_type,
            call_expr: ident,
            inner_conversion,
            is_ref: _,
            is_result: _,
        } = generate_type_and_conversion(quote! { x }, ident, inner_ty, Some(outer_ty), has_self);
        let new_type = syn::parse_quote!(Vec<#new_inner_type>);
        TypeConversion::new(
            new_type,
            quote! { RVec<#prelude_type> },
            ident,
            inner_conversion
                .map(|c| quote! { #expr.into_iter().map(|x| { #c }).collect() })
                .or(Some(quote! { #expr.into_iter().collect() })),
            false,
            false,
        )
    } else {
        TypeConversion::new(ty.clone(), quote! { #ty }, ident, None, false, false)
    }
}

/// Helper: if the type is a container (e.g. Option or Vec), return its inner type.
fn get_inner_type<'a>(ty: &'a Type, container: &str) -> Option<(&'a Type, &'a Type)> {
    if let Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.first() {
            if segment.ident == container {
                if let syn::PathArguments::AngleBracketed(angle_bracketed) = &segment.arguments {
                    if let Some(syn::GenericArgument::Type(inner_ty)) = angle_bracketed.args.first()
                    {
                        return Some((inner_ty, ty));
                    }
                }
            }
        }
    }
    None
}

/// Helper: checks whether a type is exactly the given expected type (e.g. "JsTensor").
fn is_type(ty: &Type, expected: &str) -> bool {
    if let Type::Path(type_path) = ty {
        if type_path.qself.is_none() && type_path.path.segments.len() == 1 {
            return type_path.path.segments[0].ident == expected;
        }
    }
    false
}

/// Helper: checks whether a type is an Option of the expected type (e.g. "Option<Tensor>").
fn is_optional_of_type(ty: &Type, expected: &str) -> bool {
    if let Some((Type::Path(type_path), _)) = get_inner_type(ty, "Option") {
        if type_path.qself.is_none() && type_path.path.segments.len() == 1 {
            return type_path.path.segments[0].ident == expected;
        }
    }
    false
}

fn is_optional(ty: &Type) -> bool {
    if let Some((_, _)) = get_inner_type(ty, "Option") {
        return true;
    }
    false
}

/// Helper: checks whether a type is a reference to the expected type (e.g. "&Shape").
fn is_type_ref(ty: &Type, expected: &str) -> bool {
    if let Type::Reference(type_ref) = ty {
        if let Type::Path(type_path) = &*type_ref.elem {
            if type_path.qself.is_none() && type_path.path.segments.len() == 1 {
                return type_path.path.segments[0].ident == expected;
            }
        }
    }
    false
}
