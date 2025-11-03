use heck::{ToLowerCamelCase, ToUpperCamelCase};
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::{Attribute, Expr, FnArg, Ident, ItemFn, LitStr, Result as SynResult, Token, Type};

fn to_lower_camel_case_with_underscore(ident: &str) -> String {
    let leading_count = ident.chars().take_while(|&c| c == '_').count();
    let trailing_count = ident.chars().rev().take_while(|&c| c == '_').count();

    if leading_count + trailing_count >= ident.len() {
        return ident.to_string();
    }

    format!(
        "{}{}{}",
        &ident[..leading_count],
        &ident[leading_count..ident.len() - trailing_count].to_lower_camel_case(),
        &ident[ident.len() - trailing_count..]
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum VariantKind {
    Function,
    Method,
    MethodInplace,
}

impl VariantKind {
    fn as_js_export_name(self, op_name_pascal: &str) -> String {
        match self {
            // Function exports should be lowerCamelCase of the op name (e.g., zerosLike)
            VariantKind::Function => op_name_pascal.to_lower_camel_case(),
            VariantKind::Method => format!("opMethod{op_name_pascal}"),
            VariantKind::MethodInplace => format!("opMethod{op_name_pascal}_"),
        }
    }
}

#[derive(Default)]
pub struct JsTensorWebOpAttr {
    name: String,
    variants: Vec<VariantKind>,
    dtype_generic: Option<Vec<Ident>>, // parsed but not used initially
    getter: bool,
    setter: bool,
    js_name: Option<String>,
    target: Option<String>,
}

impl Parse for JsTensorWebOpAttr {
    fn parse(input: ParseStream) -> SynResult<Self> {
        let mut name: Option<String> = None;
        let mut variants: Option<Vec<VariantKind>> = None;
        let mut dtype_generic: Option<Vec<Ident>> = None;
        let mut getter: bool = false;
        let mut setter: bool = false;
        let mut js_name: Option<String> = None;
        let mut target: Option<String> = None;

        let parser = syn::punctuated::Punctuated::<syn::MetaNameValue, Token![,]>::parse_terminated;
        // Try a simple name-value list first, but only if it consumes the entire input
        let fork = input.fork();
        let list_res = parser(&fork);
        if let Ok(list) = list_res
            && fork.is_empty()
        {
            // consume
            let _ = parser(input)?;
            for nv in list {
                if nv.path.is_ident("name") {
                    match nv.value {
                        syn::Expr::Lit(syn::ExprLit {
                            lit: syn::Lit::Str(ls),
                            ..
                        }) => {
                            name = Some(ls.value());
                        }
                        syn::Expr::Path(ref p) => {
                            // Allow identifiers (e.g., passed from macro_rules) like name = Add
                            if let Some(seg) = p.path.segments.last() {
                                name = Some(seg.ident.to_string());
                            } else {
                                return Err(syn::Error::new_spanned(
                                    &nv.value,
                                    "invalid ident for name",
                                ));
                            }
                        }
                        _ => {
                            return Err(syn::Error::new_spanned(
                                nv,
                                "name must be a string literal",
                            ));
                        }
                    }
                } else if nv.path.is_ident("variants") {
                    // Expect array literal like [function, method]
                    match nv.value {
                        syn::Expr::Array(arr) => {
                            let mut vs = Vec::new();
                            for elem in arr.elems {
                                match elem {
                                    syn::Expr::Path(p) => {
                                        let ident =
                                            p.path.segments.last().unwrap().ident.to_string();
                                        let v = match ident.as_str() {
                                            "function" => VariantKind::Function,
                                            "method" => VariantKind::Method,
                                            "method_inplace" => VariantKind::MethodInplace,
                                            other => {
                                                return Err(syn::Error::new_spanned(
                                                    p,
                                                    format!("Unknown variant '{other}'"),
                                                ));
                                            }
                                        };
                                        vs.push(v);
                                    }
                                    other => {
                                        return Err(syn::Error::new_spanned(
                                            other,
                                            "variants expects an array of identifiers",
                                        ));
                                    }
                                }
                            }
                            variants = Some(vs);
                        }
                        other => {
                            return Err(syn::Error::new_spanned(
                                other,
                                "variants must be an array literal, e.g. [function, method]",
                            ));
                        }
                    }
                } else if nv.path.is_ident("dtype_generic") {
                    // dtype_generic() or dtype_generic(f32, f16)
                    match nv.value {
                        syn::Expr::Tuple(t) => {
                            let mut dts = Vec::new();
                            for elem in t.elems {
                                if let syn::Expr::Path(p) = elem {
                                    dts.push(p.path.segments.last().unwrap().ident.clone());
                                }
                            }
                            dtype_generic = Some(dts);
                        }
                        syn::Expr::Path(_) => {
                            dtype_generic = Some(Vec::new());
                        }
                        other => {
                            return Err(syn::Error::new_spanned(
                                other,
                                "dtype_generic must be like dtype_generic or dtype_generic(f32, f16)",
                            ));
                        }
                    }
                } else if nv.path.is_ident("js_name") {
                    match nv.value {
                        syn::Expr::Lit(syn::ExprLit {
                            lit: syn::Lit::Str(ls),
                            ..
                        }) => {
                            js_name = Some(ls.value());
                        }
                        syn::Expr::Path(ref p) => {
                            if let Some(seg) = p.path.segments.last() {
                                js_name = Some(seg.ident.to_string());
                            } else {
                                return Err(syn::Error::new_spanned(
                                    &nv.value,
                                    "invalid ident for js_name",
                                ));
                            }
                        }
                        _ => {
                            return Err(syn::Error::new_spanned(
                                nv,
                                "js_name must be a string literal",
                            ));
                        }
                    }
                } else if nv.path.is_ident("target") {
                    match nv.value {
                        syn::Expr::Lit(syn::ExprLit {
                            lit: syn::Lit::Str(ls),
                            ..
                        }) => {
                            target = Some(ls.value());
                        }
                        syn::Expr::Path(ref p) => {
                            if let Some(seg) = p.path.segments.last() {
                                target = Some(seg.ident.to_string());
                            } else {
                                return Err(syn::Error::new_spanned(
                                    &nv.value,
                                    "invalid ident for target",
                                ));
                            }
                        }
                        _ => {
                            return Err(syn::Error::new_spanned(
                                nv,
                                "target must be a string literal",
                            ));
                        }
                    }
                } else if nv.path.is_ident("getter") {
                    match nv.value {
                        syn::Expr::Lit(syn::ExprLit {
                            lit: syn::Lit::Bool(b),
                            ..
                        }) => {
                            getter = b.value;
                        }
                        _ => return Err(syn::Error::new_spanned(nv, "getter must be a boolean")),
                    }
                } else if nv.path.is_ident("setter") {
                    match nv.value {
                        syn::Expr::Lit(syn::ExprLit {
                            lit: syn::Lit::Bool(b),
                            ..
                        }) => {
                            setter = b.value;
                        }
                        _ => return Err(syn::Error::new_spanned(nv, "setter must be a boolean")),
                    }
                } else {
                    return Err(syn::Error::new_spanned(
                        nv,
                        "Unknown attribute key for js_tensor_web_op",
                    ));
                }
            }
        } else {
            // Named arguments using nested meta, e.g. name = "Addcdiv", variants = [...]
            let mut parsed_any = false;
            while !input.is_empty() {
                parsed_any = true;
                let lookahead = input.lookahead1();
                if lookahead.peek(syn::Ident) {
                    let ident: Ident = input.parse()?;
                    if ident == "getter" {
                        getter = true;
                        let _ = input.parse::<Token![,]>();
                        continue;
                    } else if ident == "setter" {
                        setter = true;
                        let _ = input.parse::<Token![,]>();
                        continue;
                    }
                    input.parse::<Token![=]>()?;
                    if ident == "name" {
                        // Accept either a string literal or a bare ident here
                        if input.peek(LitStr) {
                            let lit: LitStr = input.parse()?;
                            name = Some(lit.value());
                        } else if input.peek(Ident) {
                            let id2: Ident = input.parse()?;
                            name = Some(id2.to_string());
                        } else {
                            return Err(syn::Error::new(
                                Span::call_site(),
                                "name must be a string literal or ident",
                            ));
                        }
                    } else if ident == "js_name" {
                        if input.peek(LitStr) {
                            let lit: LitStr = input.parse()?;
                            js_name = Some(lit.value());
                        } else if input.peek(Ident) {
                            let id2: Ident = input.parse()?;
                            js_name = Some(id2.to_string());
                        } else {
                            return Err(syn::Error::new(
                                Span::call_site(),
                                "js_name must be a string literal or ident",
                            ));
                        }
                    } else if ident == "variants" {
                        let content;
                        syn::bracketed!(content in input);
                        let elems =
                            syn::punctuated::Punctuated::<syn::Expr, Token![,]>::parse_terminated(
                                &content,
                            )?;
                        let mut vs = Vec::new();
                        for elem in elems {
                            if let syn::Expr::Path(p) = elem {
                                let id = p.path.segments.last().unwrap().ident.to_string();
                                let v = match id.as_str() {
                                    "function" => VariantKind::Function,
                                    "method" => VariantKind::Method,
                                    "method_inplace" => VariantKind::MethodInplace,
                                    _ => return Err(syn::Error::new_spanned(p, "Unknown variant")),
                                };
                                vs.push(v);
                            } else {
                                return Err(syn::Error::new_spanned(
                                    elem,
                                    "variants expects identifiers",
                                ));
                            }
                        }
                        variants = Some(vs);
                    } else if ident == "dtype_generic" {
                        // Accept optional parens with items
                        if input.peek(syn::token::Paren) {
                            let content;
                            syn::parenthesized!(content in input);
                            let elems =
                                syn::punctuated::Punctuated::<Ident, Token![,]>::parse_terminated(
                                    &content,
                                )?;
                            dtype_generic = Some(elems.into_iter().collect());
                        } else {
                            dtype_generic = Some(Vec::new());
                        }
                    } else if ident == "target" {
                        if input.peek(LitStr) {
                            let lit: LitStr = input.parse()?;
                            target = Some(lit.value());
                        } else if input.peek(Ident) {
                            let id2: Ident = input.parse()?;
                            target = Some(id2.to_string());
                        } else {
                            return Err(syn::Error::new(
                                Span::call_site(),
                                "target must be a string literal or ident",
                            ));
                        }
                    } else {
                        return Err(syn::Error::new_spanned(ident, "Unknown key"));
                    }
                    // Optional trailing comma
                    let _ = input.parse::<Option<Token![,]>>();
                } else {
                    return Err(lookahead.error());
                }
            }
            if !parsed_any {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "Expected attributes for js_tensor_web_op",
                ));
            }
        }

        let name = name.ok_or_else(|| {
            syn::Error::new(
                Span::call_site(),
                "js_tensor_web_op requires `name = \"PascalCase\"`",
            )
        })?;
        let variants = variants.unwrap_or_else(|| vec![VariantKind::Function, VariantKind::Method]);
        if getter && setter {
            return Err(syn::Error::new(
                Span::call_site(),
                "js_tensor_web_op: cannot set both getter and setter",
            ));
        }
        Ok(Self {
            name,
            variants,
            dtype_generic,
            getter,
            setter,
            js_name,
            target,
        })
    }
}

#[derive(Default, Clone)]
struct OpParamMeta {
    default_expr: Option<Expr>,
    keyword: bool,
    raw_js: bool,
    ts_type_override: Option<String>,
    name: Option<String>,
}

fn parse_op_param_meta(attrs: &[Attribute]) -> SynResult<OpParamMeta> {
    let mut meta = OpParamMeta::default();
    for attr in attrs {
        if attr.path().is_ident("op") {
            attr.parse_nested_meta(|nested| {
                if nested.path.is_ident("default") {
                    let value: Expr = nested.value()?.parse()?;
                    meta.default_expr = Some(value);
                    Ok(())
                } else if nested.path.is_ident("keyword") {
                    meta.keyword = true;
                    Ok(())
                } else if nested.path.is_ident("raw_js") {
                    meta.raw_js = true;
                    Ok(())
                } else if nested.path.is_ident("unchecked_type") || nested.path.is_ident("type") {
                    let lit: LitStr = nested.value()?.parse()?;
                    meta.ts_type_override = Some(lit.value());
                    Ok(())
                } else if nested.path.is_ident("name") {
                    let lit: LitStr = nested.value()?.parse()?;
                    meta.name = Some(lit.value());
                    Ok(())
                } else {
                    Err(nested.error("Unknown op(...) attribute key"))
                }
            })?;
        }
    }
    Ok(meta)
}

#[derive(Clone)]
struct ParamInfo {
    ident: Ident,
    ty: Type,
    is_option: bool,
    is_self_tensor: bool,
    meta: OpParamMeta,
}

fn is_type(ty: &Type, expected: &str) -> bool {
    if let Type::Path(tp) = ty
        && tp.qself.is_none()
        && tp.path.segments.len() == 1
    {
        return tp.path.segments[0].ident == expected;
    }
    false
}

fn get_inner_type<'a>(ty: &'a Type, container: &str) -> Option<(&'a Type, &'a Type)> {
    if let Type::Path(type_path) = ty
        && let Some(segment) = type_path.path.segments.first()
        && segment.ident == container
        && let syn::PathArguments::AngleBracketed(angle_bracketed) = &segment.arguments
        && let Some(syn::GenericArgument::Type(inner_ty)) = angle_bracketed.args.first()
    {
        return Some((inner_ty, ty));
    }
    None
}

fn is_optional_of_type(ty: &Type, expected: &str) -> bool {
    if let Some((Type::Path(type_path), _)) = get_inner_type(ty, "Option")
        && type_path.qself.is_none()
        && type_path.path.segments.len() == 1
    {
        return type_path.path.segments[0].ident == expected;
    }
    false
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ParamKind {
    Tensor,
    VecTensor,
    TensorOrScalar,
    Shape,
    ShapeWithOneHole,
    Dims,
    Dim,
    NormOrd,
    DType,
    Device,
    Bool,
    Usize,
    I32,
    U32,
    F32,
    Unknown,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct KindMeta {
    optional: bool,
}

fn classify_param_kind(ty: &Type) -> (ParamKind, KindMeta) {
    // Unwrap Option<T>
    if let Some((inner, _)) = get_inner_type(ty, "Option") {
        let (kind, _meta) = classify_param_kind(inner);
        return (kind, KindMeta { optional: true });
    }
    // Vec<Tensor>
    if let Some((inner, _)) = get_inner_type(ty, "Vec")
        && is_type(inner, "Tensor")
    {
        return (ParamKind::VecTensor, KindMeta { optional: false });
    }
    // Base kinds
    let kind = if is_type(ty, "Tensor") {
        ParamKind::Tensor
    } else if is_type(ty, "TensorOrScalar") {
        ParamKind::TensorOrScalar
    } else if is_type(ty, "Shape") {
        ParamKind::Shape
    } else if is_type(ty, "ShapeWithOneHole") {
        ParamKind::ShapeWithOneHole
    } else if is_type(ty, "Dims") {
        ParamKind::Dims
    } else if is_type(ty, "Dim") {
        ParamKind::Dim
    } else if is_type(ty, "NormOrd") {
        ParamKind::NormOrd
    } else if is_type(ty, "DType") {
        ParamKind::DType
    } else if is_type(ty, "Device") {
        ParamKind::Device
    } else if is_type(ty, "bool") {
        ParamKind::Bool
    } else if is_type(ty, "usize") {
        ParamKind::Usize
    } else if is_type(ty, "i32") {
        ParamKind::I32
    } else if is_type(ty, "u32") {
        ParamKind::U32
    } else if is_type(ty, "f32") {
        ParamKind::F32
    } else {
        ParamKind::Unknown
    };
    (kind, KindMeta { optional: false })
}

fn is_return_type_option(func: &ItemFn) -> bool {
    // Returns (is_option, inner_is_js_tensor)
    // Look at the function signature return type and detect if Ok type is Option<Tensor> or Option<JsTensor>
    use syn::ReturnType;
    let mut ty_opt: Option<&Type> = None;
    if let ReturnType::Type(_, ty) = &func.sig.output {
        ty_opt = Some(ty);
    }
    if let Some(ty) = ty_opt {
        // Unwrap Result<Ok, Err> -> Ok
        let ok_ty = if let Some((inner, _)) = get_inner_type(ty, "Result") {
            inner
        } else {
            ty
        };
        // Check Option<...>
        return is_optional_of_type(ok_ty, "JsTensor");
    }
    false
}

fn is_return_type_vec_tensor(func: &ItemFn) -> bool {
    use syn::ReturnType;
    let mut ty_opt: Option<&Type> = None;
    if let ReturnType::Type(_, ty) = &func.sig.output {
        ty_opt = Some(ty);
    }
    if let Some(ty) = ty_opt {
        // Unwrap Result<Ok, Err> -> Ok
        let ok_ty = if let Some((inner, _)) = get_inner_type(ty, "Result") {
            inner
        } else {
            ty
        };
        // Vec<Tensor>
        if let Some((inner, _)) = get_inner_type(ok_ty, "Vec") {
            return is_type(inner, "JsTensor") || is_type(inner, "Tensor");
        }
    }
    false
}

fn param_unchecked_ts_type(ty: &Type, optional: bool) -> String {
    let (kind, meta) = classify_param_kind(ty);
    let is_optional = optional || meta.optional;
    let mut s = match kind {
        ParamKind::Tensor => "Tensor".to_string(),
        ParamKind::VecTensor => "Tensor[]".to_string(),
        ParamKind::TensorOrScalar => "Tensor | number".to_string(),
        ParamKind::NormOrd => {
            "'fro' | 'inf' | '-inf' | '0' | '1' | '-1' | '2' | number".to_string()
        }
        ParamKind::Shape => "Uint32Array | number[] | number".to_string(),
        ParamKind::ShapeWithOneHole | ParamKind::Dims => {
            "Int32Array | number[] | number".to_string()
        }
        ParamKind::Dim => "number".to_string(),
        ParamKind::Device => "Device | 'cpu' | 'gpu' | 'webgpu'".to_string(),
        ParamKind::DType => "DType".to_string(),
        ParamKind::Bool => "boolean".to_string(),
        ParamKind::Usize | ParamKind::I32 | ParamKind::U32 | ParamKind::F32 => "number".to_string(),
        ParamKind::Unknown => "unknown".to_string(),
    };
    if is_optional {
        s.push_str(" | null | undefined");
    }
    s
}

fn options_field_ts_type_for_ty(ty: &Type) -> TokenStream2 {
    let (kind, _meta) = classify_param_kind(ty);
    match kind {
        ParamKind::Tensor => quote! { crate::tensor::JsTensor },
        ParamKind::TensorOrScalar => quote! { crate::tensor::JsTensorOrScalar },
        ParamKind::Shape => quote! { Vec<usize> },
        ParamKind::NormOrd | ParamKind::ShapeWithOneHole | ParamKind::Dims | ParamKind::Dim => {
            quote! { JsValue }
        }
        ParamKind::DType => quote! { crate::dtype::JsDType },
        ParamKind::Device => quote! { crate::device::JsDevice },
        _ => quote! { #ty },
    }
}

fn make_param_info_list(func: &ItemFn) -> SynResult<Vec<ParamInfo>> {
    let mut params = Vec::new();
    for (idx, arg) in func.sig.inputs.iter().enumerate() {
        match arg {
            FnArg::Receiver(_) => {
                return Err(syn::Error::new_spanned(
                    arg,
                    "js_tensor_web_op must be used on a free function (no self)",
                ));
            }
            FnArg::Typed(pt) => {
                let ident = match &*pt.pat {
                    syn::Pat::Ident(pat_ident) => pat_ident.ident.clone(),
                    _ => format_ident!("arg{idx}"),
                };
                let ty = (*pt.ty).clone();
                let meta = parse_op_param_meta(&pt.attrs)?;
                let is_option = get_inner_type(&ty, "Option").is_some();
                // Heuristic: The first param is considered self in method variant generation.
                let is_self_tensor =
                    idx == 0 && (is_type(&ty, "Tensor") || is_optional_of_type(&ty, "Tensor"));
                params.push(ParamInfo {
                    ident,
                    ty,
                    is_option,
                    is_self_tensor,
                    meta,
                });
            }
        }
    }
    Ok(params)
}

fn find_pack_start(params: &[ParamInfo]) -> Option<usize> {
    let mut pack_start: Option<usize> = None;
    for (i, p) in params.iter().enumerate() {
        if p.meta.keyword && pack_start.is_none() {
            pack_start = Some(i);
        }
        if p.is_option && pack_start.is_none() {
            pack_start = Some(i);
        }
        if p.meta.default_expr.is_some() && pack_start.is_none() {
            pack_start = Some(i);
        }
    }
    pack_start
}

fn last_param_is_tensor_options(params: &[ParamInfo]) -> bool {
    if let Some(last) = params.last() {
        return is_type(&last.ty, "TensorOptions");
    }
    false
}

fn build_options_struct(
    op_name_pascal: &str,
    params: &[ParamInfo],
    pack_start: usize,
    include_tensor_options: bool,
) -> TokenStream2 {
    let options_ident = format_ident!("{}Options", op_name_pascal);
    let mut fields_ts = Vec::<TokenStream2>::new();

    for (idx, p) in params.iter().enumerate() {
        if idx < pack_start {
            continue;
        }
        // If trailing TensorOptions sentinel, we don't add it as raw field; we'll inject special fields later
        if include_tensor_options && idx == params.len() - 1 && is_type(&p.ty, "TensorOptions") {
            continue;
        }
        let name_ident = p.ident.clone();
        let name_str = name_ident.to_string();
        let camel = if let Some(ref custom) = p.meta.name {
            custom.clone()
        } else {
            name_str.to_lower_camel_case()
        };
        // Required fields remain non-Option; optional fields become Option<...>
        let is_optional = p.is_option || p.meta.default_expr.is_some();
        // Avoid Option<Option<T>> in the generated options struct by mapping Option<T> to T first
        let ty_for_mapping: &Type = if p.is_option {
            if let Some((inner, _)) = get_inner_type(&p.ty, "Option") {
                inner
            } else {
                &p.ty
            }
        } else {
            &p.ty
        };
        let rust_field_ty = options_field_ts_type_for_ty(ty_for_mapping);
        let field_ty_tokens = if is_optional && rust_field_ty.to_string() != "JsValue" {
            quote! { Option<#rust_field_ty> }
        } else {
            quote! { #rust_field_ty }
        };
        let mut attrs = Vec::<TokenStream2>::new();

        // Rename if camelCase differs from snake_case
        if camel != name_str {
            let rename_lit = syn::LitStr::new(&camel, Span::call_site());
            attrs.push(quote! { #[serde(rename = #rename_lit)] });
        }

        // Override tsify(type=...) mapping if provided
        let lit = if let Some(ref ts_override) = p.meta.ts_type_override {
            syn::LitStr::new(ts_override, Span::call_site())
        } else {
            // Provide a default tsify(type=...) mapping based on Rust type when it is a JsValue-backed field
            let ts_str = param_unchecked_ts_type(&p.ty, p.is_option);
            // We don't want to add redundant "| null | undefined" for required fields in struct typing
            let ts_clean = if is_optional {
                ts_str
            } else {
                // TODO(vinhowe): Not sure I love this
                ts_str.replace(" | null | undefined", "")
            };
            syn::LitStr::new(&ts_clean, Span::call_site())
        };

        attrs.push(quote! { #[tsify(type = #lit)] });

        // Determine serde `with` strategy for JsValue-backed fields.
        // - For Tensor and TensorOrScalar (and their Option variants), use
        //   crate::js_util::try_from_js_value_preserve
        // - For other JsValue-backed types (Dims, Dim, ShapeWithOneHole, NormOrd), use
        //   serde_wasm_bindgen::preserve
        let is_tensor_like = is_type(&p.ty, "Tensor")
            || is_optional_of_type(&p.ty, "Tensor")
            || is_type(&p.ty, "TensorOrScalar")
            || is_optional_of_type(&p.ty, "TensorOrScalar");

        let is_other_jsvalue_field = is_type(&p.ty, "ShapeWithOneHole")
            || is_optional_of_type(&p.ty, "ShapeWithOneHole")
            || is_type(&p.ty, "Dims")
            || is_optional_of_type(&p.ty, "Dims")
            || is_type(&p.ty, "Dim")
            || is_optional_of_type(&p.ty, "Dim")
            || is_type(&p.ty, "NormOrd")
            || is_optional_of_type(&p.ty, "NormOrd");

        if is_optional {
            attrs.push(quote! { #[tsify(optional)] });
        }

        let with_mod_lit: Option<syn::LitStr> = if is_tensor_like {
            Some(syn::LitStr::new(
                "crate::js_util::try_from_js_value_preserve",
                Span::call_site(),
            ))
        } else if is_other_jsvalue_field {
            Some(syn::LitStr::new(
                "serde_wasm_bindgen::preserve",
                Span::call_site(),
            ))
        } else {
            None
        };

        if let Some(with_mod) = with_mod_lit {
            if is_optional {
                attrs.push(quote! { #[serde(default, with = #with_mod)] });
            } else {
                attrs.push(quote! { #[serde(with = #with_mod)] });
            }
        } else if is_optional {
            attrs.push(quote! { #[serde(default)] });
        }

        fields_ts.push(quote! {
            #(#attrs)*
            pub #name_ident: #field_ty_tokens
        });
    }

    if include_tensor_options {
        fields_ts.push(quote! {
            #[serde(default, with = "crate::js_util::try_from_js_value_preserve")]
            #[tsify(optional, type = "DType")]
            pub dtype: Option<crate::dtype::JsDType>
        });
        fields_ts.push(quote! {
            #[serde(default, with = "crate::js_util::try_from_js_value_preserve")]
            #[tsify(optional, type = "Device")]
            pub device: Option<crate::device::JsDevice>
        });
        fields_ts.push(quote! {
            #[serde(default, rename = "requiresGrad")]
            #[tsify(optional)]
            pub requires_grad: Option<bool>
        });
    }

    quote! {
        #[derive(tsify::Tsify, serde::Serialize, serde::Deserialize, Default)]
        #[tsify(into_wasm_abi, from_wasm_abi)]
        pub struct #options_ident {
            #(#fields_ts,)*
        }
    }
}

fn conversion_from_jsvalue(
    ident: &Ident,
    ty: &Type,
    _has_self: bool,
    fn_name: &str,
) -> (TokenStream2, TokenStream2) {
    // Returns (prelude code, call expr) for building typed variable named ident
    let (kind, meta) = classify_param_kind(ty);
    match kind {
        ParamKind::Tensor => {
            if meta.optional {
                (
                    quote! {
                        let #ident: Option<piston::Tensor> = if #ident.is_undefined() || #ident.is_null() {
                            None
                        } else {
                            Some(crate::tensor::JsTensor::try_from(#ident.clone())?.inner())
                        };
                    },
                    quote! { #ident },
                )
            } else {
                (
                    quote! { let #ident: piston::Tensor = crate::tensor::JsTensor::try_from(#ident.clone())?.inner(); },
                    quote! { #ident },
                )
            }
        }
        ParamKind::VecTensor => {
            if meta.optional {
                (
                    quote! {
                        let #ident: Option<Vec<piston::Tensor>> = if #ident.is_undefined() || #ident.is_null() {
                            None
                        } else if #ident.is_array() {
                            let array = #ident.dyn_into::<js_sys::Array>().map_err(|_| wasm_bindgen::JsError::new("Expected an array of Tensors"))?;
                            Some(array
                                .iter()
                                .map(|v| crate::tensor::JsTensor::try_from(v.clone()).map(|t| t.inner()))
                                .collect::<Result<Vec<_>, wasm_bindgen::JsError>>()?)
                        } else {
                            return Err(wasm_bindgen::JsError::new("Expected an array of Tensors"));
                        };
                    },
                    quote! { #ident },
                )
            } else {
                (
                    quote! {
                        let #ident: Vec<piston::Tensor> = if #ident.is_array() {
                            let array = #ident.dyn_into::<js_sys::Array>().map_err(|_| wasm_bindgen::JsError::new("Expected an array of Tensors"))?;
                            array
                                .iter()
                                .map(|v| crate::tensor::JsTensor::try_from(v.clone()).map(|t| t.inner()))
                                .collect::<Result<Vec<_>, wasm_bindgen::JsError>>()?
                        } else {
                            return Err(wasm_bindgen::JsError::new("Expected an array of Tensors"));
                        };
                    },
                    quote! { #ident },
                )
            }
        }
        ParamKind::TensorOrScalar => (
            quote! { let #ident: crate::tensor::JsTensorOrScalar = crate::tensor::JsTensorOrScalar { inner: #ident.clone() }; },
            quote! { #ident.tensor_or_scalar().map_err(|e| e.into_js_error())? },
        ),
        ParamKind::Shape => {
            if meta.optional {
                (
                    quote! {
                        let #ident: Option<piston::Shape> = crate::shape::FromJsVecUsize::from_js_value(#ident.clone())?.map(|v| v.into());
                    },
                    quote! { #ident },
                )
            } else {
                (
                    quote! {
                        let #ident: piston::Shape = crate::shape::FromJsVecUsize::from_js_value(#ident.clone())?.ok_or(wasm_bindgen::JsError::new("Missing required dims"))?.into();
                    },
                    quote! { #ident },
                )
            }
        }
        ParamKind::ShapeWithOneHole | ParamKind::Dims => {
            if meta.optional {
                (
                    quote! { let #ident: Option<crate::shape::FromJsVecISize> = crate::shape::FromJsVecISize::from_js_value(#ident.clone())?; },
                    quote! { #ident },
                )
            } else {
                (
                    quote! { let #ident: crate::shape::FromJsVecISize = crate::shape::FromJsVecISize::from_js_value(#ident.clone())?.ok_or(wasm_bindgen::JsError::new("Missing required dims"))?; },
                    quote! { #ident },
                )
            }
        }
        ParamKind::NormOrd => (
            quote! { let #ident: Option<piston::NormOrd> = crate::tensor::js_value_to_norm_ord(#ident.clone())?; },
            quote! { #ident },
        ),
        ParamKind::Dim => (
            quote! { let #ident = crate::shape::FromJsDim(#ident.as_f64().ok_or(wasm_bindgen::JsError::new(format!("dim must be a number; got {:?} (in {})", #ident, #fn_name).as_str()))? as isize); },
            quote! { #ident },
        ),
        ParamKind::Bool => (
            quote! { let #ident: bool = #ident.as_bool().unwrap_or(false); },
            quote! { #ident },
        ),
        ParamKind::Usize => (
            quote! { let #ident: usize = #ident.as_f64().ok_or(wasm_bindgen::JsError::new("expected number"))? as usize; },
            quote! { #ident },
        ),
        ParamKind::I32 => (
            quote! { let #ident: i32 = #ident.as_f64().ok_or(wasm_bindgen::JsError::new("expected number"))? as i32; },
            quote! { #ident },
        ),
        ParamKind::U32 => (
            quote! { let #ident: u32 = #ident.as_f64().ok_or(wasm_bindgen::JsError::new("expected number"))? as u32; },
            quote! { #ident },
        ),
        ParamKind::F32 => (
            quote! { let #ident: f32 = #ident.as_f64().ok_or(wasm_bindgen::JsError::new("expected number"))? as f32; },
            quote! { #ident },
        ),
        ParamKind::DType => (
            quote! { let #ident: piston::DType = crate::dtype::JsDType::try_from(#ident.clone())?.dtype; },
            quote! { #ident },
        ),
        ParamKind::Device => (
            quote! { let #ident: piston::Device = crate::device::JsDevice::try_from(#ident.clone())?.inner; },
            quote! { #ident },
        ),
        ParamKind::Unknown => (quote! {}, quote! { #ident }),
    }
}

fn build_positional_param_defs(
    params: &[ParamInfo],
    pack_start: usize,
    exported_param_idents: &[Ident],
    is_method: bool,
    fn_name: &str,
) -> (TokenStream2, Vec<TokenStream2>, Vec<TokenStream2>) {
    // Returns (prelude, typed_arg_exprs, overloaded_js_args)
    let mut prelude = Vec::<TokenStream2>::new();
    let mut typed_args = Vec::<TokenStream2>::new();
    let mut overloaded_js_args = Vec::<TokenStream2>::new();

    let mut js_idx = if is_method { 1 } else { 0 };
    for (idx, p) in params.iter().enumerate() {
        if idx >= pack_start {
            break;
        }
        // For method variants, the first Rust param is the JS `input`/self and is handled separately.
        if is_method && idx == 0 {
            // Do not generate a typed `input` here to avoid shadowing the JS `input` &JsValue.
            // Also, do not include it in typed positional args; the call uses `self_tensor` built later.
            continue;
        }
        let js_ident = &exported_param_idents[js_idx];
        let sym_ident = format_ident!("{}_js", p.ident);
        let ident = &p.ident;
        prelude.push(quote! { let #sym_ident = #js_ident.clone(); });
        // Convert
        let (conv_prelude, typed_call_expr) =
            conversion_from_jsvalue(ident, &p.ty, is_method, fn_name);
        // But conv_prelude uses the same ident; we need to shadow variable names accordingly.
        // We'll create a new block to avoid collisions
        prelude.push(quote! { let #ident = #sym_ident; });
        prelude.push(conv_prelude);
        typed_args.push(typed_call_expr);

        // For overloaded args, include this positional arg
        overloaded_js_args.push(quote! { &#sym_ident });
        js_idx += 1;
    }
    (quote! { #(#prelude)* }, typed_args, overloaded_js_args)
}

fn build_options_parsing(
    op_name_pascal: &str,
    params: &[ParamInfo],
    pack_start: usize,
    include_tensor_options: bool,
    _has_self: bool,
) -> (TokenStream2, Vec<TokenStream2>) {
    let options_ident = format_ident!("{}Options", op_name_pascal);
    if pack_start >= params.len() && !include_tensor_options {
        return (quote! {}, Vec::new());
    }
    let opts_ident = format_ident!("opts");
    let mut prelude = Vec::<TokenStream2>::new();
    prelude.push(quote! { let #opts_ident: #options_ident = serde_wasm_bindgen::from_value(options.clone()).unwrap_or_default(); });

    let mut typed_fields = Vec::<TokenStream2>::new();
    // For each packed param, generate a local variable with the typed value (respecting defaults)
    for (idx, p) in params.iter().enumerate() {
        if idx < pack_start {
            continue;
        }
        if include_tensor_options && idx == params.len() - 1 && is_type(&p.ty, "TensorOptions") {
            continue;
        }
        let field_ident = &p.ident;
        let default_expr = p.meta.default_expr.as_ref().map(|e| quote! { #e });
        let (kind, _meta) = classify_param_kind(&p.ty);

        if p.meta.raw_js {
            prelude.push(quote! {
                let #field_ident: wasm_bindgen::JsValue = crate::js_util::to_option(#opts_ident.#field_ident.clone()).unwrap_or(wasm_bindgen::JsValue::UNDEFINED);
            });
            typed_fields.push(quote! { #field_ident });
            continue;
        }

        match kind {
            ParamKind::Tensor => {
                if p.is_option {
                    prelude.push(quote! { let #field_ident: Option<piston::Tensor> = #opts_ident.#field_ident.map(|tensor| tensor.inner()); });
                } else {
                    prelude.push(quote! { let #field_ident: piston::Tensor = #opts_ident.#field_ident.inner(); });
                }
                typed_fields.push(quote! { #field_ident });
            }
            ParamKind::TensorOrScalar => {
                if p.is_option {
                    prelude.push(quote! {
                        let #field_ident: Option<crate::tensor::JsTensorOrScalar> = #opts_ident.#field_ident.clone();
                    });
                    typed_fields.push(quote! {
                        #field_ident
                            .map(|v| v.tensor_or_scalar())
                            .transpose()
                            .map_err(|e| e.into_js_error())?
                    });
                } else {
                    prelude.push(quote! {
                        let __js = crate::js_util::to_option(#opts_ident.#field_ident.clone()).unwrap_or(wasm_bindgen::JsValue::UNDEFINED);
                        let #field_ident: crate::tensor::JsTensorOrScalar = crate::tensor::JsTensorOrScalar { inner: __js };
                    });
                    typed_fields.push(
                        quote! { #field_ident.tensor_or_scalar().map_err(|e| e.into_js_error())? },
                    );
                }
            }
            ParamKind::Shape => {
                if p.is_option || p.meta.default_expr.is_some() {
                    prelude.push(quote! { let #field_ident: Vec<usize> = #opts_ident.#field_ident.unwrap_or_default(); });
                } else {
                    prelude
                        .push(quote! { let #field_ident: Vec<usize> = #opts_ident.#field_ident; });
                }
                typed_fields.push(quote! { #field_ident });
            }
            ParamKind::ShapeWithOneHole => {
                prelude.push(quote! {
                    let __js = crate::js_util::to_option(#opts_ident.#field_ident.clone()).unwrap_or(wasm_bindgen::JsValue::UNDEFINED);
                    let #field_ident: crate::shape::FromJsVecISize = crate::shape::FromJsVecISize::from_js_value(__js)?.ok_or(wasm_bindgen::JsError::new("Missing required dims"))?;
                });
                typed_fields.push(quote! { #field_ident });
            }
            ParamKind::Dims => {
                if let Some(def) = default_expr {
                    prelude.push(quote! {
                        let __js = #opts_ident.#field_ident.clone();
                        let #field_ident: crate::shape::FromJsVecISize = match crate::shape::FromJsVecISize::from_js_value(__js) {
                            Ok(Some(v)) => v,
                            _ => { #def }
                        };
                    });
                } else if p.is_option {
                    prelude.push(quote! {
                        let __js = #opts_ident.#field_ident.clone();
                        let #field_ident: Option<crate::shape::FromJsVecISize> = crate::shape::FromJsVecISize::from_js_value(__js)?;
                    });
                } else {
                    prelude.push(quote! {
                        let __js = #opts_ident.#field_ident.clone();
                        let #field_ident: crate::shape::FromJsVecISize = crate::shape::FromJsVecISize::from_js_value(__js)?.ok_or(wasm_bindgen::JsError::new("Missing required dims"))?;
                    });
                }
                typed_fields.push(quote! { #field_ident });
            }
            ParamKind::NormOrd => {
                prelude.push(quote! {
                    let __js = crate::js_util::to_option(#opts_ident.#field_ident.clone()).unwrap_or(wasm_bindgen::JsValue::UNDEFINED);
                    let #field_ident: Option<piston::NormOrd> = crate::tensor::js_value_to_norm_ord(__js)?;
                });
                typed_fields.push(quote! { #field_ident });
            }
            ParamKind::Dim => {
                if let Some(def) = default_expr {
                    prelude.push(quote! {
                        let __js = #opts_ident.#field_ident.clone();
                        let #field_ident = crate::shape::FromJsDim(
                            crate::js_util::to_option(__js)
                                .and_then(|v| v.as_f64())
                                .unwrap_or(#def as f64) as isize,
                        );
                    });
                } else if p.is_option {
                    prelude.push(quote! {
                        let __js = #opts_ident.#field_ident.clone();
                        let #field_ident: Option<crate::shape::FromJsDim> = crate::js_util::to_option(__js)
                            .and_then(|v| v.as_f64())
                            .map(|n| crate::shape::FromJsDim(n as isize));
                    });
                    let default = quote! { crate::shape::FromJsDim(0) };
                    typed_fields.push(quote! { #field_ident.unwrap_or(#default) });
                    continue;
                } else {
                    prelude.push(quote! {
                        let __js = #opts_ident.#field_ident.clone();
                        let #field_ident = crate::shape::FromJsDim(
                            crate::js_util::to_option(__js)
                                .and_then(|v| v.as_f64())
                                .unwrap_or(0.0) as isize,
                        );
                    });
                }
                typed_fields.push(quote! { #field_ident });
            }
            ParamKind::Bool
            | ParamKind::Usize
            | ParamKind::I32
            | ParamKind::U32
            | ParamKind::F32 => {
                let ty = &p.ty;
                if p.is_option {
                    prelude.push(quote! { let #field_ident: #ty = #opts_ident.#field_ident; });
                } else if let Some(def) = default_expr {
                    prelude.push(quote! { let #field_ident: #ty = #opts_ident.#field_ident.unwrap_or(#def); });
                } else {
                    prelude.push(quote! { let #field_ident: #ty = #opts_ident.#field_ident; });
                }
                typed_fields.push(quote! { #field_ident });
            }
            ParamKind::DType => {
                // In options we keep JsDType and convert where needed in call site; here pass through
                let ty = &p.ty;
                prelude.push(quote! { let #field_ident: #ty = #opts_ident.#field_ident; });
                typed_fields.push(quote! { #field_ident });
            }
            ParamKind::Device => {
                let ty = &p.ty;
                prelude.push(quote! { let #field_ident: #ty = #opts_ident.#field_ident; });
                typed_fields.push(quote! { #field_ident });
            }
            ParamKind::Unknown => {
                let rhs = if let Some(def) = default_expr {
                    quote! { #opts_ident.#field_ident.unwrap_or(#def) }
                } else {
                    quote! { #opts_ident.#field_ident.unwrap_or_default() }
                };
                prelude.push(quote! { let #field_ident = #rhs; });
                typed_fields.push(quote! { #field_ident });
            }
            ParamKind::VecTensor => {
                // Options struct stores Vec<JsTensor> or Option<...>? We expect Vec<Tensor> requested at call site
                prelude.push(quote! {
                    let #field_ident: Vec<piston::Tensor> = if #opts_ident.#field_ident.is_array() {
                        let array = #opts_ident.#field_ident.dyn_into::<js_sys::Array>().map_err(|_| wasm_bindgen::JsError::new("Expected an array of Tensors"))?;
                        array
                            .iter()
                            .map(|v| crate::tensor::JsTensor::try_from(v.clone()).map(|t| t.inner()))
                            .collect::<Result<Vec<_>, wasm_bindgen::JsError>>()?
                    } else {
                        return Err(wasm_bindgen::JsError::new("Expected an array of Tensors"));
                    };
                });
                typed_fields.push(quote! { #field_ident });
            }
        }
    }

    // TensorOptions sentinel
    if include_tensor_options {
        prelude.push(quote! {
            let options: piston::TensorOptions = piston::TensorOptions {
                dtype: #opts_ident.dtype.map(|d| d.dtype),
                device: #opts_ident.device.map(|d| d.inner),
                requires_grad: #opts_ident.requires_grad,
            };
        });
        typed_fields.push(quote! { options });
    }

    (quote! { #(#prelude)* }, typed_fields)
}

fn build_js_param_list(
    params: &[ParamInfo],
    pack_start: usize,
    variant: VariantKind,
    has_options: bool,
) -> Vec<Ident> {
    let mut js_param_idents = Vec::<Ident>::new();

    if matches!(variant, VariantKind::Method | VariantKind::MethodInplace) {
        // Expose the first Rust param (self) as `input` in JS for better ergonomics
        let name = format_ident!("input");
        js_param_idents.push(name);
    }

    let is_method = matches!(variant, VariantKind::Method | VariantKind::MethodInplace);
    let last_is_tensor_options = params
        .last()
        .map(|p| is_type(&p.ty, "TensorOptions"))
        .unwrap_or(false)
        && has_options;
    for (idx, p) in params.iter().enumerate() {
        if idx >= pack_start {
            break;
        }
        // For methods, skip the first Rust parameter (it becomes `self_js`)
        if is_method && idx == 0 {
            continue;
        }
        // Skip trailing TensorOptions sentinel from positional list; it will be represented by the generated Options object param
        if last_is_tensor_options && idx == params.len() - 1 {
            continue;
        }
        js_param_idents.push(p.ident.clone());
    }

    if has_options {
        let name = format_ident!("options");
        // The caller will append correct attribute for options param since it needs op_name_pascal.
        js_param_idents.push(name);
    }
    js_param_idents
}

pub fn process_js_tensor_web_op(attr: JsTensorWebOpAttr, item: ItemFn) -> SynResult<TokenStream2> {
    // Extract simple info
    let op_rust_name = item.sig.ident.clone();
    let op_name_pascal = attr.name.clone();
    // Determine target (alternative external name) if provided
    let target_name_ident = attr
        .target
        .as_ref()
        .map(|s| format_ident!("{}", s))
        .unwrap_or_else(|| op_rust_name.clone());
    let target_camel = to_lower_camel_case_with_underscore(&target_name_ident.to_string());
    let target_pascal = target_name_ident.to_string().to_upper_camel_case();
    let params = make_param_info_list(&item)?;
    let pack_start = find_pack_start(&params).unwrap_or(params.len());
    let include_tensor_options = last_param_is_tensor_options(&params);
    let has_options = pack_start < params.len() || include_tensor_options;
    let is_async = item.sig.asyncness.is_some();
    let async_token = if is_async {
        quote! { async }
    } else {
        quote! {}
    };
    let output_is_option = is_return_type_option(&item);
    let output_is_vec_tensor = is_return_type_vec_tensor(&item);
    let unchecked_ret_ts = if output_is_option {
        syn::LitStr::new("Tensor | undefined", Span::call_site())
    } else if output_is_vec_tensor {
        syn::LitStr::new("Tensor[]", Span::call_site())
    } else {
        syn::LitStr::new("Tensor", Span::call_site())
    };

    // Build Options struct
    let options_struct_tokens = if has_options {
        build_options_struct(&op_name_pascal, &params, pack_start, include_tensor_options)
    } else {
        quote! {}
    };

    let mut exported_fns = Vec::<TokenStream2>::new();
    let mut wrapper_methods = Vec::<TokenStream2>::new();
    for variant in attr.variants.iter().copied() {
        let rust_pascal = op_rust_name.to_string().to_upper_camel_case();
        let export_js_name = match variant {
            VariantKind::Function => variant.as_js_export_name(&target_pascal),
            VariantKind::Method => format!("opMethod{rust_pascal}"),
            VariantKind::MethodInplace => format!("opMethod{rust_pascal}_"),
        };
        let is_method = matches!(variant, VariantKind::Method | VariantKind::MethodInplace);
        let is_inplace = matches!(variant, VariantKind::MethodInplace);

        // JS parameter list and attributes (skip the first param for method variants, becomes self)
        let js_param_idents = build_js_param_list(&params, pack_start, variant, has_options);

        // Early function-mode dispatch setup
        // Build overloaded args slice: positional js args after first (for method) or all (for function), excluding options
        let mut overloaded_js_args = Vec::<TokenStream2>::new();
        if !params.is_empty() {
            // For methods, skip self_js; for function, skip the first param if it's an input tensor
            let first_is_tensor = params
                .first()
                .map(|p| is_type(&p.ty, "Tensor") || is_optional_of_type(&p.ty, "Tensor"))
                .unwrap_or(false);
            let start_index = if is_method || first_is_tensor { 1 } else { 0 };
            let end_index = js_param_idents.len() - if has_options { 1 } else { 0 };
            js_param_idents
                .iter()
                .skip(start_index)
                .take(end_index)
                .for_each(|id| {
                    overloaded_js_args.push(quote! { &#id });
                });
        }
        // Build local arrays for overloaded args and positional args to avoid borrowing temporaries
        let mut dispatch_prelude = Vec::<TokenStream2>::new();
        let overloaded_slice_ident = format_ident!("__overloaded_slice");
        if overloaded_js_args.is_empty() {
            dispatch_prelude
                .push(quote! { let #overloaded_slice_ident: [&wasm_bindgen::JsValue; 0] = []; });
        } else {
            let len = overloaded_js_args.len();
            dispatch_prelude.push(quote! { let #overloaded_slice_ident: [&wasm_bindgen::JsValue; #len] = [ #(#overloaded_js_args),* ]; });
        }

        // When TensorOptions sentinel is present, exclude it from positional args
        let pack_start_positional = if include_tensor_options {
            pack_start.min(params.len().saturating_sub(1))
        } else {
            pack_start
        };

        let pre_pack_count = pack_start_positional.min(params.len());
        let args_items: Vec<TokenStream2> = if is_method {
            (1..(pre_pack_count))
                .map(|i| {
                    let id = &js_param_idents[i];
                    quote! { &#id }
                })
                .collect()
        } else {
            (0..pre_pack_count)
                .map(|i| {
                    let id = &js_param_idents[i];
                    quote! { &#id }
                })
                .collect()
        };
        let args_array_ident = format_ident!("__positional_args");
        if args_items.is_empty() {
            dispatch_prelude
                .push(quote! { let #args_array_ident: [&wasm_bindgen::JsValue; 0] = []; });
        } else {
            let k = args_items.len();
            dispatch_prelude.push(quote! { let #args_array_ident: [&wasm_bindgen::JsValue; #k] = [ #(#args_items),* ]; });
        }

        // Function-mode call
        // For handle_piston_function `function_name`:
        // - Methods: use the exported method name (e.g., opMethodRequiresGrad or opMethodRequiresGrad_)
        // - Functions: use the exported function name (lowerCamelCase op name)
        let function_name_str = &export_js_name;
        let named_arg_tokens: TokenStream2 = if has_options {
            quote! { options }
        } else {
            quote! { &wasm_bindgen::JsValue::UNDEFINED }
        };

        let self_arg = if is_method {
            quote! { Some(&input) }
        } else {
            quote! { None }
        };

        let handle_call = quote! {
            #(#dispatch_prelude)*
            let overloaded_args = crate::function::get_overloaded_args(&#overloaded_slice_ident);
            if !overloaded_args.is_empty() || crate::function::is_function_mode_active() {
                return crate::function::handle_piston_function(
                    #self_arg,
                    #function_name_str,
                    &overloaded_args,
                    &#args_array_ident,
                    #named_arg_tokens,
                );
            }
        };

        // Build conversion prelude and call
        let (pos_prelude, typed_positional_args, _ol_js) = build_positional_param_defs(
            &params,
            pack_start_positional,
            &js_param_idents,
            is_method,
            function_name_str,
        );
        let (opts_prelude, typed_options_args) = build_options_parsing(
            &op_name_pascal,
            &params,
            pack_start,
            include_tensor_options,
            is_method,
        );

        let mut typed_args_for_call = Vec::<TokenStream2>::new();
        if is_method {
            typed_args_for_call.extend(typed_positional_args.clone());
        } else if typed_positional_args.len() > 1 {
            typed_args_for_call.extend(typed_positional_args.into_iter().skip(1));
        }
        typed_args_for_call.extend(typed_options_args.clone());

        let invocation_tokens = if item.block.stmts.is_empty() {
            // Direct call to core op (method on Tensor)
            let base_ident = target_name_ident.clone();
            let method_ident = if is_inplace {
                format_ident!("{}_", base_ident)
            } else {
                base_ident
            };

            let first_ident = if is_method {
                syn::Ident::new("input", Span::call_site())
            } else {
                params
                    .first()
                    .map(|p| p.ident.clone())
                    .unwrap_or_else(|| format_ident!("self"))
            };

            quote! {
                #first_ident.#method_ident( #(#typed_args_for_call),* )
            }
        } else {
            // Custom body: inline the user's function body with typed params
            let user_block = &item.block;
            quote! {
                #user_block
            }
        };

        let call_result_binding = if output_is_option {
            quote! {
                let result_opt = #invocation_tokens;
                Ok(result_opt
                    .map(|js_t| js_t.into())
                    .unwrap_or(wasm_bindgen::JsValue::UNDEFINED))
            }
        } else if output_is_vec_tensor {
            quote! {
                let result = #invocation_tokens.map_err(|e| e.into_js_error())?;
                let array = js_sys::Array::new();
                for t in result.into_iter() {
                    array.push(&crate::tensor::JsTensor::new(t).into());
                }
                Ok(array.into())
            }
        } else {
            quote! {
                let result = #invocation_tokens.map_err(|e| e.into_js_error())?;
                Ok(crate::tensor::JsTensor::new(result).into())
            }
        };

        let call_tokens = if is_method {
            quote! {
                let input = crate::tensor::JsTensor::try_from(input.clone())?._clone_weak_js().inner();
                #call_result_binding
            }
        } else {
            call_result_binding
        };

        // Build full exported function
        let export_ident = format_ident!("{}", export_js_name);
        // Compute js_name to expose. Allow override for methods; append '_' for inplace when overridden
        let js_name_effective = if is_method {
            // For method exports, keep opMethod<Name> unchanged regardless of target
            export_js_name.clone()
        } else {
            // For function exports, if target provided, use it exactly; else use the default
            if let Some(ref tgt) = attr.target {
                tgt.clone()
            } else {
                export_js_name.clone()
            }
        };
        let js_name_lit = syn::LitStr::new(&js_name_effective, Span::call_site());
        let mut param_decls = Vec::<TokenStream2>::new();
        let last_is_tensor_options = params
            .last()
            .map(|p| is_type(&p.ty, "TensorOptions"))
            .unwrap_or(false)
            && has_options;
        for (i, ident) in js_param_idents.iter().enumerate() {
            // Attach attributes
            if matches!(variant, VariantKind::Method | VariantKind::MethodInplace) && i == 0 {
                param_decls.push(quote! { #[wasm_bindgen(unchecked_param_type = "Tensor")] #ident: &wasm_bindgen::JsValue });
            } else if has_options && i == js_param_idents.len() - 1 {
                // We need the type name string, not ident token
                let ts = syn::LitStr::new(&format!("{op_name_pascal}Options?"), Span::call_site());
                param_decls.push(quote! { #[wasm_bindgen(unchecked_param_type = #ts)] #ident: &wasm_bindgen::JsValue });
            } else {
                let mut param_idx = i;
                if last_is_tensor_options && param_idx >= params.len() - 1 {
                    param_idx = params.len() - 2;
                }
                let p = &params[param_idx];
                let name_str = p.ident.to_string();
                let camel = if let Some(ref custom) = p.meta.name {
                    custom.clone()
                } else {
                    name_str.to_lower_camel_case()
                };
                let mut attrs_tokens: Vec<TokenStream2> = Vec::new();
                // js_name override when camelCase differs
                if camel != name_str {
                    let jsname_lit = syn::LitStr::new(&camel, Span::call_site());
                    attrs_tokens.push(quote! { #[wasm_bindgen(js_name = #jsname_lit)] });
                }
                // unchecked type override or derived
                if let Some(ref ts_override) = p.meta.ts_type_override {
                    let lit = syn::LitStr::new(ts_override, Span::call_site());
                    attrs_tokens.push(quote! { #[wasm_bindgen(unchecked_param_type = #lit)] });
                } else {
                    let ts = param_unchecked_ts_type(&p.ty, p.is_option);
                    let ts_lit = syn::LitStr::new(&ts, Span::call_site());
                    attrs_tokens.push(quote! { #[wasm_bindgen(unchecked_param_type = #ts_lit)] });
                }
                param_decls.push(quote! { #(#attrs_tokens)* #ident: &wasm_bindgen::JsValue });
            }
        }

        let wasm_fn = {
            quote! {
                #[wasm_bindgen(js_name = #js_name_lit, unchecked_return_type = #unchecked_ret_ts)]
                pub #async_token fn #export_ident( #(#param_decls),* ) -> Result<wasm_bindgen::JsValue, wasm_bindgen::JsError> {
                    #handle_call
                    #pos_prelude
                    #opts_prelude
                    #call_tokens
                }
            }
        };

        exported_fns.push(wasm_fn);

        // Generate instance method passthrough on JsTensor for method variants
        if is_method {
            // Build wrapper param list: skip self/input, keep other pre-pack params and options if present
            let mut wrapper_param_decls = Vec::<TokenStream2>::new();
            // Determine positional count like earlier
            let pack_start_positional = if include_tensor_options {
                pack_start.min(params.len().saturating_sub(1))
            } else {
                pack_start
            };
            let pre_pack_count = pack_start_positional.min(params.len());
            params.iter().take(pre_pack_count).skip(1).for_each(|p| {
                let name = p.ident.clone();
                // camelCase rename for param if different
                let name_str = name.to_string();
                let camel = if let Some(ref custom) = p.meta.name {
                    custom.clone()
                } else {
                    name_str.to_lower_camel_case()
                };
                let mut attrs = Vec::<TokenStream2>::new();
                if camel != name_str {
                    let jsname_lit = syn::LitStr::new(&camel, Span::call_site());
                    attrs.push(quote! { #[wasm_bindgen(js_name = #jsname_lit)] });
                }
                // type attr
                let ts_lit = if let Some(ref ts_override) = p.meta.ts_type_override {
                    syn::LitStr::new(ts_override, Span::call_site())
                } else {
                    let ts = param_unchecked_ts_type(&p.ty, p.is_option);
                    syn::LitStr::new(&ts, Span::call_site())
                };
                attrs.push(quote! { #[wasm_bindgen(unchecked_param_type = #ts_lit)] });
                wrapper_param_decls.push(quote! { #(#attrs)* #name: &wasm_bindgen::JsValue });
            });
            if has_options {
                let ts = syn::LitStr::new(&format!("{op_name_pascal}Options?"), Span::call_site());
                wrapper_param_decls.push(quote! { #[wasm_bindgen(unchecked_param_type = #ts)] options: &wasm_bindgen::JsValue });
            }

            // Compute JS method name
            // If a target is provided, use it EXACTLY (preserve case) for JS name; otherwise use lowerCamelCase of the Rust name.
            let base_js_method_name = if let Some(ref tgt) = attr.target {
                tgt.clone()
            } else if let Some(ref js_name) = attr.js_name {
                js_name.clone()
            } else {
                target_camel.clone()
            };
            let mut js_method_name = base_js_method_name.clone();
            if is_inplace && !js_method_name.ends_with('_') {
                js_method_name.push('_');
            }
            let js_method_name_lit = syn::LitStr::new(&js_method_name, Span::call_site());

            // Choose attribute key: js_name (normal) or getter/setter when set
            let method_name_key = syn::Ident::new(
                if attr.getter {
                    "getter"
                } else if attr.setter {
                    "setter"
                } else {
                    "js_name"
                },
                Span::call_site(),
            );
            let method_name_attr = quote! { #[wasm_bindgen(#method_name_key = #js_method_name_lit, unchecked_return_type = #unchecked_ret_ts)] };

            // Build call to exported free function
            let free_fn_ident = format_ident!("{}", export_js_name);
            let mut call_args: Vec<TokenStream2> = Vec::new();
            call_args.push(quote! { &input_js });
            params.iter().take(pre_pack_count).skip(1).for_each(|p| {
                let id = &p.ident;
                call_args.push(quote! { #id });
            });
            if has_options {
                call_args.push(quote! { options });
            }

            let maybe_async_await = if is_async {
                quote! { .await }
            } else {
                quote! {}
            };
            let call_tokens = quote! { #free_fn_ident( #(#call_args),* ) #maybe_async_await };

            // Name the Rust method after the original Rust function ident to avoid collisions (e.g., `T_upper`),
            // and append '_' for inplace variants.
            let wrapper_fn_ident = if is_inplace {
                format_ident!("{}_", op_rust_name)
            } else {
                op_rust_name.clone()
            };
            wrapper_methods.push(quote! {
                #[wasm_bindgen(js_class = Tensor)]
                impl JsTensor {
                    #method_name_attr
                    #[allow(non_snake_case)]
                    pub #async_token fn #wrapper_fn_ident(&self, #(#wrapper_param_decls),*) -> Result<wasm_bindgen::JsValue, wasm_bindgen::JsError> {
                        let input_js = self.js_value();
                        #call_tokens
                    }
                }
            });
        }
    }

    // Do not re-emit the original function. Its signature and optional custom body are
    // incorporated into the generated exports. Re-emitting would leak internal attrs (e.g. #[op]).
    let tokens = quote! {
        #options_struct_tokens
        #(#exported_fns)*
        #(#wrapper_methods)*
    };
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use quote::quote;

    #[test]
    fn parses_attr_with_variants() {
        let attr: JsTensorWebOpAttr = syn::parse_quote!(
            name = "Addcdiv",
            variants = [function, method, method_inplace]
        );
        assert_eq!(attr.name, "Addcdiv");
        assert_eq!(attr.variants.len(), 3);
    }

    #[test]
    fn generates_options_struct_and_exports() {
        let item: ItemFn = syn::parse2(quote! {
            pub fn addcdiv(input: Tensor, tensor1: Tensor, tensor2: Tensor, #[op(default = 1.0)] value: f32) -> Result<Tensor> {}
        }).unwrap();
        let attr: JsTensorWebOpAttr = syn::parse_quote!(
            name = "Addcdiv",
            variants = [function, method, method_inplace]
        );
        let out = process_js_tensor_web_op(attr, item).unwrap();
        let s = out.to_string();
        assert!(s.contains("struct AddcdivOptions"));
        assert!(s.contains("_opFnAddcdiv"));
        assert!(s.contains("opMethodAddcdiv"));
        assert!(s.contains("opMethodAddcdiv_"));
        assert!(s.contains("unchecked_return_type = \"Tensor\""));
    }

    #[test]
    fn packs_tensor_options_fields() {
        let item: ItemFn = syn::parse2(quote! {
            pub fn arange(#[op(keyword)] end: f32, options: TensorOptions) -> Result<Tensor> {}
        })
        .unwrap();
        let attr: JsTensorWebOpAttr = syn::parse_quote!(name = "Arange", variants = [function]);
        let out = process_js_tensor_web_op(attr, item).unwrap();
        let s = out.to_string();
        assert!(s.contains("struct ArangeOptions"));
        assert!(s.contains("requiresGrad"));
        assert!(s.contains("JsDType"));
        assert!(s.contains("JsDevice"));
    }
}
