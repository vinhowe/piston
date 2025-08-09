use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::quote;
use std::collections::HashMap;
use syn::{
    Attribute, Error, Expr, FnArg, GenericParam, Ident, ItemFn, Pat, PatIdent, PatType, Path,
    Result, ReturnType, Token, Type,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
};

/// Represents the variants that can be generated from a tensor operation
#[derive(Debug, Clone, PartialEq)]
pub enum TensorOpVariant {
    Function,
    Method,
    MethodInplace,
}

impl Parse for TensorOpVariant {
    fn parse(input: ParseStream) -> Result<Self> {
        let ident: Ident = input.parse()?;
        match ident.to_string().as_str() {
            "function" => Ok(TensorOpVariant::Function),
            "method" => Ok(TensorOpVariant::Method),
            "method_inplace" => Ok(TensorOpVariant::MethodInplace),
            _ => Err(Error::new_spanned(
                ident,
                "Expected 'function', 'method', or 'method_inplace'",
            )),
        }
    }
}

/// Represents a default parameter assignment
#[derive(Clone)]
pub struct DefaultParam {
    pub name: String,
    pub value: Expr,
}

impl std::fmt::Debug for DefaultParam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DefaultParam")
            .field("name", &self.name)
            .field("value", &"<expr>")
            .finish()
    }
}

/// Represents the parsed contents of a #[tensor_op(...)] attribute
#[derive(Debug, Clone)]
pub struct TensorOpAttr {
    pub variants: Vec<TensorOpVariant>,
}

impl Parse for TensorOpAttr {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut variants = Vec::new();

        if input.is_empty() {
            // Default variants if none specified
            return Ok(TensorOpAttr {
                variants: vec![TensorOpVariant::Function, TensorOpVariant::Method],
            });
        }

        while !input.is_empty() {
            let ident: Ident = input.parse()?;
            match ident.to_string().as_str() {
                "variants" => {
                    let _eq: Token![=] = input.parse()?;
                    let content;
                    syn::bracketed!(content in input);
                    let variant_list: Punctuated<TensorOpVariant, Token![,]> =
                        content.parse_terminated(TensorOpVariant::parse, Token![,])?;
                    variants.extend(variant_list);
                }
                _ => return Err(Error::new_spanned(ident, "Unknown tensor_op attribute")),
            }

            if !input.is_empty() {
                let _comma: Token![,] = input.parse()?;
            }
        }

        if variants.is_empty() {
            variants = vec![TensorOpVariant::Function, TensorOpVariant::Method];
        }

        Ok(TensorOpAttr { variants })
    }
}

/// Represents a function parameter with its default value if any
#[derive(Clone)]
pub struct ParamInfo {
    pub name: String,
    pub pat_type: PatType,
    pub default: Option<Expr>,
    pub is_op_tensor: bool,
    pub mentions_op_tensor: bool,
}

impl std::fmt::Debug for ParamInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParamInfo")
            .field("name", &self.name)
            .field("pat_type", &"<PatType>")
            .field("default", &self.default.as_ref().map(|_| "<expr>"))
            .field("is_op_tensor", &self.is_op_tensor)
            .field("mentions_op_tensor", &self.mentions_op_tensor)
            .finish()
    }
}

/// Generic allocator for OT1, OT2, etc.
struct OtGen {
    counter: usize,
}

impl OtGen {
    fn new() -> Self {
        Self { counter: 1 }
    }

    fn next(&mut self) -> Ident {
        let ident = Ident::new(&format!("OT{}", self.counter), Span::call_site());
        self.counter += 1;
        ident
    }
}

/// Check if a path ends with "OpTensor"
fn is_op_tensor_path(path: &Path) -> bool {
    path.segments
        .last()
        .map(|seg| seg.ident == "OpTensor")
        .unwrap_or(false)
}

/// Check if a type mentions OpTensor anywhere
fn mentions_op_tensor(ty: &Type) -> bool {
    match ty {
        Type::Path(type_path) => {
            if is_op_tensor_path(&type_path.path) {
                return true;
            }
            // Check in generic arguments
            for segment in &type_path.path.segments {
                if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                    for arg in &args.args {
                        if let syn::GenericArgument::Type(inner_ty) = arg
                            && mentions_op_tensor(inner_ty)
                        {
                            return true;
                        }
                    }
                }
            }
            false
        }
        Type::Reference(type_ref) => mentions_op_tensor(&type_ref.elem),
        Type::Tuple(type_tuple) => type_tuple.elems.iter().any(mentions_op_tensor),
        _ => false,
    }
}

/// Replace OpTensor with a new type in a Type
fn replace_op_tensor_in_type(ty: &mut Type, replacement: &Path) {
    match ty {
        Type::Path(type_path) => {
            if is_op_tensor_path(&type_path.path) {
                type_path.path = replacement.clone();
            } else {
                // Check in generic arguments
                for segment in &mut type_path.path.segments {
                    if let syn::PathArguments::AngleBracketed(args) = &mut segment.arguments {
                        for arg in &mut args.args {
                            if let syn::GenericArgument::Type(inner_ty) = arg {
                                replace_op_tensor_in_type(inner_ty, replacement);
                            }
                        }
                    }
                }
            }
        }
        Type::Reference(type_ref) => {
            replace_op_tensor_in_type(&mut type_ref.elem, replacement);
        }
        Type::Tuple(type_tuple) => {
            for elem in &mut type_tuple.elems {
                replace_op_tensor_in_type(elem, replacement);
            }
        }
        _ => {}
    }
}

/// Replace OpTensor with Tensor or Self in a Type
fn replace_op_tensor_with_tensor(ty: &mut Type, use_self: bool) {
    let replacement = if use_self {
        syn::parse_quote!(Self)
    } else {
        syn::parse_quote!(Tensor)
    };
    replace_op_tensor_in_type(ty, &replacement);
}

/// Replace OpTensor in a Type using the constraint_ot_map to find the right replacement
/// The current_generic parameter tells us which generic constraint we're currently processing
fn replace_op_tensor_in_type_for_constraints(
    ty: &mut Type,
    constraint_ot_map: &HashMap<String, Ident>,
    current_generic: &str,
) {
    match ty {
        Type::Path(type_path) => {
            if is_op_tensor_path(&type_path.path) {
                // Replace OpTensor with the OT generic assigned to this constraint
                if let Some(ot_ident) = constraint_ot_map.get(current_generic) {
                    type_path.path = syn::parse_quote!(#ot_ident);
                }
            } else {
                replace_op_tensor_in_path_for_constraints(
                    &mut type_path.path,
                    constraint_ot_map,
                    current_generic,
                );
            }
        }
        Type::Reference(type_ref) => {
            replace_op_tensor_in_type_for_constraints(
                &mut type_ref.elem,
                constraint_ot_map,
                current_generic,
            );
        }
        Type::Tuple(type_tuple) => {
            for elem in &mut type_tuple.elems {
                replace_op_tensor_in_type_for_constraints(elem, constraint_ot_map, current_generic);
            }
        }
        _ => {}
    }
}

/// Replace OpTensor in a Path using the constraint_ot_map
fn replace_op_tensor_in_path_for_constraints(
    path: &mut Path,
    constraint_ot_map: &HashMap<String, Ident>,
    current_generic: &str,
) {
    for segment in &mut path.segments {
        if let syn::PathArguments::AngleBracketed(args) = &mut segment.arguments {
            for arg in &mut args.args {
                if let syn::GenericArgument::Type(ty) = arg {
                    replace_op_tensor_in_type_for_constraints(
                        ty,
                        constraint_ot_map,
                        current_generic,
                    );
                }
            }
        }
    }
}

/// Parse default attribute from parameter attributes
fn parse_default_attr(attrs: &[Attribute]) -> Result<Option<Expr>> {
    for attr in attrs {
        if attr.path().is_ident("default") {
            return attr.parse_args::<Expr>().map(Some);
        }
    }
    Ok(None)
}

/// Parse function parameters and extract information
fn parse_parameters(inputs: &Punctuated<FnArg, Token![,]>) -> Result<Vec<ParamInfo>> {
    let mut params = Vec::new();
    let mut seen_default = false;

    for input in inputs {
        if let FnArg::Typed(pat_type) = input
            && let Pat::Ident(PatIdent { ident, .. }) = &*pat_type.pat
        {
            let default = parse_default_attr(&pat_type.attrs)?;

            // Validate default ordering
            if seen_default && default.is_none() {
                return Err(Error::new_spanned(
                    pat_type,
                    "Parameter without default follows parameter with default",
                ));
            }
            if default.is_some() {
                seen_default = true;
            }

            let is_op_tensor = if let Type::Path(type_path) = &*pat_type.ty {
                is_op_tensor_path(&type_path.path)
            } else {
                false
            };

            let mentions_op_tensor = mentions_op_tensor(&pat_type.ty);

            params.push(ParamInfo {
                name: ident.to_string(),
                pat_type: pat_type.clone(),
                default,
                is_op_tensor,
                mentions_op_tensor,
            });
        }
    }

    Ok(params)
}

/// Enhanced parameter analysis that considers generic constraints
fn analyze_parameters_with_generics(
    inputs: &Punctuated<FnArg, Token![,]>,
    generics: &syn::Generics,
) -> Result<Vec<ParamInfo>> {
    let mut params = Vec::new();
    let mut seen_default = false;

    for input in inputs {
        if let FnArg::Typed(pat_type) = input
            && let Pat::Ident(PatIdent { ident, .. }) = &*pat_type.pat
        {
            let default = parse_default_attr(&pat_type.attrs)?;

            // Validate default ordering
            if seen_default && default.is_none() {
                return Err(Error::new_spanned(
                    pat_type,
                    "Parameter without default follows parameter with default",
                ));
            }
            if default.is_some() {
                seen_default = true;
            }

            let is_op_tensor = if let Type::Path(type_path) = &*pat_type.ty {
                is_op_tensor_path(&type_path.path)
            } else {
                false
            } || is_option_with_op_tensor(&pat_type.ty)
                || is_container_type_with_op_tensor(&pat_type.ty);

            let mut mentions_op_tensor = mentions_op_tensor(&pat_type.ty);

            // Check if this parameter's type is a generic that has OpTensor constraints
            if !mentions_op_tensor
                && let Type::Path(type_path) = &*pat_type.ty
                && type_path.path.segments.len() == 1
            {
                let param_type_name = &type_path.path.segments[0].ident;
                // Check if this generic has OpTensor in its constraints
                for param in &generics.params {
                    if let GenericParam::Type(type_param) = param
                        && type_param.ident == *param_type_name
                    {
                        let has_op_tensor_constraint = type_param.bounds.iter().any(|bound| {
                            if let syn::TypeParamBound::Trait(trait_bound) = bound {
                                contains_op_tensor_in_path(&trait_bound.path)
                            } else {
                                false
                            }
                        });
                        if has_op_tensor_constraint {
                            mentions_op_tensor = true;
                            break;
                        }
                    }
                }
            }

            params.push(ParamInfo {
                name: ident.to_string(),
                pat_type: pat_type.clone(),
                default,
                is_op_tensor,
                mentions_op_tensor,
            });
        }
    }

    Ok(params)
}

/// Build a mapping from OpTensor occurrences in constraints to unique OT generics
fn build_constraint_ot_map(
    generics: &syn::Generics,
    op_tensor_map: &HashMap<String, Ident>,
) -> HashMap<String, Ident> {
    let mut constraint_ot_map = HashMap::new();
    let mut ot_gen = OtGen::new();

    // Skip existing OT generics to avoid conflicts
    for _ in 0..op_tensor_map.len() {
        ot_gen.next();
    }

    // For each generic parameter that has OpTensor in its constraints,
    // assign it a unique OT generic
    for param in &generics.params {
        if let GenericParam::Type(type_param) = param {
            let generic_name = type_param.ident.to_string();

            // Check if this generic has OpTensor in its bounds
            let has_op_tensor_constraint = type_param.bounds.iter().any(|bound| {
                if let syn::TypeParamBound::Trait(trait_bound) = bound {
                    contains_op_tensor_in_path(&trait_bound.path)
                } else {
                    false
                }
            });

            if has_op_tensor_constraint {
                let new_ot = ot_gen.next();
                constraint_ot_map.insert(generic_name, new_ot);
            }
        }
    }

    constraint_ot_map
}

/// Check if a path contains OpTensor anywhere in its arguments
fn contains_op_tensor_in_path(path: &Path) -> bool {
    for segment in &path.segments {
        if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
            for arg in &args.args {
                if let syn::GenericArgument::Type(ty) = arg
                    && mentions_op_tensor(ty)
                {
                    return true;
                }
            }
        }
    }
    false
}

/// Check if a type is TensorTypeOrScalar<...>
fn is_tensor_type_or_scalar(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty {
        type_path
            .path
            .segments
            .last()
            .map(|seg| seg.ident == "TensorTypeOrScalar")
            .unwrap_or(false)
    } else {
        false
    }
}

/// Check if a type is Option<T> where T contains OpTensor
fn is_option_with_op_tensor(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty
        && let Some(last_segment) = type_path.path.segments.last()
        && last_segment.ident == "Option"
        && let syn::PathArguments::AngleBracketed(args) = &last_segment.arguments
    {
        return args.args.iter().any(|arg| {
            if let syn::GenericArgument::Type(inner_ty) = arg {
                mentions_op_tensor(inner_ty)
            } else {
                false
            }
        });
    }
    false
}

/// Check if a type is Option<Tensor> or Option<Self>
fn is_option_with_tensor_or_self(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty
        && let Some(last_segment) = type_path.path.segments.last()
        && last_segment.ident == "Option"
        && let syn::PathArguments::AngleBracketed(args) = &last_segment.arguments
    {
        return args.args.iter().any(|arg| {
            if let syn::GenericArgument::Type(Type::Path(inner_path)) = arg {
                inner_path.path.segments.len() == 1
                    && (inner_path.path.segments[0].ident == "Tensor"
                        || inner_path.path.segments[0].ident == "Self")
            } else {
                false
            }
        });
    }
    false
}

/// Check if a type is a container type that needs iter().map().collect() conversion
/// These types need special handling for OpTensor conversion
fn is_container_type_with_op_tensor(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty
        && let Some(last_segment) = type_path.path.segments.last()
    {
        // Check for container types like RVec, Vec, etc. that contain OpTensor
        let container_names = ["RVec", "Vec", "Array", "SmallVec"];
        if container_names.contains(&last_segment.ident.to_string().as_str())
            && let syn::PathArguments::AngleBracketed(args) = &last_segment.arguments
        {
            return args.args.iter().any(|arg| {
                if let syn::GenericArgument::Type(inner_ty) = arg {
                    mentions_op_tensor(inner_ty)
                } else {
                    false
                }
            });
        }
    }
    false
}

/// Generate the kernel function with OT generics
fn generate_kernel_function(
    original_fn: &ItemFn,
    params: &[ParamInfo],
    op_tensor_map: &HashMap<String, Ident>,
) -> Result<ItemFn> {
    let mut kernel_fn = original_fn.clone();

    // Rename to {name}_kernel
    let kernel_name = Ident::new(
        &format!("{}_kernel", original_fn.sig.ident),
        original_fn.sig.ident.span(),
    );
    kernel_fn.sig.ident = kernel_name;

    // Make it pub(crate)
    kernel_fn.vis = syn::parse_quote!(pub(crate));

    // Build constraint OT mapping for unique OpTensor replacements in constraints
    let constraint_ot_map = build_constraint_ot_map(&original_fn.sig.generics, op_tensor_map);

    // Add OT generics and replace OpTensor in original generics
    let mut new_generics = original_fn.sig.generics.clone();

    // First, replace OpTensor in existing generic bounds
    for param in &mut new_generics.params {
        if let GenericParam::Type(type_param) = param {
            let generic_name = type_param.ident.to_string();
            for bound in &mut type_param.bounds {
                if let syn::TypeParamBound::Trait(trait_bound) = bound {
                    replace_op_tensor_in_path_for_constraints(
                        &mut trait_bound.path,
                        &constraint_ot_map,
                        &generic_name,
                    );
                }
            }
        }
    }

    // Replace OpTensor in where clauses
    if let Some(where_clause) = &mut new_generics.where_clause {
        for predicate in &mut where_clause.predicates {
            if let syn::WherePredicate::Type(type_predicate) = predicate {
                // Extract the generic name from the bounded type
                let generic_name = if let Type::Path(type_path) = &type_predicate.bounded_ty {
                    type_path
                        .path
                        .segments
                        .first()
                        .map(|seg| seg.ident.to_string())
                        .unwrap_or_default()
                } else {
                    String::new()
                };

                for bound in &mut type_predicate.bounds {
                    if let syn::TypeParamBound::Trait(trait_bound) = bound {
                        replace_op_tensor_in_path_for_constraints(
                            &mut trait_bound.path,
                            &constraint_ot_map,
                            &generic_name,
                        );
                    }
                }
            }
        }
    }

    // Add OT generics (both from parameters and constraints)
    for ot_ident in op_tensor_map.values() {
        let generic_param: GenericParam = syn::parse_quote!(#ot_ident: Into<OpTensor>);
        new_generics.params.push(generic_param);
    }

    // Add constraint OT generics
    for ot_ident in constraint_ot_map.values() {
        let generic_param: GenericParam = syn::parse_quote!(#ot_ident: Into<OpTensor>);
        new_generics.params.push(generic_param);
    }

    // Replace OpTensor in parameters with OT generics
    let mut new_inputs = Punctuated::new();
    for input in &original_fn.sig.inputs {
        if let FnArg::Typed(mut pat_type) = input.clone() {
            // Remove default attributes from kernel function
            pat_type
                .attrs
                .retain(|attr| !attr.path().is_ident("default"));

            if let Pat::Ident(PatIdent { ident, .. }) = &*pat_type.pat
                && let Some(ot_ident) = op_tensor_map.get(&ident.to_string())
            {
                let ot_path: Path = syn::parse_quote!(#ot_ident);
                replace_op_tensor_in_type(&mut pat_type.ty, &ot_path);
            }
            new_inputs.push(FnArg::Typed(pat_type));
        } else {
            new_inputs.push(input.clone());
        }
    }

    kernel_fn.sig.inputs = new_inputs;
    kernel_fn.sig.generics = new_generics;

    // Generate parameter conversions at the beginning of the function
    let mut conversions = Vec::new();

    for param in params {
        let param_name = Ident::new(&param.name, Span::call_site());

        if param.is_op_tensor || param.mentions_op_tensor {
            // Check the actual type to determine the right conversion
            if is_container_type_with_op_tensor(&param.pat_type.ty) {
                // Container types like RVec<OpTensor>, Vec<OpTensor> need iter().map().collect() conversion
                if let Type::Path(type_path) = &*param.pat_type.ty
                    && let Some(last_segment) = type_path.path.segments.last()
                {
                    let container_name = &last_segment.ident;
                    conversions.push(quote! {
                            let #param_name = #param_name.into_iter().map(|inner| inner.into()).collect::<#container_name<OpTensor>>();
                        });
                }
            } else if is_option_with_op_tensor(&param.pat_type.ty) {
                conversions.push(quote! {
                    let #param_name = #param_name.map(|inner| inner.into());
                });
            } else if is_tensor_type_or_scalar(&param.pat_type.ty) {
                conversions.push(quote! {
                    let #param_name = #param_name.map_tensor(|inner| inner.into())?;
                });
            } else if param.is_op_tensor {
                // Direct OpTensor parameter: let param = param.into();
                conversions.push(quote! {
                    let #param_name = #param_name.into();
                });
            } else {
                // This might be a generic type constrained by TensorTypeOrScalar<OpTensor>
                conversions.push(quote! {
                    let #param_name = #param_name.map_tensor(|inner| inner.into())?;
                });
            }
        }
    }

    // Preserve the original function body but add conversions at the beginning
    let original_block = &original_fn.block;
    kernel_fn.block = syn::parse_quote!({
        #(#conversions)*
        #original_block
    });

    Ok(kernel_fn)
}

/// Generate function variant
fn generate_function_variant(
    original_fn: &ItemFn,
    params: &[ParamInfo],
    kernel_name: &Ident,
) -> Result<ItemFn> {
    let mut func = original_fn.clone();

    // Replace OpTensor with Tensor in signature
    let mut new_inputs = Punctuated::new();
    for input in &original_fn.sig.inputs {
        if let FnArg::Typed(mut pat_type) = input.clone() {
            // Remove default attributes
            pat_type
                .attrs
                .retain(|attr| !attr.path().is_ident("default"));
            replace_op_tensor_with_tensor(&mut pat_type.ty, false);
            new_inputs.push(FnArg::Typed(pat_type));
        } else {
            new_inputs.push(input.clone());
        }
    }
    func.sig.inputs = new_inputs;

    // Replace OpTensor with Tensor in return type
    if let ReturnType::Type(_, ref mut ty) = func.sig.output {
        replace_op_tensor_with_tensor(ty, false);
    }

    // Replace OpTensor with Tensor in generics
    let mut new_generics = original_fn.sig.generics.clone();
    for param in &mut new_generics.params {
        if let GenericParam::Type(type_param) = param {
            for bound in &mut type_param.bounds {
                if let syn::TypeParamBound::Trait(trait_bound) = bound {
                    for segment in &mut trait_bound.path.segments {
                        if let syn::PathArguments::AngleBracketed(args) = &mut segment.arguments {
                            for arg in &mut args.args {
                                if let syn::GenericArgument::Type(ty) = arg {
                                    replace_op_tensor_with_tensor(ty, false);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    func.sig.generics = new_generics;

    // Generate call arguments
    let mut call_args = Vec::new();
    for param in params {
        let param_name = Ident::new(&param.name, Span::call_site());
        // if param.is_op_tensor || param.mentions_op_tensor {
        //     call_args.push(quote!(#param_name));
        // } else {
        //     call_args.push(quote!(#param_name));
        // }
        call_args.push(quote!(#param_name));
    }

    func.block = syn::parse_quote!({
        #kernel_name(#(#call_args),*).map(Tensor::wrap)
    });

    Ok(func)
}

/// Generate method variant
fn generate_method_variant(
    original_fn: &ItemFn,
    params: &[ParamInfo],
    kernel_name: &Ident,
) -> Result<TokenStream2> {
    let method_name = &original_fn.sig.ident;
    let mut generics = original_fn.sig.generics.clone();

    // Replace OpTensor with Self in generics
    for param in &mut generics.params {
        if let GenericParam::Type(type_param) = param {
            for bound in &mut type_param.bounds {
                if let syn::TypeParamBound::Trait(trait_bound) = bound {
                    for segment in &mut trait_bound.path.segments {
                        if let syn::PathArguments::AngleBracketed(args) = &mut segment.arguments {
                            for arg in &mut args.args {
                                if let syn::GenericArgument::Type(ty) = arg {
                                    replace_op_tensor_with_tensor(ty, true);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Generate method parameters (skip first parameter which becomes self)
    let mut method_params = Vec::new();
    let mut call_args = Vec::new();
    call_args.push(quote!(self.inner_or_source().clone()));

    // Skip the first parameter since it becomes self
    for param in params.iter().skip(1) {
        let param_name = Ident::new(&param.name, Span::call_site());
        let mut param_type = param.pat_type.ty.clone();
        replace_op_tensor_with_tensor(&mut param_type, true);

        method_params.push(quote!(#param_name: #param_type));

        if is_option_with_tensor_or_self(&param_type) {
            call_args.push(quote!(#param_name.map(|t| t.inner_or_source().clone())));
        } else if param.is_op_tensor {
            call_args.push(quote!(#param_name.inner_or_source().clone()));
        } else {
            call_args.push(quote!(#param_name));
        }
    }

    // Generate return type
    let mut return_type = original_fn.sig.output.clone();
    if let ReturnType::Type(_, ref mut ty) = return_type {
        replace_op_tensor_with_tensor(ty, true);
    }

    let method = quote! {
        pub fn #method_name #generics (self, #(#method_params),*) #return_type {
            #kernel_name(#(#call_args),*).map(Self::wrap)
        }
    };

    Ok(method)
}

/// Generate inplace method variant
fn generate_method_inplace_variant(
    original_fn: &ItemFn,
    params: &[ParamInfo],
    kernel_name: &Ident,
) -> Result<TokenStream2> {
    let method_name = Ident::new(
        &format!("{}_", original_fn.sig.ident),
        original_fn.sig.ident.span(),
    );
    let mut generics = original_fn.sig.generics.clone();

    // Replace OpTensor with Self in generics
    for param in &mut generics.params {
        if let GenericParam::Type(type_param) = param {
            for bound in &mut type_param.bounds {
                if let syn::TypeParamBound::Trait(trait_bound) = bound {
                    for segment in &mut trait_bound.path.segments {
                        if let syn::PathArguments::AngleBracketed(args) = &mut segment.arguments {
                            for arg in &mut args.args {
                                if let syn::GenericArgument::Type(ty) = arg {
                                    replace_op_tensor_with_tensor(ty, true);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Generate method parameters (skip first parameter which becomes self)
    let mut method_params = Vec::new();
    let mut call_args = Vec::new();
    call_args.push(quote!(inner));

    // Skip the first parameter since it becomes self
    for param in params.iter().skip(1) {
        let param_name = Ident::new(&param.name, Span::call_site());
        let mut param_type = param.pat_type.ty.clone();
        replace_op_tensor_with_tensor(&mut param_type, true);

        method_params.push(quote!(#param_name: #param_type));

        if is_option_with_tensor_or_self(&param_type) {
            call_args.push(quote!(#param_name.map(|t| t.inner_or_source().clone())));
        } else if param.is_op_tensor {
            call_args.push(quote!(#param_name.inner_or_source().clone()));
        } else {
            call_args.push(quote!(#param_name));
        }
    }

    // Generate return type
    let mut return_type = original_fn.sig.output.clone();
    if let ReturnType::Type(_, ref mut ty) = return_type {
        replace_op_tensor_with_tensor(ty, true);
    }

    let method = quote! {
        pub fn #method_name #generics (self, #(#method_params),*) #return_type {
            let inner = self.inner_or_source().clone();
            Ok(self.wrap_inplace(#kernel_name(#(#call_args),*)?))
        }
    };

    Ok(method)
}

/// Main function to process a tensor operation
pub fn process_tensor_op(attr: TensorOpAttr, item: ItemFn) -> Result<TokenStream2> {
    // Parse parameters with generic constraints consideration
    let params = analyze_parameters_with_generics(&item.sig.inputs, &item.sig.generics)?;

    // Build OpTensor mapping - only for direct OpTensor parameters
    let mut op_tensor_map = HashMap::new();
    let mut ot_gen = OtGen::new();

    for param in &params {
        if param.is_op_tensor {
            op_tensor_map.insert(param.name.clone(), ot_gen.next());
        }
    }

    // Generate kernel function
    let kernel_name = Ident::new(&format!("{}_kernel", item.sig.ident), item.sig.ident.span());
    let kernel_fn = generate_kernel_function(&item, &params, &op_tensor_map)?;

    let mut output = quote!(#kernel_fn);

    // Generate variants
    for variant in &attr.variants {
        match variant {
            TensorOpVariant::Function => {
                let func = generate_function_variant(&item, &params, &kernel_name)?;
                output.extend(quote!(#func));
            }
            TensorOpVariant::Method => {
                let method = generate_method_variant(&item, &params, &kernel_name)?;
                output.extend(quote! {
                    impl Tensor {
                        #method
                    }
                });
            }
            TensorOpVariant::MethodInplace => {
                let method = generate_method_inplace_variant(&item, &params, &kernel_name)?;
                output.extend(quote! {
                    impl Tensor {
                        #method
                    }
                });
            }
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use quote::ToTokens;
    use syn::parse_quote;

    #[test]
    fn test_parse_tensor_op_attr_empty() {
        let attr: TensorOpAttr = syn::parse_str("").unwrap();
        assert_eq!(attr.variants.len(), 2);
        assert!(attr.variants.contains(&TensorOpVariant::Function));
        assert!(attr.variants.contains(&TensorOpVariant::Method));
    }

    #[test]
    fn test_parse_tensor_op_attr_with_variants() {
        let attr: TensorOpAttr =
            syn::parse_str("variants = [function, method, method_inplace]").unwrap();
        assert_eq!(attr.variants.len(), 3);
        assert!(attr.variants.contains(&TensorOpVariant::Function));
        assert!(attr.variants.contains(&TensorOpVariant::Method));
        assert!(attr.variants.contains(&TensorOpVariant::MethodInplace));
    }

    #[test]
    fn test_is_op_tensor_path() {
        let path: Path = parse_quote!(OpTensor);
        assert!(is_op_tensor_path(&path));

        let path: Path = parse_quote!(crate::tensor::OpTensor);
        assert!(is_op_tensor_path(&path));

        let path: Path = parse_quote!(Tensor);
        assert!(!is_op_tensor_path(&path));
    }

    #[test]
    fn test_mentions_op_tensor() {
        let ty: Type = parse_quote!(OpTensor);
        assert!(mentions_op_tensor(&ty));

        let ty: Type = parse_quote!(TensorTypeOrScalar<OpTensor>);
        assert!(mentions_op_tensor(&ty));

        let ty: Type = parse_quote!(f64);
        assert!(!mentions_op_tensor(&ty));

        let ty: Type = parse_quote!(TensorTypeOrScalar<Tensor>);
        assert!(!mentions_op_tensor(&ty));
    }

    #[test]
    fn test_replace_op_tensor_with_tensor() {
        let mut ty: Type = parse_quote!(OpTensor);
        replace_op_tensor_with_tensor(&mut ty, false);
        assert_eq!(ty.to_token_stream().to_string(), "Tensor");

        let mut ty: Type = parse_quote!(TensorTypeOrScalar<OpTensor>);
        replace_op_tensor_with_tensor(&mut ty, true);
        assert_eq!(
            ty.to_token_stream().to_string(),
            "TensorTypeOrScalar < Self >"
        );
    }

    #[test]
    fn test_parse_parameters_with_defaults() {
        let inputs: Punctuated<FnArg, Token![,]> = parse_quote!(
            lhs: OpTensor,
            rhs: OpTensor,
            #[default(1.0)] alpha: f64
        );

        let params = parse_parameters(&inputs).unwrap();
        assert_eq!(params.len(), 3);
        assert_eq!(params[0].name, "lhs");
        assert!(params[0].is_op_tensor);
        assert!(params[0].default.is_none());
        assert_eq!(params[2].name, "alpha");
        assert!(params[2].default.is_some());
    }

    #[test]
    fn test_parse_parameters_invalid_default_order() {
        let inputs: Punctuated<FnArg, Token![,]> = parse_quote!(
            #[default(1.0)] alpha: f64,
            beta: f64
        );

        let result = parse_parameters(&inputs);
        assert!(result.is_err());
    }

    #[test]
    fn test_generate_kernel_function() {
        let original_fn: ItemFn = parse_quote! {
            fn add<T: TensorTypeOrScalar<OpTensor>>(lhs: OpTensor, rhs: T) -> Result<OpTensor> {
                // implementation
            }
        };

        let params = parse_parameters(&original_fn.sig.inputs).unwrap();
        let mut op_tensor_map = HashMap::new();
        let mut ot_gen = OtGen::new();

        for param in &params {
            if param.is_op_tensor {
                op_tensor_map.insert(param.name.clone(), ot_gen.next());
            }
        }

        let kernel_fn = generate_kernel_function(&original_fn, &params, &op_tensor_map).unwrap();

        assert_eq!(kernel_fn.sig.ident, "add_kernel");
        assert!(kernel_fn.sig.generics.params.len() >= 2); // Should have OT1, OT2 generics

        let kernel_str = kernel_fn.to_token_stream().to_string();
        assert!(kernel_str.contains("pub (crate)"));
        assert!(kernel_str.contains("OT1 : Into < OpTensor >"));
        assert!(kernel_str.contains("OT2 : Into < OpTensor >"));

        let formatted_output = prettyplease::unparse(&syn::File {
            shebang: None,
            attrs: vec![],
            items: vec![syn::Item::Fn(kernel_fn)],
        });

        println!("{formatted_output}");
    }

    #[test]
    fn test_generate_function_variant() {
        let original_fn: ItemFn = parse_quote! {
            fn add(lhs: OpTensor, rhs: OpTensor) -> Result<OpTensor> {
                // implementation
            }
        };

        let params = parse_parameters(&original_fn.sig.inputs).unwrap();
        let kernel_name = Ident::new("add_kernel", Span::call_site());

        let func = generate_function_variant(&original_fn, &params, &kernel_name).unwrap();

        let func_str = func.to_token_stream().to_string();
        assert!(func_str.contains("fn add"));
        assert!(func_str.contains("lhs : Tensor"));
        assert!(func_str.contains("rhs : Tensor"));
        assert!(func_str.contains("Result < Tensor >"));
        assert!(func_str.contains("add_kernel"));
        assert!(func_str.contains(". map (Tensor :: wrap)"));

        // let formatted_output = prettyplease::unparse(&syn::File {
        //     shebang: None,
        //     attrs: vec![],
        //     items: vec![syn::Item::Fn(func)],
        // });

        // println!("{formatted_output}");
    }

    #[test]
    fn test_generate_method_variant() {
        let original_fn: ItemFn = parse_quote! {
            fn add(lhs: OpTensor, rhs: OpTensor) -> Result<OpTensor> {
                // implementation
            }
        };

        let params = parse_parameters(&original_fn.sig.inputs).unwrap();
        let kernel_name = Ident::new("add_kernel", Span::call_site());

        let method = generate_method_variant(&original_fn, &params, &kernel_name).unwrap();

        let method_str = method.to_token_stream().to_string();
        assert!(method_str.contains("pub fn add"));
        assert!(method_str.contains("self"));
        assert!(method_str.contains("Result < Self >"));
        assert!(method_str.contains("inner_or_source"));
        assert!(method_str.contains("Self :: wrap"));
    }

    #[test]
    fn test_generate_method_inplace_variant() {
        let original_fn: ItemFn = parse_quote! {
            fn add(lhs: OpTensor, rhs: OpTensor) -> Result<OpTensor> {
                // implementation
            }
        };

        let params = parse_parameters(&original_fn.sig.inputs).unwrap();
        let kernel_name = Ident::new("add_kernel", Span::call_site());

        let method = generate_method_inplace_variant(&original_fn, &params, &kernel_name).unwrap();

        let method_str = method.to_token_stream().to_string();
        assert!(method_str.contains("pub fn add_"));
        assert!(method_str.contains("self"));
        assert!(method_str.contains("wrap_inplace"));
    }

    #[test]
    fn test_process_tensor_op_complete() {
        let attr = TensorOpAttr {
            variants: vec![TensorOpVariant::Function, TensorOpVariant::Method],
        };

        let item: ItemFn = parse_quote! {
            fn add<T: TensorTypeOrScalar<OpTensor>>(lhs: OpTensor, rhs: T) -> Result<OpTensor> {
                // implementation
            }
        };

        let result = process_tensor_op(attr, item).unwrap();
        let result_str = result.to_token_stream().to_string();

        // Should contain kernel function
        assert!(result_str.contains("pub (crate) fn add_kernel"));

        // Should contain function variant
        assert!(result_str.contains("fn add"));
        assert!(result_str.contains("lhs : Tensor"));

        // Should contain method variant
        assert!(result_str.contains("impl Tensor"));
        assert!(result_str.contains("pub fn add"));
    }

    #[test]
    fn test_process_tensor_op_with_generics() {
        let attr = TensorOpAttr {
            variants: vec![TensorOpVariant::Function],
        };

        let item: ItemFn = parse_quote! {
            fn scatter_add<D: Dim>(input: OpTensor, indices: OpTensor, source: OpTensor, dim: D) -> Result<OpTensor> {
                // implementation
            }
        };

        let result = process_tensor_op(attr, item).unwrap();
        let result_str = result.to_token_stream().to_string();

        // Should preserve original generics
        assert!(result_str.contains("D : Dim"));

        // Should add OT generics
        assert!(result_str.contains("OT1 : Into < OpTensor >"));
        assert!(result_str.contains("OT2 : Into < OpTensor >"));
        assert!(result_str.contains("OT3 : Into < OpTensor >"));

        // Should handle non-tensor parameters correctly
        assert!(result_str.contains("dim : D"));
    }

    #[test]
    fn test_process_tensor_op_with_defaults() {
        let attr = TensorOpAttr {
            variants: vec![TensorOpVariant::Function],
        };

        let item: ItemFn = parse_quote! {
            fn add(lhs: OpTensor, rhs: OpTensor, #[default(1.0)] alpha: f64) -> Result<OpTensor> {
                // implementation
            }
        };

        let result = process_tensor_op(attr, item).unwrap();
        let result_str = result.to_token_stream().to_string();

        // Kernel should not have default attributes
        assert!(result_str.contains("alpha : f64"));

        // Function variant should not have default attributes either
        // (defaults would be handled at a higher level)
        assert!(!result_str.contains("#[default"));
    }

    #[test]
    fn test_process_tensor_op_with_generic_constraints() {
        let attr = TensorOpAttr {
            variants: vec![TensorOpVariant::Function],
        };

        let item: ItemFn = parse_quote! {
            fn add<T: TensorTypeOrScalar<OpTensor>, T2: TensorTypeOrScalar<OpTensor>>(
                lhs: OpTensor,
                rhs: T,
                rhs2: T2,
            ) -> Result<OpTensor> {
                // implementation
            }
        };

        let result = process_tensor_op(attr, item).unwrap();
        let result_str = result.to_token_stream().to_string();

        // Should contain kernel function with proper generic constraint replacement
        assert!(result_str.contains("pub (crate) fn add_kernel"));

        // The generic constraints should be updated to use unique OT generics
        // T: TensorTypeOrScalar<OpTensor> should become T: TensorTypeOrScalar<OT2>
        // T2: TensorTypeOrScalar<OpTensor> should become T2: TensorTypeOrScalar<OT3>
        assert!(result_str.contains("T : TensorTypeOrScalar < OT2 >"));
        assert!(result_str.contains("T2 : TensorTypeOrScalar < OT3 >"));

        // Should have OT generics (OT1 for lhs, OT2-OT3 for constraints)
        assert!(result_str.contains("OT1 : Into < OpTensor >"));
        assert!(result_str.contains("OT2 : Into < OpTensor >"));
        assert!(result_str.contains("OT3 : Into < OpTensor >"));
    }

    #[test]
    fn test_kernel_function_body_conversions() {
        let attr = TensorOpAttr {
            variants: vec![TensorOpVariant::Function],
        };

        let item: ItemFn = parse_quote! {
            fn complex_op<T: TensorTypeOrScalar<OpTensor>>(
                input: OpTensor,
                param: T,
                optional: Option<OpTensor>,
            ) -> Result<OpTensor> {
                // Some original implementation
                let result = input + param.unwrap_or_default();
                Ok(result)
            }
        };

        let result = process_tensor_op(attr, item).unwrap();
        let result_str = result.to_token_stream().to_string();

        // Should contain the parameter conversions
        assert!(result_str.contains("let input = input . into ()"));
        assert!(result_str.contains("let param = param . map_tensor (| inner | inner . into ())"));
        assert!(result_str.contains("let optional = optional . map (| inner | inner . into ())"));

        // Should contain the original implementation
        assert!(result_str.contains("let result = input + param . unwrap_or_default ()"));
        assert!(result_str.contains("Ok (result)"));
    }

    #[test]
    fn test_method_first_parameter_becomes_self() {
        let attr = TensorOpAttr {
            variants: vec![TensorOpVariant::Method, TensorOpVariant::MethodInplace],
        };

        let item: ItemFn = parse_quote! {
            pub fn eq2<T: TensorTypeOrScalar<T>>(input: OpTensor, other: T) -> Result<OpTensor> {
                // Some implementation
                Ok(input)
            }
        };

        let result = process_tensor_op(attr, item).unwrap();
        let result_str = result.to_token_stream().to_string();

        // Should contain correct method signatures (only 'other' parameter, not 'input')
        assert!(
            result_str.contains("pub fn eq2 < T : TensorTypeOrScalar < T > > (self , other : T)")
        );
        assert!(
            result_str.contains("pub fn eq2_ < T : TensorTypeOrScalar < T > > (self , other : T)")
        );

        // Should NOT contain 'input' in method parameters
        assert!(!result_str.contains("(self , input"));

        // Should have correct kernel call arguments (self replaces first parameter)
        assert!(result_str.contains("eq2_kernel (self . inner_or_source () . clone () , other"));
    }

    #[test]
    fn test_your_eq2_example() {
        let attr = TensorOpAttr {
            variants: vec![
                TensorOpVariant::Function,
                TensorOpVariant::Method,
                TensorOpVariant::MethodInplace,
            ],
        };

        let item: ItemFn = parse_quote! {
            pub fn eq2<T: TensorTypeOrScalar<OpTensor>>(input: OpTensor, other: T) -> Result<OpTensor> {
                let device = input.device().clone();
                match other.tensor_or_scalar() {
                    Ok(TensorTypeOrScalarEnum::Tensor(other)) => {
                        let (lhs, rhs) = input.broadcast_for_binary_op(other)?;
                        let cmp = Cmp::new(lhs, TensorTypeOrScalarEnum::Tensor(rhs), (CmpOp::Eq));
                        let new_view = cmp.compute_view()?;
                        Ok(OpTensor::lazy(LazyOp::Cmp(cmp), new_view, device, false))
                    }
                    Ok(TensorTypeOrScalarEnum::Scalar(other)) => {
                        let device = input.device.clone();
                        let cmp = Cmp::new(input, TensorTypeOrScalarEnum::Scalar(other), (CmpOp::Eq));
                        let new_view = cmp.compute_view()?;
                        Ok(OpTensor::lazy(LazyOp::Cmp(cmp), new_view, device, false))
                    }
                    Err(e) => Err(e),
                }
            }
        };

        let result = process_tensor_op(attr, item).unwrap();
        let result_str = result.to_token_stream().to_string();

        // Should have kernel with original implementation
        assert!(result_str.contains("pub (crate) fn eq2_kernel"));
        assert!(result_str.contains("let input = input . into ()"));
        assert!(result_str.contains("let device = input . device () . clone ()"));

        // Should have function variant
        assert!(result_str.contains(
            "pub fn eq2 < T : TensorTypeOrScalar < Tensor > > (input : Tensor , other : T)"
        ));

        // Should have method variants with correct signatures (input becomes self)
        assert!(
            result_str
                .contains("pub fn eq2 < T : TensorTypeOrScalar < Self > > (self , other : T)")
        );
        assert!(
            result_str
                .contains("pub fn eq2_ < T : TensorTypeOrScalar < Self > > (self , other : T)")
        );

        // Should NOT have extra input parameter in methods
        assert!(!result_str.contains("(self , input : Self , other"));
    }

    #[test]
    fn test_option_op_tensor_parameters() {
        let attr = TensorOpAttr {
            variants: vec![TensorOpVariant::Function],
        };

        let item: ItemFn = parse_quote! {
            pub fn group_norm(
                input: OpTensor,
                num_groups: usize,
                weight: Option<OpTensor>,
                bias: Option<OpTensor>,
                eps: f32,
            ) -> Result<OpTensor> {
                // implementation
                Ok(input)
            }
        };

        let result = process_tensor_op(attr, item).unwrap();
        let result_str = result.to_token_stream().to_string();

        // Should have kernel with Option<OTn> generics
        assert!(result_str.contains("pub (crate) fn group_norm_kernel"));
        assert!(result_str.contains("weight : Option < OT2 >"));
        assert!(result_str.contains("bias : Option < OT3 >"));

        // Should have OT generics for all OpTensor parameters including Options
        assert!(result_str.contains("OT1 : Into < OpTensor >")); // input
        assert!(result_str.contains("OT2 : Into < OpTensor >")); // weight
        assert!(result_str.contains("OT3 : Into < OpTensor >")); // bias

        // Should have correct conversions
        assert!(result_str.contains("let input = input . into ()"));
        assert!(result_str.contains("let weight = weight . map (| inner | inner . into ())"));
        assert!(result_str.contains("let bias = bias . map (| inner | inner . into ())"));
    }

    #[test]
    fn test_option_tensor_in_methods() {
        let attr = TensorOpAttr {
            variants: vec![TensorOpVariant::Method, TensorOpVariant::MethodInplace],
        };

        let item: ItemFn = parse_quote! {
            pub fn group_norm(
                input: OpTensor,
                num_groups: usize,
                weight: Option<OpTensor>,
                bias: Option<OpTensor>,
                eps: f32,
            ) -> Result<OpTensor> {
                // implementation
                Ok(input)
            }
        };

        let result = process_tensor_op(attr, item).unwrap();
        let result_str = result.to_token_stream().to_string();

        // Should have method signatures with Option<Self>
        assert!(result_str.contains("weight : Option < Self >"));
        assert!(result_str.contains("bias : Option < Self >"));

        // Should have correct method calls with .map for Option<Self> parameters
        assert!(result_str.contains("weight . map (| t | t . inner_or_source () . clone ())"));
        assert!(result_str.contains("bias . map (| t | t . inner_or_source () . clone ())"));

        // Should NOT have .inner_or_source().clone() directly on Option parameters
        assert!(!result_str.contains("weight . inner_or_source () . clone ()"));
        assert!(!result_str.contains("bias . inner_or_source () . clone ()"));
    }

    #[test]
    fn test_option_tensor_in_function() {
        let attr = TensorOpAttr {
            variants: vec![TensorOpVariant::Function],
        };

        let item: ItemFn = parse_quote! {
            pub fn group_norm(
                input: OpTensor,
                num_groups: usize,
                weight: Option<OpTensor>,
                bias: Option<OpTensor>,
                eps: f32,
            ) -> Result<OpTensor> {
                // implementation
                Ok(input)
            }
        };

        let result = process_tensor_op(attr, item).unwrap();
        let result_str = result.to_token_stream().to_string();

        // Should have function signature with Option<Tensor>
        assert!(result_str.contains("weight : Option < Tensor >"));
        assert!(result_str.contains("bias : Option < Tensor >"));

        // Should have simple parameter passing (no .map needed for function variant)
        assert!(
            result_str.contains("group_norm_kernel (input , num_groups , weight , bias , eps)")
        );
    }

    #[test]
    fn test_container_type_correct_automatic_conversion() {
        let attr = TensorOpAttr {
            variants: vec![TensorOpVariant::Function],
        };

        let item: ItemFn = parse_quote! {
            pub fn cat<D: Dim>(tensors: RVec<OpTensor>, dim: D) -> Result<OpTensor> {
                let dim = dim.to_index(tensors[0].shape(), "cat")?;
                let device = tensors[0].device().clone();
                assert!(tensors.iter().all(|t| t.device == device), "Mixed devices");

                let cat = Concat::new(tensors, dim);
                let new_view = cat.compute_view()?;
                Ok(OpTensor::lazy(LazyOp::Concat(cat), new_view, device, false))
            }
        };

        let result = process_tensor_op(attr, item).unwrap();
        let result_str = result.to_token_stream().to_string();

        // Should NOT contain wrong conversion type (map_tensor) for RVec<OpTensor>
        assert!(!result_str.contains("let tensors = tensors . map_tensor"));

        // Should contain correct automatic conversion for container types
        assert!(result_str.contains(
            "let tensors = tensors . into_iter () . map (| inner | inner . into ()) . collect :: < RVec < OpTensor > > ()"
        ));

        // Should have function signature with RVec<Tensor>
        assert!(result_str.contains("tensors : RVec < Tensor >"));

        // Should have kernel signature with RVec<OT1> where OT1: Into<OpTensor>
        assert!(result_str.contains("tensors : RVec < OT1 >"));
        assert!(result_str.contains("OT1 : Into < OpTensor >"));

        // Should contain kernel function call
        assert!(result_str.contains("cat_kernel (tensors , dim)"));
    }
}
