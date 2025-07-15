use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{parse2, ImplItem, ItemImpl, Type};

pub fn scoped_module(item: TokenStream) -> TokenStream {
    let input = proc_macro2::TokenStream::from(item);
    scoped_module_impl(input).into()
}

fn scoped_module_impl(input: TokenStream2) -> TokenStream2 {
    let mut impl_block = parse2::<ItemImpl>(input).expect("Expected impl block");
    let self_ty = &impl_block.self_ty;

    let mut has_module_name = false;

    for item in &impl_block.items {
        if let ImplItem::Fn(method) = item {
            if method.sig.ident == "module_name" {
                has_module_name = true;
                break;
            }
        }
    }

    for item in &mut impl_block.items {
        if let ImplItem::Fn(method) = item {
            if method.sig.ident == "schedule" {
                let original_body = method.block.clone();

                method.block = syn::parse2(quote! {
                    {
                        let _scope_guard = piston::ScopePusher::new(&format!("mod:{}", self.module_name()));

                        #original_body
                    }
                })
                .unwrap();

                break;
            }
        }
    }

    // Generate a default module_name if not present
    if !has_module_name {
        let type_name = match self_ty.as_ref() {
            Type::Path(type_path) => {
                if let Some(segment) = type_path.path.segments.last() {
                    segment.ident.to_string()
                } else {
                    "unknown".to_string()
                }
            }
            _ => "unknown".to_string(),
        };

        let module_name_method = quote! {
            fn module_name(&self) -> &str {
                #type_name
            }
        };

        impl_block
            .items
            .push(syn::parse2(module_name_method).unwrap());
    }

    quote! { #impl_block }
}
