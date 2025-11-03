use heck::ToSnakeCase;
use proc_macro2::TokenStream;
use quote::quote;
use syn::{DeriveInput, parse2};

pub fn derive(input: TokenStream) -> TokenStream {
    let input = parse2::<DeriveInput>(input).unwrap();
    let struct_name = input.ident;

    let (with_fields, match_arms) = match input.data {
        syn::Data::Struct(data_struct) => {
            let with_fields = data_struct.fields.iter().enumerate().map(|(i, field)| {
                let (field_name, field_access) = match &field.ident {
                    Some(ident) => (quote! { stringify!(#ident) }, quote! { #ident }),
                    None => {
                        let index = syn::Index::from(i);
                        let field_name =
                            syn::LitStr::new(&i.to_string(), proc_macro2::Span::call_site());
                        (quote! { #field_name }, quote! { #index })
                    }
                };
                quote! {
                    .with_field(#field_name, self.#field_access.clone())
                }
            });
            (Some(with_fields.collect::<Vec<_>>()), None)
        }
        syn::Data::Enum(data_enum) => {
            let snake_name = struct_name.to_string().to_snake_case();
            let match_arms = data_enum.variants.iter().map(|variant| {
                if variant.fields.len() > 1 {
                    panic!("Enum variants with multiple fields are not supported");
                }
                let variant_ident = &variant.ident;

                if variant.fields.is_empty() {
                    return quote! {
                        #struct_name::#variant_ident => {
                            ir.with_field(#snake_name, stringify!(#variant_ident));
                        }
                    };
                }

                quote! {
                    #struct_name::#variant_ident(inner) => {
                        ir.with_field(#snake_name, stringify!(#variant_ident));
                        inner.ir_fields(ir);
                    }
                }
            });
            let match_arms = quote! { match self { #(#match_arms)* } };

            (None, Some(match_arms))
        }
        _ => panic!("Only structs and enums are supported"),
    };

    let with_fields_statement = with_fields.map(|fields| quote! { ir #(#fields)*; });

    let expanded = quote! {
        impl crate::IrFields for #struct_name {
            fn ir_fields(&self, ir: &mut crate::Ir) {
                #with_fields_statement
                #match_arms
            }
        }
    };

    expanded
}
