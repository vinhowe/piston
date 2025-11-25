use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{ImplItem, ItemImpl, Type, parse2};

pub fn scoped_module(item: TokenStream) -> TokenStream {
    let input = proc_macro2::TokenStream::from(item);
    scoped_module_impl(input).into()
}

fn scoped_module_impl(input: TokenStream2) -> TokenStream2 {
    let mut impl_block = parse2::<ItemImpl>(input).expect("Expected impl block");
    let self_ty = &impl_block.self_ty;

    let mut has_module_name = false;

    for item in &impl_block.items {
        if let ImplItem::Fn(method) = item
            && method.sig.ident == "module_name"
        {
            has_module_name = true;
            break;
        }
    }

    for item in &mut impl_block.items {
        if let ImplItem::Fn(method) = item
            && method.sig.ident == "schedule"
        {
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

#[cfg(test)]
mod tests {
    use super::*;
    use quote::quote;

    #[test]
    fn test_scoped_module_with_existing_module_name() {
        let input = quote! {
            impl MyModule {
                fn module_name(&self) -> &str {
                    "custom_module"
                }

                fn schedule(&self) {
                    println!("original schedule");
                }
            }
        };

        let output = scoped_module_impl(input);
        let output_str = output.to_string();

        // Should wrap the schedule method with scope guard
        assert!(output_str.contains("_scope_guard"));
        assert!(output_str.contains("ScopePusher"));
        assert!(output_str.contains("self . module_name ()"));
        assert!(output_str.contains("original schedule"));

        // Should not add a new module_name method
        assert_eq!(output_str.matches("fn module_name").count(), 1);
    }

    #[test]
    fn test_scoped_module_without_module_name() {
        let input = quote! {
            impl TestModule {
                fn schedule(&self) {
                    self.run_logic();
                }
            }
        };

        let output = scoped_module_impl(input);
        let output_str = output.to_string();

        // Should wrap the schedule method with scope guard
        assert!(output_str.contains("_scope_guard"));
        assert!(output_str.contains("ScopePusher"));

        // Should add a default module_name method
        assert!(output_str.contains("fn module_name"));
        assert!(output_str.contains("TestModule"));

        // Should preserve original schedule logic
        assert!(output_str.contains("self . run_logic ()"));
    }

    #[test]
    fn test_scoped_module_without_schedule_method() {
        let input = quote! {
            impl AnotherModule {
                fn some_other_method(&self) {
                    println!("other method");
                }
            }
        };

        let output = scoped_module_impl(input);
        let output_str = output.to_string();

        // Should add a default module_name method
        assert!(output_str.contains("fn module_name"));
        assert!(output_str.contains("AnotherModule"));

        // Should preserve the original method
        assert!(output_str.contains("some_other_method"));
        assert!(output_str.contains("other method"));

        // Should not add scope guard since no schedule method
        assert!(!output_str.contains("_scope_guard"));
    }

    #[test]
    fn test_scoped_module_with_complex_type() {
        let input = quote! {
            impl std::collections::HashMap<String, i32> {
                fn schedule(&self) {
                    self.process();
                }
            }
        };

        let output = scoped_module_impl(input);
        let output_str = output.to_string();

        // Should extract the last segment of the path for module name
        assert!(output_str.contains("HashMap"));
        assert!(output_str.contains("fn module_name"));
    }

    #[test]
    fn test_scoped_module_with_generic_type() {
        let input = quote! {
            impl<T> GenericModule<T> {
                fn schedule(&self) {
                    self.execute();
                }
            }
        };

        let output = scoped_module_impl(input);
        let output_str = output.to_string();

        // Should handle generic types
        assert!(output_str.contains("GenericModule"));
        assert!(output_str.contains("fn module_name"));
        assert!(output_str.contains("_scope_guard"));
    }

    #[test]
    fn test_scoped_module_preserves_other_methods() {
        let input = quote! {
            impl MultiMethodModule {
                fn schedule(&self) {
                    self.main_task();
                }

                fn helper_method(&self) -> i32 {
                    42
                }

                fn another_method(&mut self, param: String) {
                    self.value = param;
                }
            }
        };

        let output = scoped_module_impl(input);
        let output_str = output.to_string();

        // Should preserve all methods
        assert!(output_str.contains("helper_method"));
        assert!(output_str.contains("another_method"));
        assert!(output_str.contains("main_task"));

        // Should add scope guard to schedule method
        assert!(output_str.contains("_scope_guard"));

        // Should add module_name method
        assert!(output_str.contains("fn module_name"));
        assert!(output_str.contains("MultiMethodModule"));
    }
}
