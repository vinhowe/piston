use crate::{DType, HashMap, RVec, Tensor};
use anyhow::Result;
use bitflags::bitflags;
use once_cell::sync::Lazy;
use parking_lot::RwLock;

bitflags! {
    #[derive(Default, Copy, Clone, Debug, PartialEq, Eq, Hash)]
    pub struct DispatchKeySet: u32 {
        #[cfg(target_arch = "wasm32")]
        const JAVASCRIPT  = 1 << 3;
        const PRINT       = 1 << 2;   // log every call - enabled by default
        const BACKEND     = 1 << 1;   // engine-side implementation
    }
}

// Thread-local include / exclude sets (mirrors PyTorch's TLS flags).
thread_local! {
    static TLS_INCLUDE: std::cell::Cell<DispatchKeySet> =
        const { std::cell::Cell::new(DispatchKeySet::PRINT) }; // PRINT enabled by default
    static TLS_EXCLUDE: std::cell::Cell<DispatchKeySet> =
        const { std::cell::Cell::new(DispatchKeySet::empty()) };
}

/// RAII guard for including dispatch keys in TLS
pub struct IncludeGuard {
    prev_include: DispatchKeySet,
    keys: DispatchKeySet,
}

impl IncludeGuard {
    pub fn new(keys: DispatchKeySet) -> Self {
        let prev_include = TLS_INCLUDE.with(|tls| tls.get());
        let new_include = prev_include | keys;
        TLS_INCLUDE.with(|tls| tls.set(new_include));

        Self { prev_include, keys }
    }
}

impl Drop for IncludeGuard {
    fn drop(&mut self) {
        TLS_INCLUDE.with(|tls| tls.set(self.prev_include));
    }
}

/// RAII guard for excluding dispatch keys from TLS
pub struct ExcludeGuard {
    prev_exclude: DispatchKeySet,
    keys: DispatchKeySet,
}

impl ExcludeGuard {
    pub fn new(keys: DispatchKeySet) -> Self {
        let prev_exclude = TLS_EXCLUDE.with(|tls| tls.get());
        let new_exclude = prev_exclude | keys;
        TLS_EXCLUDE.with(|tls| tls.set(new_exclude));

        Self { prev_exclude, keys }
    }
}

impl Drop for ExcludeGuard {
    fn drop(&mut self) {
        TLS_EXCLUDE.with(|tls| tls.set(self.prev_exclude));
    }
}

/// Iterator over dispatch keys in priority order (highest bit first)
pub struct DispatchKeyIterator {
    keys: DispatchKeySet,
}

impl DispatchKeyIterator {
    pub fn new(keys: DispatchKeySet) -> Self {
        Self { keys }
    }
}

impl Iterator for DispatchKeyIterator {
    type Item = DispatchKeySet;

    fn next(&mut self) -> Option<Self::Item> {
        if self.keys.bits() == 0 {
            return None;
        }

        // index of the highest bit that is still set
        let leading = self.keys.bits().leading_zeros();
        let idx = 31 - leading; // 0-based position from LSB
        let mask = DispatchKeySet::from_bits_truncate(1 << idx);

        self.keys.remove(mask); // clear it
        Some(mask)
    }
}

impl DispatchKeySet {
    /// Returns an iterator over the dispatch keys in priority order (highest bit first)
    pub fn iter_highest_first(self) -> DispatchKeyIterator {
        DispatchKeyIterator::new(self)
    }
}

// We do this just because we'll never do anything truly multithreaded with JS references, but it
// does throw us an error about safely sharing *mut u8 somewhere inside of wasm-bindgen.
unsafe impl Sync for OperatorHandlerRegistration {}

/// Metadata for an operator (one per operator name)
pub struct OperatorHandle {
    pub name: &'static str,
    pub call_fn: fn(&str, &mut RVec<Value>) -> Result<()>,
}

impl OperatorHandle {
    /// Re-dispatch through the operator system
    pub fn call(&self, stack: &mut RVec<Value>) -> Result<()> {
        (self.call_fn)(self.name, stack)
    }
}

/// New trait for handling dispatch with access to operator metadata
pub trait BoxedKernel: Send + Sync {
    fn handle(
        &self,
        op: &OperatorHandle,
        keys: DispatchKeySet,
        stack: &mut RVec<Value>,
    ) -> Result<()>;
}

pub struct KernelFunction {
    pub unboxed: Option<fn(&OperatorHandle, DispatchKeySet, &mut RVec<Value>) -> Result<()>>,
    // Store a reference to a statically allocated kernel implementation so we
    // can build these structures in const/static contexts without needing a
    // heap allocation (Box::new is not const).
    pub boxed: Option<Box<dyn BoxedKernel>>,
}

impl KernelFunction {
    pub fn call<R: StackValue>(
        &self,
        op: &OperatorHandle,
        keys: DispatchKeySet,
        mut stack: RVec<Value>,
    ) -> Result<R> {
        if let Some(f) = self.unboxed {
            // Fast path for unboxed functions
            f(op, keys, &mut stack)?;
            return Ok(R::pop(&mut stack));
        }

        self.boxed
            .as_ref()
            .expect("No boxed function")
            .handle(op, keys, &mut stack)?;

        Ok(R::pop(&mut stack))
    }
}

// We do this just because we'll never do anything truly multithreaded with JS references, but it
// does throw us an error about safely sharing *mut u8 somewhere inside of wasm-bindgen.
unsafe impl Sync for KernelFunction {}

/// Build the operators metadata map from all inventory submissions
pub static OPERATORS: Lazy<RwLock<HashMap<String, &'static OperatorHandle>>> = Lazy::new(|| {
    let mut map = HashMap::default();
    for reg in inventory::iter::<OperatorRegistration> {
        map.insert(reg.name.to_string(), reg.handle);
    }
    RwLock::new(map)
});

pub struct OperatorRegistration {
    pub name: &'static str,
    pub handle: &'static OperatorHandle,
}

inventory::collect!(OperatorRegistration);

/// Build the operators handlers map from all inventory submissions
/// Outer map: operator name -> inner map
/// Inner map: dispatch key -> handler implementation
pub static OPERATORS_HANDLERS: Lazy<
    RwLock<HashMap<String, HashMap<DispatchKeySet, &'static KernelFunction>>>,
> = Lazy::new(|| {
    let mut map = HashMap::default();
    for reg in inventory::iter::<OperatorHandlerRegistration> {
        let inner_map = map
            .entry(reg.name.to_string())
            .or_insert_with(HashMap::default);
        inner_map.insert(reg.key, reg.handler);
    }
    RwLock::new(map)
});

pub struct OperatorHandlerRegistration {
    pub name: &'static str,
    pub key: DispatchKeySet,
    pub handler: &'static KernelFunction,
}

inventory::collect!(OperatorHandlerRegistration);

// /// Helper function to register a kernel for a specific operator name and key
// pub fn register_kernel(
//     name: &str,
//     key: DispatchKeySet,
//     handler: &'static dyn DispatchOperatorHandler,
// ) {
//     OPERATORS_HANDLERS
//         .write()
//         .entry(name.to_string())
//         .or_default()
//         .insert(key, handler);
// }

// /// Implementation for concrete operator types
// impl<Args, Ret> DispatchableOperator for Operator<Args, Ret>
// where
//     Args: ExtractKeys + Send + 'static,
//     Ret: Send + 'static,
// {
//     fn dispatch_boxed(&self, args: Box<dyn Any + Send>) -> Result<Box<dyn Any + Send>> {
//         let args = *args
//             .downcast::<Args>()
//             .map_err(|_| anyhow::anyhow!("Type mismatch in operator arguments"))?;

//         let result = dispatch(self, args)?;
//         Ok(Box::new(result))
//     }

//     fn extract_keys_boxed(&self, args: &dyn Any) -> DispatchKeySet {
//         if let Some(args) = args.downcast_ref::<Args>() {
//             args.extract_keys()
//         } else {
//             DispatchKeySet::empty()
//         }
//     }
// }

// /// An unboxed kernel is a plain fn pointer.
// type UnboxedFn<Args, Ret> = fn(Args) -> Result<Ret>;
// #[cfg(target_arch = "wasm32")]
// /// A boxed kernel is a `JsValue` callable.
// type BoxedFn = wasm_bindgen::JsValue;

// pub struct Operator<Args, Ret> {
//     pub unboxed: UnboxedFn<Args, Ret>,
//     #[cfg(target_arch = "wasm32")]
//     pub boxed: BoxedFn, // only used when JS keys win
// }

// /// Type-safe dispatch for concrete operator types
// pub fn dispatch<Args, Ret>(op: &Operator<Args, Ret>, args: Args) -> Result<Ret>
// where
//     Args: ExtractKeys,
// {
//     let mut keys = args.extract_keys();
//     keys |= TLS_INCLUDE.with(|tls| tls.get());
//     keys &= !TLS_EXCLUDE.with(|tls| tls.get());

//     // #[cfg(target_arch = "wasm32")]
//     // if keys.contains(DispatchKeySet::JS_MODE) {
//     //     return call_js_mode(op, args, keys);
//     // }

//     (op.unboxed)(args)
// }

#[cfg(target_arch = "wasm32")]
fn call_js_mode<Args, Ret>(_op: &OperatorHandle, _args: Args, _ks: DispatchKeySet) -> Result<Ret>
where
    Args: 'static, // can wrap each element to JsValue
    Ret: 'static,  // simplify: works for tensor return
{
    // JS mode is not implemented yet
    Err(anyhow::anyhow!("JS mode dispatch not implemented"))
}

/// Type-erased dispatch function for registered operators
#[track_caller]
pub fn dispatch_operator<
    Args: ToStack + ExtractKeys + Send + 'static,
    Ret: StackValue + Send + 'static,
>(
    name: &str,
    args: Args,
) -> Result<Ret> {
    let meta_map = OPERATORS.read();
    let meta = meta_map
        .get(name)
        .ok_or_else(|| anyhow::anyhow!("Operator '{}' not found", name))?;

    let handlers_map = OPERATORS_HANDLERS.read();
    let inner_map = handlers_map
        .get(name)
        .ok_or_else(|| anyhow::anyhow!("No handlers for operator '{}'", name))?;

    // Compute effective key set
    let mut keys = args.extract_keys(); // Extract keys directly from typed args
    keys |= TLS_INCLUDE.with(|tls| tls.get());
    keys &= !TLS_EXCLUDE.with(|tls| tls.get());

    // let boxed_args = Box::new(args) as Box<dyn Any + Send>;

    // Dispatch in priority order (highest bit first, so BACKEND is last)
    let chosen_kernel = keys
        .iter_highest_first()
        .find_map(|key| inner_map.get(&key));

    let stack = args.to_stack();

    match chosen_kernel {
        Some(kernel) => {
            let result = kernel.call::<Ret>(meta, keys, stack)?;
            Ok(result)
        }
        None => Err(anyhow::anyhow!("No handler found for operator '{}'", name)),
    }
}

// /// Generic print operator that logs calls and redispatches
// struct PrintOp();

// impl BoxedKernel for PrintOp {
//     fn handle(
//         &self,
//         op: &OperatorHandle,
//         _keys: DispatchKeySet,
//         stack: &mut RVec<Value>,
//     ) -> Result<()> {
//         // Log the operation call
//         log::info!("[PRINT] {} called", op.name);

//         // Exclude PRINT to avoid infinite recursion and redispatch
//         let _guard = ExcludeGuard::new(DispatchKeySet::PRINT);

//         // Re-dispatch through the operator metadata
//         op.call(stack)
//     }
// }

// Macro to create print kernels for operators
macro_rules! register_print_kernel {
    ($op_name:literal) => {
        inventory::submit! {
            OperatorHandlerRegistration {
                name: $op_name,
                key: DispatchKeySet::PRINT,
                handler: &KernelFunction {
                    unboxed: Some(|op, _keys, stack| {
                        // Log the operation call
                        log::info!("[PRINT] {} called", op.name);

                        // Exclude PRINT
                        let _guard = ExcludeGuard::new(DispatchKeySet::PRINT);

                        // Re-dispatch through the operator metadata
                        op.call(stack)?;

                        Ok(())
                    }),
                    boxed: None,
                },
            }
        }
    };
}

// Register print kernels for common operations
// register_print_kernel!("cast");

// /// JavaScript operator handler
// #[cfg(target_arch = "wasm32")]
// struct JsOp();

// #[cfg(target_arch = "wasm32")]
// impl BoxedKernel for JsOp {
//     fn handle(
//         &self,
//         op: &OperatorHandle,
//         _keys: DispatchKeySet,
//         _stack: &mut RVec<Value>,
//     ) -> Result<()> {
//         // JS mode is not implemented yet
//         Err(anyhow::anyhow!(
//             "JavaScript dispatch not implemented for {}",
//             op.name
//         ))
//     }
// }

#[cfg(target_arch = "wasm32")]
macro_rules! register_js_kernel {
    ($op_name:literal) => {
        inventory::submit! {
            OperatorHandlerRegistration {
                name: $op_name,
                key: DispatchKeySet::JAVASCRIPT,
                handler: &KernelFunction {
                    unboxed: Some(|op, _keys, _stack| {
                        Err(anyhow::anyhow!(
                            "JavaScript dispatch not implemented for {}",
                            op.name
                        ))
                    }),
                    boxed: None,
                    // boxed: Some(Box::new(JsOp())),
                },
            }
        }
    };
}

#[cfg(target_arch = "wasm32")]
register_js_kernel!("cast");

pub trait ToStack {
    fn to_stack(self) -> RVec<Value>;
}

pub trait ExtractKeys {
    fn extract_keys(&self) -> DispatchKeySet;
}

// Blanket implementation for single values
impl<T: StackValue> ToStack for T {
    fn to_stack(self) -> RVec<Value> {
        let mut stack = RVec::new();
        self.push(&mut stack);
        stack
    }
}

// Helper macro that counts the number of identifiers passed in and expands to a `usize` constant.
macro_rules! count_idents {
    () => { 0usize };
    ($head:ident $(, $tail:ident)*) => { 1usize + count_idents!($($tail),*) };
}

// Helper macro that pushes the provided values onto the given stack **in reverse order** so that
// popping returns them in their original forward order. This is done entirely at compile-time with
// zero runtime overhead.
macro_rules! push_reverse {
    ($stack:expr $(,)?) => {};
    ($stack:expr, $head:expr $(, $tail:expr)* $(,)?) => {
        push_reverse!($stack $(, $tail)*);
        $head.push($stack);
    };
}

// Macro to generate tuple implementations
macro_rules! impl_tuple_traits {
    ( $( $name:ident ),+ ) => {
        impl< $( $name: ExtractKeys ),+ > ExtractKeys for ( $( $name, )+ ) {
            fn extract_keys(&self) -> DispatchKeySet {
                let mut keys = DispatchKeySet::empty();
                let ( $( ref $name, )+ ) = *self;
                $( keys |= $name.extract_keys(); )+
                keys
            }
        }

        impl< $( $name: StackValue ),+ > ToStack for ( $( $name, )+ ) {
            fn to_stack(self) -> RVec<Value> {
                // Reserve exactly the required capacity to avoid potential reallocations when the
                // tuple has more elements than the inline `SmallVec` capacity.
                let mut stack: RVec<Value> = RVec::new();
                stack.reserve_exact(count_idents!( $( $name ),+ ));

                // Destructure the tuple and push each element **in reverse order**.
                let ( $( $name, )+ ) = self;
                push_reverse!(&mut stack $(, $name )+ );
                stack
            }
        }
    };
}

// Generate implementations for tuples up to 6 elements
impl_tuple_traits! { T1 }
impl_tuple_traits! { T1, T2 }
impl_tuple_traits! { T1, T2, T3 }
impl_tuple_traits! { T1, T2, T3, T4 }
impl_tuple_traits! { T1, T2, T3, T4, T5 }
impl_tuple_traits! { T1, T2, T3, T4, T5, T6 }

impl<T: ExtractKeys> ExtractKeys for Option<&T> {
    fn extract_keys(&self) -> DispatchKeySet {
        self.map_or(DispatchKeySet::empty(), |t| t.extract_keys())
    }
}

impl<T: ExtractKeys> ExtractKeys for RVec<T> {
    fn extract_keys(&self) -> DispatchKeySet {
        self.iter()
            .map(|t| t.extract_keys())
            .fold(DispatchKeySet::empty(), |acc, keys| acc | keys)
    }
}

impl ExtractKeys for Tensor {
    fn extract_keys(&self) -> DispatchKeySet {
        let mut k = DispatchKeySet::BACKEND;
        // Tensors always have BACKEND by default
        // k |= DispatchKeySet::FAKE_TENSOR;
        // if self.autograd.requires_grad || self.autograd.grad_fn.is_some() {
        //     k |= DispatchKeySet::AUTOGRAD
        // }
        // #[cfg(target_arch = "wasm32")]
        // {
        //     if js_sys::Object::is_prototype_of(
        //         &wasm_bindgen::JsCast::unchecked_from_js(self.clone()),
        //         &JsTensorSubclass::prototype(),
        //     )
        //     // pseudo-API
        //     {
        //         k |= DispatchKeySet::JS_MODE;
        //     }
        // }
        k
    }
}

/// Macro to implement empty ExtractKeys for types that don't contribute any dispatch keys
macro_rules! impl_empty_extract_keys {
    ($($ty:ty),* $(,)?) => {
        $(
            impl ExtractKeys for $ty {
                fn extract_keys(&self) -> DispatchKeySet {
                    DispatchKeySet::empty()
                }
            }
        )*
    };
}

// Implement empty ExtractKeys for unit type and other basic types
impl_empty_extract_keys!((), DType);

#[derive(Clone)]
pub enum Value {
    Int(i64),
    Float(f64),
    Tensor(Tensor),
    DType(DType),
}

// --- boxing ---
pub trait StackValue: Sized {
    fn push(self, stack: &mut RVec<Value>);
    fn pop(stack: &mut RVec<Value>) -> Self;
}

macro_rules! impl_stack_value {
    ($type:ty, $variant:ident) => {
        impl StackValue for $type {
            fn push(self, stack: &mut RVec<Value>) {
                stack.push(Value::$variant(self));
            }
            fn pop(stack: &mut RVec<Value>) -> Self {
                match stack.pop().unwrap() {
                    Value::$variant(value) => value,
                    _ => panic!("Expected {} but got different type", stringify!($variant)),
                }
            }
        }
    };
}

impl_stack_value!(i64, Int);
impl_stack_value!(f64, Float);
impl_stack_value!(Tensor, Tensor);
impl_stack_value!(DType, DType);

// // variadic macro to box all args
// #[macro_export]
// macro_rules! box_args {
//     ($stack:expr,) => {};
//     ($stack:expr, $head:expr $(, $tail:expr)* $(,)?) => {
//         $head.push($stack);
//         box_args!($stack $(, $tail)*);
//     };
// }

// --- unboxing ---
// pub trait PopFromStack: Sized {}
// impl PopFromStack for () {
//     fn pop(_: &mut Vec<Value>) -> () {}
// }
// impl PopFromStack for Tensor { … }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_system_basic_functionality() {
        // Test that the dispatch system can find and execute handlers
        // This test primarily ensures no panics occur during dispatch

        // The cast operation should have metadata registered
        let meta_map = OPERATORS.read();
        assert!(
            meta_map.contains_key("cast"),
            "Cast operator metadata should be registered"
        );

        // The cast operation should have both PRINT and BACKEND handlers registered
        let handlers_map = OPERATORS_HANDLERS.read();
        let cast_handlers = handlers_map.get("cast");

        assert!(
            cast_handlers.is_some(),
            "Cast operator handlers should be registered"
        );
        let cast_handlers = cast_handlers.unwrap();

        assert!(
            cast_handlers.contains_key(&DispatchKeySet::PRINT),
            "Cast should have PRINT handler"
        );
        assert!(
            cast_handlers.contains_key(&DispatchKeySet::BACKEND),
            "Cast should have BACKEND handler"
        );
    }

    #[test]
    fn exclude_guard_works() {
        // Test that ExcludeGuard properly excludes keys
        let initial_exclude = TLS_EXCLUDE.with(|tls| tls.get());

        {
            let _guard = ExcludeGuard::new(DispatchKeySet::PRINT);
            let current_exclude = TLS_EXCLUDE.with(|tls| tls.get());
            assert!(current_exclude.contains(DispatchKeySet::PRINT));
        }

        // After guard is dropped, should be back to initial state
        let final_exclude = TLS_EXCLUDE.with(|tls| tls.get());
        assert_eq!(initial_exclude, final_exclude);
    }

    #[test]
    fn include_guard_works() {
        // Test that IncludeGuard properly includes keys
        let initial_include = TLS_INCLUDE.with(|tls| tls.get());

        {
            let _guard = IncludeGuard::new(DispatchKeySet::BACKEND);
            let current_include = TLS_INCLUDE.with(|tls| tls.get());
            assert!(current_include.contains(DispatchKeySet::BACKEND));
        }

        // After guard is dropped, should be back to initial state
        let final_include = TLS_INCLUDE.with(|tls| tls.get());
        assert_eq!(initial_include, final_include);
    }

    #[test]
    fn priority_ordering_works() {
        // Test that keys are processed in priority order (highest bit first)
        let keys = DispatchKeySet::PRINT | DispatchKeySet::BACKEND;
        let visited: Vec<DispatchKeySet> = keys.iter_highest_first().collect();

        // PRINT (bit 2) should come before BACKEND (bit 1)
        assert_eq!(visited.len(), 2);
        assert_eq!(visited[0], DispatchKeySet::PRINT);
        assert_eq!(visited[1], DispatchKeySet::BACKEND);
    }
}
