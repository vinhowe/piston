use js_sys::Array;
use js_sys::Promise;
use js_sys::{Function, Reflect};
use piston::{RVec, rvec};
use std::cell::RefCell;
use wasm_bindgen::JsCast;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::{JsFuture, future_to_promise};

use crate::error::IntoJsError;
use crate::js_util::is_subclass;
use crate::with_piston_web_module;

// Function-mode constructor set from JavaScript
thread_local! {
    static FUNCTION_MODE_CONSTRUCTOR: RefCell<Option<Function>> = const { RefCell::new(None) };
}

/// Register the base FunctionMode constructor from JS.
#[wasm_bindgen(js_name = _setFunctionModeConstructor)]
pub fn set_function_mode_constructor(constructor: &Function) {
    FUNCTION_MODE_CONSTRUCTOR.with(|cell| {
        cell.borrow_mut().replace(constructor.clone());
    });
}

// MarkStep-mode constructor set from JavaScript
thread_local! {
    static MARK_STEP_MODE_CONSTRUCTOR: RefCell<Option<Function>> = const { RefCell::new(None) };
}

/// Register the base MarkStepMode constructor from JS.
#[wasm_bindgen(js_name = _setMarkStepModeConstructor)]
pub fn set_mark_step_mode_constructor(constructor: &Function) {
    MARK_STEP_MODE_CONSTRUCTOR.with(|cell| {
        cell.borrow_mut().replace(constructor.clone());
    });
}

#[inline]
fn get_type_of_jsvalue(obj_or_type: &JsValue) -> JsValue {
    if obj_or_type.is_instance_of::<Function>() {
        // Already a constructor / class
        obj_or_type.clone()
    } else {
        // For regular values look up their `constructor` property
        Reflect::get(obj_or_type, &JsValue::from_str("constructor")).unwrap_or(JsValue::UNDEFINED)
    }
}

fn append_overloaded_arg<'a>(
    overloaded_args: &mut RVec<&'a JsValue>,
    obj: &'a JsValue,
    obj_is_type: bool,
) {
    let obj_type = if obj_is_type {
        obj
    } else {
        &get_type_of_jsvalue(obj)
    };

    // Skip if we've already seen this constructor
    if overloaded_args
        .iter()
        .any(|arg| JsValue::eq(obj_type, &get_type_of_jsvalue(arg)))
    {
        return;
    }

    // Default insertion point is "after everyone else"
    let mut insert_at = overloaded_args.len();

    // But if this is a *subclass* of something we have seen,
    // it has to come *before* its superclass.
    for (idx, arg) in overloaded_args.iter().enumerate() {
        if is_subclass(obj_type, &get_type_of_jsvalue(arg)) {
            insert_at = idx;
            break;
        }
    }

    overloaded_args.insert(insert_at, obj);
}

fn check_has_piston_function(mode_obj: &JsValue) -> bool {
    let pistion_function_func: Option<Function> =
        Reflect::get(mode_obj, &JsValue::from_str("_pistonFunction"))
            .and_then(|v: JsValue| v.dyn_into())
            .ok();
    pistion_function_func.is_some()
}

pub fn get_overloaded_args<'a, const N: usize>(args: &[&'a JsValue; N]) -> RVec<&'a JsValue> {
    // If we're currently in subclass dispatch, suppress further subclass-based overloading
    // to avoid recursive redispatch when the subclass calls the original function.
    if is_subclass_dispatch_active() {
        return rvec![];
    }
    let mut overloaded_args = rvec![];
    for &arg in args {
        if check_has_piston_function(arg) {
            append_overloaded_arg(&mut overloaded_args, arg, false);
        }
    }
    overloaded_args
}

// #[cfg(disable)]
fn dispatch_on_mode(
    args: &RVec<&JsValue>,
    named_args: &JsValue,
    js_types: &RVec<JsValue>,
    original_function: &Function,
) -> Result<Option<JsValue>, JsError> {
    let mut _mode_guard = StashFunctionModeGuard::new();
    let mode_obj = &_mode_guard
        .current_mode
        .as_ref()
        .expect("No mode object")
        .js_mode_obj;

    let piston_function_func: Function =
        Reflect::get(mode_obj, &JsValue::from_str("_pistonFunction"))
            .and_then(|v: JsValue| v.dyn_into())
            .map_err(|_| JsError::new("Mode object does not have a _pistonFunction method"))?;

    let ret = piston_function_func
        .apply(
            mode_obj,
            &Array::of4(
                original_function,
                &Array::from_iter(js_types),
                &Array::from_iter(args),
                named_args,
            ),
        )
        .map_err(|e| e.into_js_error())?;

    // If the handler returned a Promise, keep function-mode disabled until it settles.
    if let Some(promise) = ret.dyn_ref::<Promise>() {
        // Prevent Drop from restoring the mode now; it will be restored in the finally callback.
        let saved_mode = std::rc::Rc::new(std::cell::RefCell::new(_mode_guard.current_mode.take()));
        let saved_mode_for_cb = saved_mode.clone();

        let finally_cb = Closure::once(move || {
            if let Some(mode) = saved_mode_for_cb.borrow_mut().take() {
                push_function_mode(mode);
            }
        });

        let new_promise = promise.finally(&finally_cb);
        // Leak the closure to keep it alive until JS calls it.
        finally_cb.forget();
        return Ok(Some(new_promise.unchecked_into()));
    }

    // Immediate value: let the guard restore on drop; just propagate value/None.
    if ret.is_undefined() || ret.is_null() {
        Ok(None)
    } else {
        Ok(Some(ret))
    }
}

// #[cfg(disable)]
fn dispatch_on_subclass(
    args: &RVec<&JsValue>,
    named_args: &JsValue,
    overloaded_args: &RVec<&JsValue>,
    js_types: &RVec<JsValue>,
    original_function: &Function,
) -> Result<JsValue, JsError> {
    let js_types_array = Array::from_iter(js_types);

    let mut ret = None;
    for (arg, js_type) in overloaded_args.iter().zip(js_types_array.iter()) {
        // Get function using reflection, getting _pistonFunction or whatever we call it
        let piston_function_func: Function =
            Reflect::get(arg, &JsValue::from_str("_pistonFunction"))
                .and_then(|v: JsValue| v.dyn_into())
                .map_err(|_| JsError::new("Argument does not have a _pistonFunction method"))?;

        // -- In PyTorch, we skip disabled torch dispatches for infra modes. Don't know how that'll
        // pan out here

        // -- In PyTorch, if it's a plain method, not a classmethod, we throw an error here

        // -- !!! (this seems important) In PyTorch, we do something different if this isn't in
        // torchfunction mode. I don't know what that means, and most of the useful places I see it
        // do call it with torchfunction mode.

        // Suppress nested subclass overloading while the original function is called.
        push_subclass_dispatch();
        let applied = piston_function_func
            .apply(
                // This is a static method, so we pass in the type
                &js_type,
                &Array::of4(
                    original_function,
                    &js_types_array,
                    &Array::from_iter(args),
                    named_args,
                ),
            )
            .map_err(|e| e.into_js_error());

        let ret_ = match applied {
            Ok(v) => v,
            Err(e) => {
                pop_subclass_dispatch();
                return Err(e);
            }
        };

        if let Some(promise) = ret_.dyn_ref::<Promise>() {
            // Keep subclass-dispatch suppression active until the Promise settles.
            let finally_cb = Closure::once(move || {
                pop_subclass_dispatch();
            });
            let new_promise = promise.finally(&finally_cb);
            finally_cb.forget();
            ret = Some(new_promise.unchecked_into());
            break;
        } else {
            // Immediate value: restore suppression now.
            pop_subclass_dispatch();
            let is_undefined_or_null = ret_.is_undefined() || ret_.is_null();
            ret = Some(ret_);
            if !is_undefined_or_null {
                break;
            }
        }
    }
    Ok(ret.unwrap_or(JsValue::UNDEFINED))
}

// #[cfg(disable)]
pub fn handle_piston_function<const N: usize>(
    input: Option<&JsValue>,
    function_name: &str,
    overloaded_args: &RVec<&JsValue>,
    args: &[&JsValue; N],
    named_args: &JsValue,
) -> Result<JsValue, JsError> {
    // Resolve the function only from the overall module (global exports like opMethodAdd)
    let generic_function: Function = with_piston_web_module(|module| {
        let val = Reflect::get(module, &JsValue::from_str(function_name))
            .map_err(|e| e.into_js_error())?;
        val.dyn_into().map_err(|_| {
            JsError::new(&format!(
                "Target module does not have function {function_name}"
            ))
        })
    })?;

    // Combine self and args, if self exists
    let args: RVec<&JsValue> = input.into_iter().chain(args.iter().copied()).collect();

    // INTERESTINGLY: If self exists, we don't pull from overloaded_args, we just use self.

    // Create js_types from overloaded_args
    let js_types: RVec<JsValue> = overloaded_args
        .iter()
        .map(|arg| get_type_of_jsvalue(arg))
        .collect::<RVec<_>>();

    let mode_active = is_function_mode_active();

    let mut ret = None;

    if mode_active {
        ret = dispatch_on_mode(&args, named_args, &js_types, &generic_function)?;
    }

    if ret.is_none()
        && let Ok(curr_ret) = dispatch_on_subclass(
            &args,
            named_args,
            overloaded_args,
            &js_types,
            &generic_function,
        )
    {
        ret = Some(curr_ret);
    }

    // In PyTorch, they .release() the return value—we probably want to make sure we're not
    // refcounting it, whatever that most closely looks like here.
    ret.ok_or_else(|| JsError::new("Piston function handling failed; function modes or subclasses were registered but none returned a value."))
}

/// Placeholder for a future function-mode object.
#[derive(Clone, Debug)]
pub struct FunctionMode {
    js_mode_obj: JsValue,
}

thread_local! {
    static FUNCTION_MODE_STACK: RefCell<Vec<FunctionMode>> = const { RefCell::new(Vec::new()) };
}

/// Push a new function mode onto the per-thread stack.
pub fn push_function_mode(mode: FunctionMode) {
    FUNCTION_MODE_STACK.with(|stack| stack.borrow_mut().push(mode));
}

/// Pop the current function mode from the stack.
pub fn pop_function_mode() -> Option<FunctionMode> {
    FUNCTION_MODE_STACK.with(|stack| stack.borrow_mut().pop())
}

/// Push a new JS function mode object onto the stack.
#[wasm_bindgen(js_name = _pushFunctionMode)]
pub fn push_function_mode_js(mode_obj: &JsValue) -> Result<(), JsValue> {
    push_function_mode(FunctionMode {
        js_mode_obj: mode_obj.clone(),
    });
    Ok(())
}

/// Pop the current function mode and return it back to JS.
#[wasm_bindgen(js_name = _popFunctionMode)]
pub fn pop_function_mode_js() -> JsValue {
    pop_function_mode()
        .map(|fm| fm.js_mode_obj)
        .unwrap_or_else(JsValue::undefined)
}

/// Is there at least one active function mode on this thread?
pub fn is_function_mode_active() -> bool {
    FUNCTION_MODE_STACK.with(|stack| !stack.borrow().is_empty())
}

// Subclass-dispatch suppression: prevent re-entrant subclass redispatch when a subclass
// implementation calls the original function. We model this as a depth counter.
thread_local! {
    static SUBCLASS_DISPATCH_DEPTH: RefCell<u32> = const { RefCell::new(0) };
}

fn push_subclass_dispatch() {
    SUBCLASS_DISPATCH_DEPTH.with(|d| *d.borrow_mut() = d.borrow().saturating_add(1));
}

fn pop_subclass_dispatch() {
    SUBCLASS_DISPATCH_DEPTH.with(|d| {
        let mut b = d.borrow_mut();
        if *b > 0 {
            *b -= 1;
        }
    });
}

fn is_subclass_dispatch_active() -> bool {
    SUBCLASS_DISPATCH_DEPTH.with(|d| *d.borrow() > 0)
}

/// RAII guard that temporarily removes the active function mode so that nested
/// calls don’t immediately recurse back into the same handler.  The mode is
/// restored when the guard goes out of scope.
pub struct StashFunctionModeGuard {
    current_mode: Option<FunctionMode>,
}

impl StashFunctionModeGuard {
    pub fn new() -> Self {
        let saved_mode = pop_function_mode();
        Self {
            current_mode: saved_mode,
        }
    }
}

impl Drop for StashFunctionModeGuard {
    fn drop(&mut self) {
        if let Some(mode) = self.current_mode.take() {
            push_function_mode(mode);
        }
    }
}

/// MarkStep mode: simpler analog to FunctionMode for intercepting mark_step.
#[derive(Clone, Debug)]
pub struct MarkStepMode {
    js_mode_obj: JsValue,
}

thread_local! {
    static MARK_STEP_MODE_STACK: RefCell<Vec<MarkStepMode>> = const { RefCell::new(Vec::new()) };
}

pub fn push_mark_step_mode(mode: MarkStepMode) {
    MARK_STEP_MODE_STACK.with(|stack| stack.borrow_mut().push(mode));
}

pub fn pop_mark_step_mode() -> Option<MarkStepMode> {
    MARK_STEP_MODE_STACK.with(|stack| stack.borrow_mut().pop())
}

#[wasm_bindgen(js_name = _pushMarkStepMode)]
pub fn push_mark_step_mode_js(mode_obj: &JsValue) -> Result<(), JsValue> {
    if mode_obj.is_undefined() || mode_obj.is_null() {
        return Ok(());
    }
    push_mark_step_mode(MarkStepMode {
        js_mode_obj: mode_obj.clone(),
    });
    Ok(())
}

#[wasm_bindgen(js_name = _popMarkStepMode)]
pub fn pop_mark_step_mode_js() -> JsValue {
    pop_mark_step_mode()
        .map(|m| m.js_mode_obj)
        .unwrap_or_else(JsValue::undefined)
}

pub fn is_mark_step_mode_active() -> bool {
    MARK_STEP_MODE_STACK.with(|stack| !stack.borrow().is_empty())
}

/// Handle mark_step dispatch via the active MarkStepMode if present.
/// If the mode handler returns undefined/null, fall back to the default implementation.
pub async fn handle_mark_step(device: &crate::device::JsDevice) -> Result<(), JsError> {
    if !is_mark_step_mode_active() {
        // No mode active: run default behavior
        device
            .inner
            .try_gpu()
            .unwrap()
            .mark_step()
            .await
            .map_err(|e| JsError::new(&e.to_string()))?;
        return Ok(());
    }

    let mut _mode_guard = StashMarkStepModeGuard::new();
    let mode_obj = &_mode_guard
        .current_mode
        .as_ref()
        .expect("No MarkStep mode object")
        .js_mode_obj;

    let piston_mark_step_func: Function =
        Reflect::get(mode_obj, &JsValue::from_str("_pistonMarkStep"))
            .and_then(|v: JsValue| v.dyn_into())
            .map_err(|_| JsError::new("Mode object does not have a _pistonMarkStep method"))?;

    // Create original function closure returning a Promise<void>
    let device_inner = std::rc::Rc::new(device.inner.clone());
    let device_inner_for_closure = device_inner.clone();
    let original_fn_closure = Closure::wrap(Box::new(move || -> JsValue {
        let device_inner = device_inner_for_closure.clone();
        let fut = async move {
            device_inner
                .try_gpu()
                .unwrap()
                .mark_step()
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            Ok(JsValue::UNDEFINED)
        };
        future_to_promise(fut).unchecked_into::<JsValue>()
    }) as Box<dyn Fn() -> JsValue>);
    // Cast to Function for JS .apply. Keep the Closure alive in scope until we finish.
    let original_fn: Function = original_fn_closure
        .as_ref()
        .unchecked_ref::<Function>()
        .clone();

    let ret = piston_mark_step_func
        .apply(mode_obj, &Array::of1(&original_fn))
        .map_err(|e| e.into_js_error())?;

    if let Some(promise) = ret.dyn_ref::<Promise>() {
        // Keep mode disabled until promise settles, then restore it
        let saved_mode = _mode_guard.current_mode.take();
        JsFuture::from(promise.clone())
            .await
            .map_err(|e| e.into_js_error())?;
        if let Some(mode) = saved_mode {
            push_mark_step_mode(mode);
        }
        // original_fn_closure drops here
        return Ok(());
    }

    // Immediate value: if undefined/null, fall back to default implementation
    if ret.is_undefined() || ret.is_null() {
        device
            .inner
            .try_gpu()
            .unwrap()
            .mark_step()
            .await
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(())
    } else {
        // Non-null immediate indicates the handler completed the step
        Ok(())
    }
}

/// RAII guard for temporarily stashing the current MarkStepMode
pub struct StashMarkStepModeGuard {
    current_mode: Option<MarkStepMode>,
}

impl StashMarkStepModeGuard {
    pub fn new() -> Self {
        let saved_mode = pop_mark_step_mode();
        Self {
            current_mode: saved_mode,
        }
    }
}

impl Drop for StashMarkStepModeGuard {
    fn drop(&mut self) {
        if let Some(mode) = self.current_mode.take() {
            push_mark_step_mode(mode);
        }
    }
}
