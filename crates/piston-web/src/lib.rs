#![cfg(target_arch = "wasm32")]
mod device;
mod dtype;
mod error;
mod function;
mod js_util;
mod model;
mod serialization;
mod shape;
mod tensor;

#[cfg(test)]
mod test_utils;

use std::cell::RefCell;

use wasm_bindgen::prelude::*;

// Fix: https://github.com/rustwasm/wasm-bindgen/issues/4446#issuecomment-2729543167
mod wasm_ctor_workaround {
    unsafe extern "C" {
        pub(super) fn __wasm_call_ctors();
    }
}

thread_local! {
    static PISTON_WEB_MOD: RefCell<Option<JsValue>> = const { RefCell::new(None) };
}

#[wasm_bindgen(start)]
pub fn start() {
    // This is important as long as we use inventory, which presumably uses ctors
    unsafe { wasm_ctor_workaround::__wasm_call_ctors() };

    console_error_panic_hook::set_once();
    let logger = fern::Dispatch::new()
        .format(|out, _message, record| {
            out.finish(format_args!(
                "[WASM {file}:{line}] {text}",
                file = record.file().unwrap_or_else(|| record.target()),
                line = record
                    .line()
                    .map_or_else(|| "[Unknown]".to_string(), |line| line.to_string()),
                text = record.args(),
            ))
        })
        .level_for("tokenizers", log::LevelFilter::Off)
        .level(log::LevelFilter::Info)
        .chain(fern::Output::call(console_log::log))
        .apply();

    match logger {
        Ok(_) => log::info!("Logging initialized."),
        Err(error) => eprintln!("Error initializing logging: {error:?}"),
    }
}

/// Called once from JS right after you finish instantiating the wasm.
#[wasm_bindgen(js_name = "_setPistonWebModule")]
pub fn set_piston_web_module(module: &JsValue) {
    PISTON_WEB_MOD.with(|cell| {
        cell.borrow_mut().replace(module.clone());
    });
}

pub(crate) fn with_piston_web_module<F, R>(f: F) -> Result<R, JsError>
where
    F: FnOnce(&JsValue) -> Result<R, JsError>,
{
    PISTON_WEB_MOD.with(|cell| {
        let module = cell.borrow();
        let Some(module) = module.as_ref() else {
            return Err(JsError::new("PistonWeb module not loaded"));
        };
        f(module)
    })
}
