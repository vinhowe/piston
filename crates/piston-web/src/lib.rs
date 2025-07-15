#![cfg(target_arch = "wasm32")]
mod device;
mod dtype;
mod model;
mod serialization;
mod shape;
mod tensor;

#[cfg(test)]
mod test_utils;

use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn start() {
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
        Err(error) => eprintln!("Error initializing logging: {:?}", error),
    }
}
