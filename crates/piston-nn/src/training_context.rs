use std::cell::RefCell;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ModuleMode {
    Train,
    Eval,
}

thread_local! {
    static CURRENT_MODE: RefCell<ModuleMode> = const { RefCell::new(ModuleMode::Train) };
}

// Functions to get/set the mode
pub fn set_train_mode() {
    CURRENT_MODE.with(|mode| *mode.borrow_mut() = ModuleMode::Train);
}

pub fn set_eval_mode() {
    CURRENT_MODE.with(|mode| *mode.borrow_mut() = ModuleMode::Eval);
}

pub fn current_module_mode() -> ModuleMode {
    CURRENT_MODE.with(|mode| *mode.borrow())
}

/// Context manager for scoped mode changes
pub struct ModuleModeGuard {
    previous: ModuleMode,
}

impl ModuleModeGuard {
    pub fn new(mode: ModuleMode) -> Self {
        let previous = current_module_mode();
        CURRENT_MODE.with(|m| *m.borrow_mut() = mode);
        Self { previous }
    }
}

impl Drop for ModuleModeGuard {
    fn drop(&mut self) {
        CURRENT_MODE.with(|mode| *mode.borrow_mut() = self.previous);
    }
}
