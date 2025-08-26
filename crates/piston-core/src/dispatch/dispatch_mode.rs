#![cfg(target_arch = "wasm32")]
use std::cell::RefCell;

use super::{DispatchKeySet, TLS_INCLUDE};

/// Placeholder for a future dispatch-mode object.  We haven't defined what a mode looks like yet,
/// but we still need *something* to push onto the stack so that we can exercise the control-flow
/// plumbing.
#[derive(Clone, Debug)]
pub struct DispatchMode;

// Per-thread stack that tracks which dispatch modes are currently active, most-recently-pushed at
// the back of the vector.  (Mirrors PyTorch's `TorchDispatchModeTLS`.)
thread_local! {
    static DISPATCH_MODE_STACK: RefCell<Vec<DispatchMode>> = const { RefCell::new(Vec::new()) };
}

/// Push a new mode onto the thread-local stack.
///
/// If this is the *first* mode (stack transition 0 → 1) we also enable the
/// `DispatchKeySet::DISPATCH_MODE` bit so that the rest of the dispatcher knows to route through
/// the (future) JS fallback.
pub fn push_mode(mode: DispatchMode) {
    DISPATCH_MODE_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        let was_empty = stack.is_empty();
        stack.push(mode);

        if was_empty {
            TLS_INCLUDE.with(|tls| {
                let new_include = tls.get() | DispatchKeySet::JAVASCRIPT;
                tls.set(new_include);
            });
        }
    });
}

/// Pop the current mode from the stack, returning it to the caller.
///
/// Whenever the stack becomes empty we clear the `DISPATCH_MODE` bit so that the fast path is
/// re-enabled.
pub fn pop_mode() -> Option<DispatchMode> {
    DISPATCH_MODE_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        let popped = stack.pop();

        if stack.is_empty() {
            TLS_INCLUDE.with(|tls| {
                let mut include = tls.get();
                include.remove(DispatchKeySet::JAVASCRIPT);
                tls.set(include);
            });
        }

        popped
    })
}

/// Current depth of the dispatch-mode stack.
pub fn stack_len() -> usize {
    DISPATCH_MODE_STACK.with(|stack| stack.borrow().len())
}

/// RAII guard that **temporarily pops** the active mode for the lifetime of the guard, pushing it
/// back when the guard is dropped.  This exactly mirrors PyTorch's `StashTorchDispatchModeGuard`
/// and prevents infinite recursion when the user-defined handler calls back into the dispatcher.
pub struct StashDispatchModeGuard {
    saved_mode: Option<DispatchMode>,
}

impl StashDispatchModeGuard {
    pub fn new() -> Self {
        Self {
            saved_mode: pop_mode(),
        }
    }
}

impl Drop for StashDispatchModeGuard {
    fn drop(&mut self) {
        if let Some(mode) = self.saved_mode.take() {
            push_mode(mode);
        }
    }
}
