use crate::HashMap;
use std::cell::RefCell;

// Each scope entry stores a name and the next id that was saved.
#[derive(Debug)]
struct ScopeEntry {
    name: String,
}

// The scope context holds a stack of scope entries and a counter for the next id.
#[derive(Debug)]
struct ScopeContext {
    scopes: Vec<ScopeEntry>,
    duplicate_counter: Vec<HashMap<String, usize>>,
}

// Create a thread-local ScopeContext using RefCell for interior mutability.
thread_local! {
    static SCOPE_CONTEXT: RefCell<ScopeContext> = RefCell::new(ScopeContext {
        scopes: Vec::new(),
        duplicate_counter: vec![HashMap::default()],
    });
}

/// Returns the current scope as a concatenated string of the scope names,
/// separated by slashes.
pub fn get_current_scope() -> String {
    SCOPE_CONTEXT.with(|ctx| {
        let ctx = ctx.borrow();
        ctx.scopes
            .iter()
            .map(|entry| entry.name.clone())
            .collect::<Vec<_>>()
            .join("/")
    })
}

/// Push a new scope with the given name.
/// This function formats the name with an id to ensure uniqueness.
fn push_scope(name: &str) {
    SCOPE_CONTEXT.with(|cell| {
        let mut ctx = cell.borrow_mut();
        let name_count = ctx
            .duplicate_counter
            .last_mut()
            .unwrap()
            .entry(name.to_string())
            .or_insert(0);
        let formatted_name = format!("{}.{}", name, *name_count);
        *name_count += 1;
        // Save the current next_id (incremented by one) in the entry.
        ctx.scopes.push(ScopeEntry {
            name: formatted_name,
        });
        ctx.duplicate_counter.push(HashMap::default());
    });
}

/// Pop the most recent scope off the stack.
/// Panics if there are no scopes to pop.
fn pop_scope() {
    SCOPE_CONTEXT.with(|ctx| {
        let mut ctx = ctx.borrow_mut();
        if !ctx.scopes.is_empty() {
            ctx.duplicate_counter.pop();
            ctx.scopes.pop();
        } else {
            panic!("Attempted to pop scope from an empty stack");
        }
    });
}

/// Resets the scope context, ensuring that there are no remaining scopes.
/// Panics if the scope stack is not empty.
pub fn reset_scope_context() {
    SCOPE_CONTEXT.with(|ctx| {
        let mut ctx = ctx.borrow_mut();
        if !ctx.scopes.is_empty() || ctx.duplicate_counter.len() > 1 {
            panic!(
                "Expecting scope to be empty but it is '{}'",
                get_current_scope()
            );
        }
        ctx.duplicate_counter.last_mut().unwrap().clear();
    });
}

/// A RAII-style scope pusher that pushes a scope on creation and pops it
/// when dropped.
pub struct ScopePusher;

impl ScopePusher {
    /// Create a new scope pusher that pushes the given scope.
    pub fn new(name: &str) -> Self {
        push_scope(name);
        ScopePusher
    }
}

// When a ScopePusher goes out of scope, its Drop implementation will automatically
// pop the scope.
impl Drop for ScopePusher {
    fn drop(&mut self) {
        pop_scope();
    }
}
