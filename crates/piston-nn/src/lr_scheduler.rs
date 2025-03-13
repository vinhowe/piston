use anyhow::Result;
use maybe_async::maybe_async;
use piston::{Device, GradStore};

use crate::Optimizer;

/// A trait that provides common functionalities for learning rate schedulers.
#[maybe_async]
pub trait LRScheduler<O: Optimizer + Send + Sync>: Send + Sync {
    /// Returns a mutable reference to the underlying optimizer.
    fn optimizer_mut(&mut self) -> &mut O;

    /// Returns the base learning rate.
    fn base_lr(&self) -> f64;

    /// Returns the current step count.
    fn step_count(&self) -> usize;

    /// Updates the internal step counter.
    fn set_step_count(&mut self, count: usize);

    /// Computes the new learning rate for the current step.
    fn compute_lr(&self) -> f64;

    /// Advances the scheduler by one step.
    async fn step(&mut self, grads: &GradStore, device: &Device) -> Result<()> {
        let lr = self.compute_lr();
        self.optimizer_mut().set_learning_rate(lr);
        self.optimizer_mut().step(grads, device).await?;
        self.set_step_count(self.step_count() + 1);
        Ok(())
    }

    /// Returns the current learning rate.
    fn get_lr(&self) -> f64 {
        self.compute_lr()
    }
}

/// A helper struct to hold common state shared by schedulers.
pub struct LRSchedulerCore<O: Optimizer + Send + Sync> {
    pub optimizer: O,
    pub base_lr: f64,
    pub step_count: usize,
}

impl<O: Optimizer + Send + Sync> LRSchedulerCore<O> {
    pub fn new(optimizer: O) -> Self {
        let base_lr = optimizer.learning_rate();
        Self {
            optimizer,
            base_lr,
            step_count: 0,
        }
    }
}

/// A scheduler that keeps the learning rate reduced by a `factor` for a certain
/// number of steps (`total_iters`) and then returns it to the original `base_lr`.
///
/// Similar to PyTorch's `ConstantLR` behavior.
pub struct ConstantLR<O: Optimizer + Send + Sync> {
    inner: LRSchedulerCore<O>,
    factor: f64,
    total_iters: usize,
}

impl<O: Optimizer + Send + Sync> ConstantLR<O> {
    /// Create a new `ConstantLR` wrapping the given `optimizer`.
    ///
    /// * `factor` - multiplied with the base_lr for the first `total_iters` steps.
    /// * `total_iters` - number of steps for which the factor is applied.
    pub fn new(optimizer: O, factor: f64, total_iters: usize) -> Self {
        Self {
            inner: LRSchedulerCore::new(optimizer),
            factor,
            total_iters,
        }
    }
}

impl<O: Optimizer + Send + Sync> LRScheduler<O> for ConstantLR<O> {
    fn optimizer_mut(&mut self) -> &mut O {
        &mut self.inner.optimizer
    }

    fn base_lr(&self) -> f64 {
        self.inner.base_lr
    }

    fn step_count(&self) -> usize {
        self.inner.step_count
    }

    fn set_step_count(&mut self, count: usize) {
        self.inner.step_count = count;
    }

    fn compute_lr(&self) -> f64 {
        if self.inner.step_count < self.total_iters {
            self.inner.base_lr * self.factor
        } else {
            self.inner.base_lr
        }
    }
}

/// A scheduler that linearly interpolates the learning rate from `start_factor`
/// to `end_factor` over `total_iters` steps.
///
/// Similar to PyTorch's `LinearLR`.
pub struct LinearLR<O: Optimizer + Send + Sync> {
    core: LRSchedulerCore<O>,
    start_factor: f64,
    end_factor: f64,
    total_iters: usize,
}

impl<O: Optimizer + Send + Sync> LinearLR<O> {
    /// Create a new `LinearLR` wrapping the given `optimizer`.
    ///
    /// * `start_factor` - multiplier at step 0 (relative to base_lr).
    /// * `end_factor` - multiplier at or after `total_iters` steps.
    /// * `total_iters` - number of steps over which we linearly transition.
    pub fn new(optimizer: O, start_factor: f64, end_factor: f64, total_iters: usize) -> Self {
        Self {
            core: LRSchedulerCore::new(optimizer),
            start_factor,
            end_factor,
            total_iters,
        }
    }
}

impl<O: Optimizer + Send + Sync> LRScheduler<O> for LinearLR<O> {
    fn optimizer_mut(&mut self) -> &mut O {
        &mut self.core.optimizer
    }

    fn base_lr(&self) -> f64 {
        self.core.base_lr
    }

    fn step_count(&self) -> usize {
        self.core.step_count
    }

    fn set_step_count(&mut self, count: usize) {
        self.core.step_count = count;
    }

    fn compute_lr(&self) -> f64 {
        if self.core.step_count == 0 {
            return self.core.base_lr * self.start_factor;
        }
        if self.core.step_count >= self.total_iters {
            return self.core.base_lr * self.end_factor;
        }
        let progress = self.core.step_count as f64 / self.total_iters as f64;
        let factor = self.start_factor + progress * (self.end_factor - self.start_factor);
        self.core.base_lr * factor
    }
}

/// A scheduler that anneals the learning rate following a cosine curve
/// from `base_lr` down to `eta_min` over `t_max` steps.
///
/// Similar to PyTorch's `CosineAnnealingLR` (without restarts).
pub struct CosineAnnealingLR<O: Optimizer + Send + Sync> {
    core: LRSchedulerCore<O>,
    eta_min: f64,
    t_max: usize,
}

impl<O: Optimizer + Send + Sync> CosineAnnealingLR<O> {
    /// Create a new `CosineAnnealingLR` wrapping the given `optimizer`.
    ///
    /// * `t_max` - number of steps over which the schedule is applied.
    /// * `eta_min` - minimum learning rate after the full schedule.
    pub fn new(optimizer: O, t_max: usize, eta_min: f64) -> Self {
        Self {
            core: LRSchedulerCore::new(optimizer),
            eta_min,
            t_max,
        }
    }
}

impl<O: Optimizer + Send + Sync> LRScheduler<O> for CosineAnnealingLR<O> {
    fn optimizer_mut(&mut self) -> &mut O {
        &mut self.core.optimizer
    }

    fn base_lr(&self) -> f64 {
        self.core.base_lr
    }

    fn step_count(&self) -> usize {
        self.core.step_count
    }

    fn set_step_count(&mut self, count: usize) {
        self.core.step_count = count;
    }

    fn compute_lr(&self) -> f64 {
        if self.t_max == 0 {
            return self.core.base_lr;
        }
        let step = self.core.step_count as f64;
        let t_max = self.t_max as f64;
        let cos_inner = std::f64::consts::PI * step / t_max;
        let cos_val = cos_inner.cos();
        self.eta_min + (self.core.base_lr - self.eta_min) * (1.0 + cos_val) / 2.0
    }
}
