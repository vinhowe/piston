// Taken from candle

use core::panic;

// Variables are wrappers around tensors that can be modified, they are typically used for holding
// weights and being modified by gradient descent.
// We do not expose a public way to create variables as this would break the invariant that the
// tensor within a variable is actually with `is_variable` set to `true`.
use crate::{Device, GPUBuffer, Inner, LazyOp, Shape, Storage, StorageView, Tensor, TensorDType};
use anyhow::Result;

/// A variable is a wrapper around a tensor, however variables can have their content modified
/// whereas tensors are immutable.
#[derive(Clone, Debug)]
pub struct Var(Tensor);

impl std::ops::Deref for Var {
    type Target = Inner;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl Var {
    pub fn zeros<T: TensorDType>(shape: &Shape, device: &Device) -> Self {
        let inner = Tensor::zeros_impl::<T>(shape, device, true);
        Self(inner)
    }

    pub fn ones<T: TensorDType>(shape: &Shape, device: &Device) -> Self {
        let inner = Tensor::ones_impl::<T>(shape, device, true);
        Self(inner)
    }

    // Convert a tensor to a variable, if the tensor is already a variable then it is returned as is.
    pub fn from_tensor(t: &Tensor) -> Result<Self> {
        if t.is_variable() {
            Ok(Self(t.clone()))
        } else {
            let inner = t.make_var()?;
            Ok(Self(inner))
        }
    }

    #[cfg(feature = "rand")]
    pub fn randint<T: TensorDType + rand_distr::uniform::SampleUniform + PartialOrd>(
        low: T,
        high: T,
        shape: Shape,
        device: Device,
    ) -> Self {
        let inner = Tensor::randint_impl(low, high, shape, device, true);
        Self(inner)
    }

    #[cfg(feature = "rand")]
    pub fn rand<T: TensorDType + num_traits::Float>(
        lo: f32,
        up: f32,
        shape: Shape,
        device: Device,
    ) -> Self {
        let inner = Tensor::rand_impl::<T>(lo, up, shape, device, true);
        Self(inner)
    }

    #[cfg(feature = "rand")]
    pub fn randn<T: TensorDType + num_traits::Float>(
        mean: f32,
        std: f32,
        shape: Shape,
        device: Device,
    ) -> Self {
        let inner = Tensor::randn_impl::<T>(mean, std, shape, device, true);
        Self(inner)
    }

    /// Creates a new tensor on the specified device using the content and shape of the input.
    /// This is similar to `new` but the resulting tensor is a variable.
    pub fn new(op: LazyOp, meta: StorageView, storage: Option<Storage>, device: Device) -> Self {
        let inner = Tensor::new_impl(op, meta, storage, device, true);
        Self(inner)
    }

    pub fn from_data<T: TensorDType, U: AsRef<[T]>>(data: U, shape: Shape, device: Device) -> Self {
        let inner = Tensor::from_data_impl::<T, U>(data, shape, device, true);
        Self(inner)
    }

    pub fn as_detached_tensor(&self) -> Tensor {
        self.0.detach()
    }

    pub fn as_tensor(&self) -> &Tensor {
        &self.0
    }

    /// Consumes this `Var` and return the underlying tensor.
    pub fn into_inner(self) -> Tensor {
        self.0
    }

    /// Sets the content of the inner tensor, this does not require a mutable reference as inner
    /// mutability is used.
    pub fn set(&self, src: Tensor) -> Result<()> {
        if self.as_tensor().same_storage(&src) {
            panic!("cannot set a variable to a tensor that is derived from its value");
        }
        if self.as_tensor().shape() != src.shape() {
            panic!(
                "shape mismatch: {:?} != {:?} (target id: {:?}, source id: {:?})",
                self.as_tensor().shape(),
                src.shape(),
                self.as_tensor().id(),
                src.id()
            );
        }
        let dst = self.as_tensor();
        dst.update_storage(Storage::GPU(GPUBuffer {
            inner: src.storage().as_ref().unwrap().try_gpu()?.inner.clone(),
            alignment: dst.dt().size_of(),
            cpu_size: Some(dst.num_bytes()),
        }));
        Ok(())
    }
}
