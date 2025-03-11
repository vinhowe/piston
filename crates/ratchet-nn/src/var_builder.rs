//! A `VarBuilder` is used to retrieve variables used by a model. These variables can either come
//! from a pre-trained checkpoint, e.g. using `VarBuilder::from_mmaped_safetensors`, or initialized
//! for training, e.g. using `VarBuilder::from_varmap`.
use crate::VarMap;
use async_trait::async_trait;
use ratchet::HashMap;
use ratchet::{DType, Device, OperationError, Shape, Tensor};
use std::sync::Arc;

use maybe_async::maybe_async;

// thiserror error for Tensor
#[derive(thiserror::Error, Debug)]
pub enum VarBuilderError {
    #[error("Cannot find tensor {path}")]
    CannotFindTensor { path: String },
    #[error(transparent)]
    OperationError(#[from] OperationError),
    #[error("shape mismatch on {path}: {shape:?} <> {tensor_shape:?}")]
    ShapeMismatch {
        path: String,
        shape: Shape,
        tensor_shape: Shape,
    },
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// A structure used to retrieve variables, these variables can either come from storage or be
/// generated via some form of initialization.
///
/// The way to retrieve variables is defined in the backend embedded in the `VarBuilder`.
pub struct VarBuilderArgs<'a, B: Backend> {
    data: Arc<TensorData<B>>,
    path: Vec<String>,
    _phantom: std::marker::PhantomData<&'a B>,
}

impl<'a, B: Backend> Clone for VarBuilderArgs<'a, B> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            path: self.path.clone(),
            _phantom: self._phantom,
        }
    }
}

/// A simple `VarBuilder`, this is less generic than `VarBuilderArgs` but should cover most common
/// use cases.
pub type VarBuilder<'a> = VarBuilderArgs<'a, Box<dyn SimpleBackend + 'a>>;

struct TensorData<B: Backend> {
    backend: B,
    pub dtype: DType,
    pub device: Device,
}

/// A trait that defines how tensor data is retrieved.
///
/// Typically this would use disk storage in some specific format, or random initialization.
/// Note that there is a specialized version of this trait (`SimpleBackend`) that can be used most
/// of the time. The main restriction is that it doesn't allow for specific args (besides
/// initialization hints).
// #[maybe_async(AFIT)]
#[maybe_async]
pub trait Backend {
    type Hints: Default;

    /// Retrieve a tensor with some target shape.
    async fn get(
        &self,
        s: Shape,
        name: &str,
        h: Self::Hints,
        dev: Device,
    ) -> anyhow::Result<Tensor, VarBuilderError>;

    fn contains_tensor(&self, name: &str) -> bool;
}

#[maybe_async]
pub trait SimpleBackend: Send + Sync {
    /// Retrieve a tensor based on a target name and shape.
    async fn get(
        &self,
        s: Shape,
        name: &str,
        h: crate::Init,
        dev: Device,
    ) -> anyhow::Result<Tensor, VarBuilderError>;

    fn contains_tensor(&self, name: &str) -> bool;
}

#[maybe_async]
impl<'a> Backend for Box<dyn SimpleBackend + 'a> {
    // impl<'a> Backend for Box<dyn SimpleBackend + 'a> {
    type Hints = crate::Init;

    async fn get(
        &self,
        s: Shape,
        name: &str,
        h: Self::Hints,
        dev: Device,
    ) -> anyhow::Result<Tensor, VarBuilderError> {
        self.as_ref().get(s, name, h, dev).await
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.as_ref().contains_tensor(name)
    }
}

#[maybe_async]
impl<'a, B: Backend> VarBuilderArgs<'a, B> {
    pub fn new_with_args(backend: B, dtype: DType, dev: Device) -> Self {
        let data = TensorData {
            backend,
            dtype,
            device: dev.clone(),
        };
        Self {
            data: Arc::new(data),
            path: vec![],
            _phantom: std::marker::PhantomData,
        }
    }

    /// Returns the prefix of the `VarBuilder`.
    pub fn prefix(&self) -> String {
        self.path.join(".")
    }

    /// Returns a new `VarBuilder` using the root path.
    pub fn root(&self) -> Self {
        Self {
            data: self.data.clone(),
            path: vec![],
            _phantom: std::marker::PhantomData,
        }
    }

    /// Returns a new `VarBuilder` with the prefix set to `prefix`.
    pub fn set_prefix(&self, prefix: impl ToString) -> Self {
        Self {
            data: self.data.clone(),
            path: vec![prefix.to_string()],
            _phantom: std::marker::PhantomData,
        }
    }

    /// Return a new `VarBuilder` adding `s` to the current prefix. This can be think of as `cd`
    /// into a directory.
    pub fn push_prefix<S: ToString>(&self, s: S) -> Self {
        let mut path = self.path.clone();
        path.push(s.to_string());
        Self {
            data: self.data.clone(),
            path,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Short alias for `push_prefix`.
    pub fn pp<S: ToString>(&self, s: S) -> Self {
        self.push_prefix(s)
    }

    /// The device used by default.
    pub fn device(&self) -> &Device {
        &self.data.device
    }

    /// The dtype used by default.
    pub fn dtype(&self) -> DType {
        self.data.dtype
    }

    fn path(&self, tensor_name: &str) -> String {
        if self.path.is_empty() {
            tensor_name.to_string()
        } else {
            [&self.path.join("."), tensor_name].join(".")
        }
    }

    /// This returns true only if a tensor with the passed in name is available. E.g. when passed
    /// `a`, true is returned if `prefix.a` exists but false is returned if only `prefix.a.b`
    /// exists.
    pub fn contains_tensor(&self, tensor_name: &str) -> bool {
        let path = self.path(tensor_name);
        self.data.backend.contains_tensor(&path)
    }

    /// Retrieve the tensor associated with the given name at the current path.
    pub async fn get_with_hints(
        &self,
        s: Shape,
        name: &str,
        hints: B::Hints,
    ) -> anyhow::Result<Tensor, VarBuilderError> {
        self.get_with_hints_dtype(s, name, hints).await
    }

    // #[cfg(not(target_arch = "wasm32"))]
    // /// Retrieve the tensor associated with the given name at the current path.
    // pub fn get_with_hints(
    //     &self,
    //     s: Shape,
    //     name: &str,
    //     hints: B::Hints,
    // ) -> anyhow::Result<Tensor, VarBuilderError> {
    //     self.get_with_hints_dtype(s, name, hints)
    // }

    /// Retrieve the tensor associated with the given name at the current path.
    pub async fn get(&self, s: Shape, name: &str) -> anyhow::Result<Tensor, VarBuilderError> {
        self.get_with_hints(s, name, Default::default()).await
    }

    // #[cfg(not(target_arch = "wasm32"))]
    // /// Retrieve the tensor associated with the given name & dtype at the current path.
    // pub fn get_with_hints_dtype<S: Into<Shape>>(
    //     &self,
    //     s: S,
    //     name: &str,
    //     hints: B::Hints,
    // ) -> anyhow::Result<Tensor, VarBuilderError> {
    //     let path = self.path(name);
    //     self.data
    //         .backend
    //         .get(s.into(), &path, hints, self.data.device.clone())
    // }

    // #[cfg(target_arch = "wasm32")]
    /// Retrieve the tensor associated with the given name & dtype at the current path.
    pub async fn get_with_hints_dtype<S: Into<Shape>>(
        &self,
        s: S,
        name: &str,
        hints: B::Hints,
    ) -> anyhow::Result<Tensor, VarBuilderError> {
        let path = self.path(name);
        self.data
            .backend
            .get(s.into(), &path, hints, self.data.device.clone())
            .await
    }
}

struct Zeros;

#[maybe_async]
impl SimpleBackend for Zeros {
    async fn get(
        &self,
        s: Shape,
        _: &str,
        _: crate::Init,
        dev: Device,
    ) -> anyhow::Result<Tensor, VarBuilderError> {
        Ok(Tensor::zeros::<f32>(&s, &dev)?)
    }

    fn contains_tensor(&self, _name: &str) -> bool {
        true
    }
}

#[maybe_async]
impl SimpleBackend for HashMap<String, Tensor> {
    async fn get(
        &self,
        s: Shape,
        name: &str,
        _: crate::Init,
        dev: Device,
    ) -> anyhow::Result<Tensor, VarBuilderError> {
        let tensor = self
            .get(name)
            .ok_or_else(|| VarBuilderError::CannotFindTensor {
                path: name.to_string(),
            })?
            .clone();
        if tensor.shape() != &s {
            return Err(VarBuilderError::ShapeMismatch {
                path: name.to_string(),
                shape: s,
                tensor_shape: tensor.shape().clone(),
            });
        }
        Ok(tensor.to(&dev).await.unwrap())
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.contains_key(name)
    }
}

#[async_trait]
impl SimpleBackend for VarMap {
    #[cfg(target_arch = "wasm32")]
    async fn get(
        &self,
        s: Shape,
        name: &str,
        h: crate::Init,
        dev: Device,
    ) -> anyhow::Result<Tensor, VarBuilderError> {
        Ok(VarMap::get(self, s, name, h, dev).unwrap())
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn get(
        &self,
        s: Shape,
        name: &str,
        h: crate::Init,
        dev: Device,
    ) -> anyhow::Result<Tensor, VarBuilderError> {
        Ok(VarMap::get(self, s, name, h, dev).unwrap())
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.data().lock().unwrap().contains_key(name)
    }
}

impl<'a> VarBuilder<'a> {
    pub fn from_backend(
        backend: Box<dyn SimpleBackend + 'a>,
        dtype: DType,
        device: Device,
    ) -> Self {
        let data = TensorData {
            backend,
            dtype,
            device,
        };
        Self {
            data: Arc::new(data),
            path: vec![],
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn zeros(dtype: DType, dev: &Device) -> Self {
        Self::from_backend(Box::new(Zeros), dtype, dev.clone())
    }

    pub fn from_tensors(ts: HashMap<String, Tensor>, dtype: DType, dev: &Device) -> Self {
        Self::from_backend(Box::new(ts), dtype, dev.clone())
    }

    pub fn from_varmap(varmap: &VarMap, dtype: DType, dev: &Device) -> Self {
        Self::from_backend(Box::new(varmap.clone()), dtype, dev.clone())
    }

    pub fn rename_f<F: Fn(&str) -> String + Sync + Send + 'static>(self, f: F) -> Self {
        let f: Box<dyn Fn(&str) -> String + Sync + Send + 'static> = Box::new(f);
        self.rename(f)
    }

    pub fn rename<R: Renamer + Send + Sync + 'a>(self, renamer: R) -> Self {
        let dtype = self.dtype();
        let device = self.device().clone();
        let path = self.path.clone();
        let backend = Rename::new(self, renamer);
        let backend: Box<dyn SimpleBackend + 'a> = Box::new(backend);
        let data = TensorData {
            backend,
            dtype,
            device,
        };
        Self {
            data: Arc::new(data),
            path,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// This traits specifies a way to rename the queried names into names that are stored in an inner
/// VarBuilder.
pub trait Renamer {
    /// This is applied to the name obtained by a name call and the resulting name is passed to the
    /// inner VarBuilder.
    fn rename(&self, v: &str) -> std::borrow::Cow<'_, str>;
}

pub struct Rename<'a, R: Renamer> {
    inner: VarBuilder<'a>,
    renamer: R,
}

#[maybe_async]
impl<'a, R: Renamer + Send + Sync> SimpleBackend for Rename<'a, R> {
    async fn get(
        &self,
        s: Shape,
        name: &str,
        h: crate::Init,
        dev: Device,
    ) -> anyhow::Result<Tensor, VarBuilderError> {
        let name = self.renamer.rename(name);
        Ok(self
            .inner
            .get_with_hints_dtype(s, &name, h)
            .await?
            .to(&dev)
            .await
            .unwrap())
    }

    fn contains_tensor(&self, name: &str) -> bool {
        let name = self.renamer.rename(name);
        self.inner.contains_tensor(&name)
    }
}

impl<'a, R: Renamer> Rename<'a, R> {
    pub fn new(inner: VarBuilder<'a>, renamer: R) -> Self {
        Self { inner, renamer }
    }
}

impl Renamer for Box<dyn Fn(&str) -> String + Sync + Send> {
    fn rename(&self, v: &str) -> std::borrow::Cow<'_, str> {
        std::borrow::Cow::Owned(self(v))
    }
}
