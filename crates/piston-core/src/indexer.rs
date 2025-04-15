// Adapted from Candle: https://github.com/huggingface/candle/blob/main/candle-core/src/indexer.rs
use crate::OpTensor;
use anyhow::Error;
use std::ops::{
    Bound, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

impl OpTensor {
    fn index(&self, indexers: &[TensorIndexer]) -> Result<Self, Error> {
        let mut x = self.clone();
        let dims = self.shape().as_slice();
        let mut current_dim = 0;
        for (i, indexer) in indexers.iter().enumerate() {
            x = match indexer {
                TensorIndexer::Select(n) => x.narrow(current_dim, *n, 1)?.squeeze(current_dim)?,
                TensorIndexer::Narrow(left_bound, right_bound) => {
                    let start = match left_bound {
                        Bound::Included(n) => *n,
                        Bound::Excluded(n) => *n + 1,
                        Bound::Unbounded => 0,
                    };
                    let stop = match right_bound {
                        Bound::Included(n) => *n + 1,
                        Bound::Excluded(n) => *n,
                        Bound::Unbounded => dims[i],
                    };
                    let out = x.narrow(current_dim, start, stop.saturating_sub(start))?;
                    current_dim += 1;
                    out
                }
                TensorIndexer::IndexSelect(indexes) => {
                    if indexes.rank() != 1 {
                        anyhow::bail!("multi-dimensional tensor indexing is not supported")
                    }
                    if indexes.device() != x.device() {
                        anyhow::bail!("indexing device mismatch: index tensor is on {:?} but input tensor is on {:?}", indexes.device(), x.device())
                    }
                    let out = x.index_select(indexes.clone(), current_dim)?;
                    current_dim += 1;
                    out
                }
                TensorIndexer::Err(e) => anyhow::bail!("indexing error {e:?}"),
            };
        }
        Ok(x)
    }
}

#[derive(Debug)]
/// Generic structure used to index a slice of the tensor
pub enum TensorIndexer {
    /// This selects the elements for which an index has some specific value.
    Select(usize),
    /// This is a regular slice, purely indexing a chunk of the tensor
    Narrow(Bound<usize>, Bound<usize>),
    /// Indexing via a 1d tensor
    IndexSelect(OpTensor),
    Err(Error),
}

impl From<usize> for TensorIndexer {
    fn from(index: usize) -> Self {
        TensorIndexer::Select(index)
    }
}

impl From<&OpTensor> for TensorIndexer {
    fn from(tensor: &OpTensor) -> Self {
        TensorIndexer::IndexSelect(tensor.clone())
    }
}

trait RB: RangeBounds<usize> {}
impl RB for Range<usize> {}
impl RB for RangeFrom<usize> {}
impl RB for RangeFull {}
impl RB for RangeInclusive<usize> {}
impl RB for RangeTo<usize> {}
impl RB for RangeToInclusive<usize> {}

impl<T: RB> From<T> for TensorIndexer {
    fn from(range: T) -> Self {
        use std::ops::Bound::*;
        let start = match range.start_bound() {
            Included(idx) => Included(*idx),
            Excluded(idx) => Excluded(*idx),
            Unbounded => Unbounded,
        };
        let end = match range.end_bound() {
            Included(idx) => Included(*idx),
            Excluded(idx) => Excluded(*idx),
            Unbounded => Unbounded,
        };
        TensorIndexer::Narrow(start, end)
    }
}

/// Trait used to implement multiple signatures for ease of use of the slicing
/// of a tensor
pub trait IndexOp<T> {
    /// Returns a slicing iterator which are the chunks of data necessary to
    /// reconstruct the desired tensor.
    fn i(&self, index: T) -> Result<OpTensor, Error>;
}

impl<T> IndexOp<T> for OpTensor
where
    T: Into<TensorIndexer>,
{
    fn i(&self, index: T) -> Result<OpTensor, Error> {
        self.index(&[index.into()])
    }
}

impl<A> IndexOp<(A,)> for OpTensor
where
    A: Into<TensorIndexer>,
{
    fn i(&self, (a,): (A,)) -> Result<OpTensor, Error> {
        self.index(&[a.into()])
    }
}

macro_rules! index_op_tuple {
    ($($t:ident),+) => {
        #[allow(non_snake_case)]
        impl<$($t),*> IndexOp<($($t,)*)> for OpTensor
        where
            $($t: Into<TensorIndexer>,)*
        {
            fn i(&self, ($($t,)*): ($($t,)*)) -> Result<OpTensor, Error> {
                self.index(&[$($t.into(),)*])
            }
        }
    };
}

index_op_tuple!(A, B, C);
index_op_tuple!(A, B, C, D);
index_op_tuple!(A, B, C, D, E);
index_op_tuple!(A, B, C, D, E, F);
index_op_tuple!(A, B, C, D, E, F, G);
