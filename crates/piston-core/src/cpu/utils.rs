use crate::{CPUBuffer, Shape, Storage, Stride, OpTensor};
use bytemuck::NoUninit;
use std::ops::Range;

pub fn cpu_store_result<T: NoUninit>(dst: &OpTensor, data: &[T]) {
    dst.update_storage(Storage::CPU(CPUBuffer::from_slice(data, dst.shape())));
}

#[derive(Clone)]
pub enum TensorIterator<'a> {
    Contiguous(Range<usize>),
    Strided(StridedIterator<'a>),
}

impl<'a> TensorIterator<'a> {
    pub fn new(shape: &'a Shape, stride: &'a Stride, offset: usize) -> Self {
        let mut block_size: usize = 1;
        let mut contiguous_dims: usize = 0;
        for (&stride, &dim) in stride.iter().zip(shape.iter()).rev() {
            if stride as usize != block_size {
                break;
            }
            block_size *= dim;
            contiguous_dims += 1;
        }
        let index_dims = shape.dim() - contiguous_dims;
        if index_dims == 0 {
            Self::Contiguous(offset..block_size)
        } else {
            Self::Strided(StridedIterator::new(shape, stride, offset, block_size))
        }
    }
}

impl Iterator for TensorIterator<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Contiguous(r) => r.next(),
            Self::Strided(s) => s.next(),
        }
    }
}

#[derive(Clone)]
pub struct StridedIterator<'a> {
    shape: &'a [usize],
    stride: &'a [isize],
    next_index: Option<usize>,
    multi_index: Vec<usize>,
    block_size: usize,
    block_step: usize,
}

impl<'a> StridedIterator<'a> {
    pub fn new(
        shape: &'a [usize],
        stride: &'a [isize],
        start_offset: usize,
        block_len: usize,
    ) -> Self {
        Self {
            shape,
            stride,
            next_index: if shape.iter().product::<usize>() == 0 {
                None
            } else {
                Some(start_offset)
            },
            multi_index: vec![0; shape.len()],
            block_size: block_len,
            block_step: 0,
        }
    }
}

impl Iterator for StridedIterator<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let storage_index = self.next_index?;

        if self.block_size > 1 {
            if self.block_step < self.block_size {
                self.block_step += 1;
                return Some(storage_index + self.block_step - 1);
            } else {
                self.block_step = 0;
            }
        }

        let mut updated = false;
        let mut next_storage_index = storage_index;
        for ((multi_i, max_i), stride_i) in self
            .multi_index
            .iter_mut()
            .zip(self.shape.iter())
            .zip(self.stride.iter())
            .rev()
        {
            let next_i = *multi_i + 1;
            if next_i < *max_i {
                *multi_i = next_i;
                updated = true;
                next_storage_index += *stride_i as usize;
                break;
            } else {
                next_storage_index -= *multi_i * *stride_i as usize;
                *multi_i = 0
            }
        }
        self.next_index = if updated {
            Some(next_storage_index)
        } else {
            None
        };
        Some(storage_index)
    }
}

impl<'a> From<(&'a Shape, &'a Stride)> for StridedIterator<'a> {
    fn from((shape, stride): (&'a Shape, &'a Stride)) -> Self {
        StridedIterator::new(shape.as_slice(), stride.as_slice(), 0, 1)
    }
}

impl<'a> From<(&'a Shape, &'a Stride, usize)> for StridedIterator<'a> {
    fn from((shape, stride, offset): (&'a Shape, &'a Stride, usize)) -> Self {
        StridedIterator::new(shape.as_slice(), stride.as_slice(), offset, 1)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_strategy::proptest;

    use crate::{shape, Shape, Stride};

    use super::TensorIterator;

    #[derive(Debug)]
    struct IterProblem {
        shape: Shape,
        offset: usize,
    }

    impl Arbitrary for IterProblem {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
            let ranges = vec![1..=2, 1..=4, 1..=256, 1..=256];
            Shape::arbitrary_with(ranges)
                .prop_flat_map(|shape| (Just(shape.clone()), 0..shape.numel()))
                .prop_map(|(shape, offset)| IterProblem { shape, offset })
                .boxed()
        }
    }

    #[proptest(cases = 16)]
    fn test_tensor_iter_contiguous(prob: IterProblem) {
        let shape = prob.shape;
        let stride = Stride::from(&shape);
        let offset = prob.offset;

        let iter = TensorIterator::new(&shape, &stride, offset);
        assert!(matches!(iter, TensorIterator::Contiguous(_)));

        match iter {
            TensorIterator::Contiguous(r) => assert_eq!(r, offset..shape.numel()),
            _ => unreachable!(),
        }
    }

    #[proptest(cases = 16)]
    fn test_tensor_iter_strided(prob: IterProblem) {
        let mut shape = prob.shape;
        let mut stride = Stride::from(&shape);
        stride.transpose();
        shape.transpose();
        let offset = prob.offset;

        let iter = TensorIterator::new(&shape, &stride, offset);
        assert!(matches!(iter, TensorIterator::Strided(_)));

        match iter {
            TensorIterator::Strided(strided_iter) => {
                let mut indices: Vec<usize> = strided_iter.collect();
                assert_eq!(indices.len(), shape.numel());
                let contiguous: Vec<usize> = (offset..shape.numel() + offset).collect();
                assert_ne!(indices, contiguous);
                indices.sort();
                assert_eq!(indices, contiguous);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_tensor_iter_strided_sanity() {
        let mut shape = shape!(2, 4, 3);
        let mut stride = Stride::from(&shape);
        stride.transpose();
        shape.transpose();
        let offset = 2;

        let iter = TensorIterator::new(&shape, &stride, offset);
        let actual: Vec<usize> = iter.collect();
        let expected = vec![
            2, 5, 8, 11, 3, 6, 9, 12, 4, 7, 10, 13, 14, 17, 20, 23, 15, 18, 21, 24, 16, 19, 22, 25,
        ];
        assert_eq!(actual, expected);
    }
}
