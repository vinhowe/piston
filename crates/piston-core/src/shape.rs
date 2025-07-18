use crate::{rvec, shape, RVec, Stride};
use anyhow::Result;
use encase::impl_wrapper;
use smallvec::ToSmallVec;
use std::{
    ops::{RangeFrom, RangeTo},
    slice::Iter,
};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg_attr(target_arch = "wasm32", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
#[derive(Clone, PartialEq, Eq, Hash, Default)]
pub struct Shape(RVec<usize>);

impl_wrapper!(Shape; using);

impl Shape {
    pub fn scalar() -> Self {
        // TODO(vinhowe): Move to an empty scalar shape, once I have time to debug
        Self(rvec![1])
    }

    pub fn new(shape: RVec<usize>) -> Self {
        Self(shape)
    }

    pub fn inner(&self) -> &RVec<usize> {
        &self.0
    }

    pub fn get(&self, index: usize) -> Option<&usize> {
        self.0.get(index)
    }

    pub fn insert(&mut self, index: usize, dim: usize) {
        self.0.insert(index, dim);
    }

    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }

    pub fn to_vec(&self) -> Vec<usize> {
        self.0.to_vec()
    }

    pub fn iter(&self) -> Iter<'_, usize> {
        self.0.iter()
    }

    pub fn reverse(&mut self) {
        self.0.reverse();
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dim(&self) -> usize {
        self.len()
    }

    pub fn push(&mut self, dim: usize) {
        self.0.push(dim);
    }

    pub fn remove(&mut self, index: usize) -> usize {
        self.0.remove(index)
    }

    pub fn is_scalar(&self) -> bool {
        self.0.iter().all(|&x| x == 1)
    }

    pub fn is_vector(&self) -> bool {
        let mut shape = self.clone();
        shape.squeeze(None);
        shape.dim() <= 1
    }

    #[inline]
    pub fn left_pad_to(&mut self, scalar: usize, rank: usize) {
        while self.0.len() < rank {
            self.0.insert(0, scalar);
        }
    }

    #[inline]
    pub fn right_pad_to(&mut self, scalar: usize, rank: usize) {
        while self.0.len() < rank {
            self.0.push(scalar);
        }
    }

    #[inline]
    pub fn promote(shape: Shape, rank: usize) -> Shape {
        let mut shape = shape;
        shape.left_pad_to(1, rank);
        shape
    }

    #[inline]
    pub fn squeeze(&mut self, dims: Option<RVec<usize>>) {
        if let Some(dims) = dims {
            // Create a sorted copy of the dims in descending order
            // This way, removing elements won't affect the indices of elements we haven't processed
            // yet
            let mut sorted_dims = dims.to_vec();
            sorted_dims.sort_by(|a, b| b.cmp(a));

            for dim in sorted_dims {
                if dim < self.0.len() {
                    self.0.remove(dim);
                }
            }
        } else {
            self.0.retain(|x| *x != 1);
        }
    }

    #[inline]
    pub fn unsqueeze(&mut self, usize: usize) {
        self.0.insert(usize, 1);
    }

    pub fn drain<R>(&mut self, range: R) -> smallvec::Drain<'_, [usize; 4]>
    where
        R: std::ops::RangeBounds<usize>,
    {
        self.0.drain(range)
    }

    pub fn slice<R>(&self, range: R) -> Self
    where
        R: std::ops::RangeBounds<usize> + std::slice::SliceIndex<[usize], Output = [usize]>,
    {
        Shape(self.0[range].to_vec().into())
    }

    pub fn as_slice(&self) -> &[usize] {
        &self.0
    }

    pub fn multi_broadcast(shapes: &[&Shape]) -> Option<Shape> {
        let max_rank = shapes.iter().map(|shape| shape.dim()).max()?;
        let mut shape: Shape = shape![];
        for i in 0..max_rank {
            let mut current_dim_size = 1;
            for shape in shapes {
                let len = shape.dim();
                let dim = if i < len { &shape[len - i - 1] } else { &1 };
                if dim != &1 {
                    if current_dim_size != 1 && dim != &current_dim_size {
                        return None;
                    }
                    current_dim_size = *dim;
                }
            }
            shape.0.insert(0, current_dim_size)
        }
        Some(shape)
    }

    /// Returns true if the stride is C contiguous (aka row major).
    pub fn is_contiguous(&self, stride: &Stride) -> bool {
        let stride_vec = stride.to_vec();
        if self.0.len() != stride_vec.len() {
            return false;
        }
        let mut acc = 1;
        for (&stride, &dim) in stride_vec.iter().zip(self.0.iter()).rev() {
            if dim > 1 && stride != acc {
                return false;
            }
            acc *= dim as isize;
        }
        true
    }

    pub fn transpose(&mut self) {
        let rank = self.dim();
        if rank < 2 {
            return;
        }
        self.0.swap(rank - 2, rank - 1);
    }
}

impl core::ops::Deref for Shape {
    type Target = [usize];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::fmt::Debug for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut shape = format!("[{}", self.0.first().unwrap_or(&0));
        for dim in self.0.iter().skip(1) {
            shape.push_str(&format!("x{dim}"));
        }
        write!(f, "{shape}]")
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

impl std::ops::Index<usize> for Shape {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl std::ops::IndexMut<usize> for Shape {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl std::ops::Index<RangeFrom<usize>> for Shape {
    type Output = [usize];

    fn index(&self, index: RangeFrom<usize>) -> &Self::Output {
        &self.0[index]
    }
}

impl std::ops::Index<RangeTo<usize>> for Shape {
    type Output = [usize];

    fn index(&self, index: RangeTo<usize>) -> &Self::Output {
        &self.0[index]
    }
}

impl<const C: usize> From<&[usize; C]> for Shape {
    fn from(dims: &[usize; C]) -> Self {
        Self(dims.to_smallvec())
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self(dims.into())
    }
}

impl From<&Shape> for Shape {
    fn from(shape: &Shape) -> Self {
        Self(shape.0.to_smallvec())
    }
}

impl From<()> for Shape {
    fn from(_: ()) -> Self {
        Self(rvec![])
    }
}

impl std::iter::Iterator for Shape {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.pop()
    }
}

impl std::iter::DoubleEndedIterator for Shape {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.pop()
    }
}

impl From<&Shape> for glam::UVec4 {
    fn from(shape: &Shape) -> Self {
        glam::UVec4::new(
            shape[0] as u32,
            shape[1] as u32,
            shape[2] as u32,
            shape[3] as u32,
        )
    }
}

impl From<Shape> for glam::UVec4 {
    fn from(shape: Shape) -> Self {
        (&shape).into()
    }
}

impl From<&Shape> for glam::IVec3 {
    fn from(shape: &Shape) -> Self {
        glam::IVec3::new(shape[0] as i32, shape[1] as i32, shape[2] as i32)
    }
}

impl From<Shape> for glam::IVec3 {
    fn from(shape: Shape) -> Self {
        (&shape).into()
    }
}

impl From<Shape> for RVec<usize> {
    fn from(shape: Shape) -> Self {
        shape.0
    }
}

impl From<usize> for Shape {
    fn from(d1: usize) -> Self {
        Self(rvec![d1])
    }
}

macro_rules! impl_try_into_arr_for_shape {
    ($($N:expr),*) => {
        $(
            impl TryInto<[usize; $N]> for &Shape {
                type Error = anyhow::Error;

                fn try_into(self) -> Result<[usize; $N], Self::Error> {
                    if self.0.len() == $N {
                        let mut arr = [0; $N];
                        for (i, &item) in self.0.iter().enumerate().take($N) {
                            arr[i] = item;
                        }
                        Ok(arr)
                    } else {
                        Err(anyhow::anyhow!("Shape has length {} but expected {}", self.0.len(), $N))
                    }
                }
            }
        )*
    };
}

impl_try_into_arr_for_shape!(1, 2, 3, 4);

macro_rules! impl_from_tuple {
    ($tuple:ty, $($index:tt),+) => {
        impl From<$tuple> for Shape {
            fn from(d: $tuple) -> Self {
                Self(rvec![$(d.$index,)+])
            }
        }
    }
}

impl_from_tuple!((usize,), 0);
impl_from_tuple!((usize, usize), 0, 1);
impl_from_tuple!((usize, usize, usize), 0, 1, 2);
impl_from_tuple!((usize, usize, usize, usize), 0, 1, 2, 3);
impl_from_tuple!((usize, usize, usize, usize, usize), 0, 1, 2, 3, 4);
impl_from_tuple!((usize, usize, usize, usize, usize, usize), 0, 1, 2, 3, 4, 5);

impl From<RVec<usize>> for Shape {
    fn from(dims: RVec<usize>) -> Self {
        Self(dims)
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self(dims.into())
    }
}

macro_rules! extract_dims {
    ($fn_name:ident, $cnt:tt, $dims:expr, $out_type:ty) => {
        pub fn $fn_name(dims: &[usize]) -> Result<$out_type> {
            if dims.len() != $cnt {
                Err(anyhow::anyhow!(
                    "Unexpected number of dimensions: expected {}, got {}, shape: {:?}",
                    $cnt,
                    dims.len(),
                    Shape::from(dims)
                ))
            } else {
                Ok($dims(dims))
            }
        }

        impl Shape {
            pub fn $fn_name(&self) -> Result<$out_type> {
                $fn_name(self.0.as_slice())
            }
        }

        impl crate::OpTensor {
            pub fn $fn_name(&self) -> Result<$out_type> {
                self.shape().$fn_name()
            }
        }

        impl std::convert::TryInto<$out_type> for Shape {
            type Error = anyhow::Error;
            fn try_into(self) -> anyhow::Result<$out_type> {
                self.$fn_name()
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use crate::Shape;
    use proptest::prelude::*;
    use std::ops::RangeInclusive;

    impl Arbitrary for Shape {
        type Parameters = Vec<RangeInclusive<usize>>;
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
            args.prop_map(Into::<Shape>::into).boxed()
        }
    }

    impl Shape {
        pub fn as_torch(&self) -> String {
            let mut shape = format!("({}", self[0]);
            for dim in self.iter().skip(1) {
                shape.push_str(&format!(", {dim}"));
            }
            shape.push(')');
            shape
        }
    }
}

pub trait Dim {
    fn to_index(&self, shape: &Shape, op: &'static str) -> Result<usize>;
    fn to_index_plus_one(&self, shape: &Shape, op: &'static str) -> Result<usize>;
}

impl Dim for usize {
    fn to_index(&self, shape: &Shape, op: &'static str) -> Result<usize> {
        let rank = shape.dim();
        if *self >= rank {
            Err(anyhow::anyhow!("Dimension out of range for op: {}", op))
        } else {
            Ok(*self)
        }
    }

    fn to_index_plus_one(&self, shape: &Shape, op: &'static str) -> Result<usize> {
        let rank = shape.dim();
        if *self > rank {
            Err(anyhow::anyhow!("Dimension out of range for op: {}", op))
        } else {
            Ok(*self)
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum D {
    Minus1,
    Minus2,
    Minus(usize),
}

impl D {
    fn out_of_range(&self, _: &Shape, op: &'static str) -> anyhow::Error {
        let dim = match self {
            Self::Minus1 => -1,
            Self::Minus2 => -2,
            Self::Minus(u) => -(*u as i32),
        };
        // TODO(vinhowe): include shape
        anyhow::anyhow!("Dimension {} out of range for op: {}", dim, op)
    }
}

impl Dim for D {
    fn to_index(&self, shape: &Shape, op: &'static str) -> Result<usize> {
        let rank = shape.dim();
        match self {
            Self::Minus1 if rank >= 1 => Ok(rank - 1),
            Self::Minus2 if rank >= 2 => Ok(rank - 2),
            Self::Minus(u) if *u > 0 && rank >= *u => Ok(rank - *u),
            _ => Err(self.out_of_range(shape, op)),
        }
    }

    fn to_index_plus_one(&self, shape: &Shape, op: &'static str) -> Result<usize> {
        let rank = shape.dim();
        match self {
            Self::Minus1 => Ok(rank),
            Self::Minus2 if rank >= 1 => Ok(rank - 1),
            Self::Minus(u) if *u > 0 && rank + 1 >= *u => Ok(rank + 1 - *u),
            _ => Err(self.out_of_range(shape, op)),
        }
    }
}

pub trait Dims: Sized {
    fn to_indexes_internal(self, shape: &Shape, op: &'static str) -> Result<RVec<usize>>;

    fn to_indexes(self, shape: &Shape, op: &'static str) -> Result<RVec<usize>> {
        let dims = self.to_indexes_internal(shape, op)?;
        for (i, &dim) in dims.iter().enumerate() {
            if dims[..i].contains(&dim) {
                anyhow::bail!("Duplicate dimension index: {}", dim)
            }
            if dim >= shape.dim() {
                anyhow::bail!("Dimension out of range: {}", dim)
            }
        }
        Ok(dims)
    }
}

impl<T: Dim + Sized> Dims for T {
    fn to_indexes_internal(self, shape: &Shape, op: &'static str) -> Result<RVec<usize>> {
        let dim = self.to_index(shape, op)?;
        Ok(rvec![dim])
    }
}

impl Dims for RVec<usize> {
    fn to_indexes_internal(self, _: &Shape, _: &'static str) -> Result<RVec<usize>> {
        Ok(self)
    }
}

impl<const N: usize> Dims for [usize; N] {
    fn to_indexes_internal(self, _: &Shape, _: &'static str) -> Result<RVec<usize>> {
        Ok(self.to_vec().into())
    }
}

impl Dims for &[usize] {
    fn to_indexes_internal(self, _: &Shape, _: &'static str) -> Result<RVec<usize>> {
        Ok(self.to_vec().into())
    }
}

impl Dims for () {
    fn to_indexes_internal(self, _: &Shape, _: &'static str) -> Result<RVec<usize>> {
        Ok(rvec![])
    }
}

impl<D: Dim> Dims for (D,) {
    fn to_indexes_internal(self, shape: &Shape, op: &'static str) -> Result<RVec<usize>> {
        let dim = self.0.to_index(shape, op)?;
        Ok(rvec![dim])
    }
}

impl<D1: Dim, D2: Dim> Dims for (D1, D2) {
    fn to_indexes_internal(self, shape: &Shape, op: &'static str) -> Result<RVec<usize>> {
        let d0 = self.0.to_index(shape, op)?;
        let d1 = self.1.to_index(shape, op)?;
        Ok(rvec![d0, d1])
    }
}

impl<D1: Dim, D2: Dim, D3: Dim> Dims for (D1, D2, D3) {
    fn to_indexes_internal(self, shape: &Shape, op: &'static str) -> Result<RVec<usize>> {
        let d0 = self.0.to_index(shape, op)?;
        let d1 = self.1.to_index(shape, op)?;
        let d2 = self.2.to_index(shape, op)?;
        Ok(rvec![d0, d1, d2])
    }
}

impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim> Dims for (D1, D2, D3, D4) {
    fn to_indexes_internal(self, shape: &Shape, op: &'static str) -> Result<RVec<usize>> {
        let d0 = self.0.to_index(shape, op)?;
        let d1 = self.1.to_index(shape, op)?;
        let d2 = self.2.to_index(shape, op)?;
        let d3 = self.3.to_index(shape, op)?;
        Ok(rvec![d0, d1, d2, d3])
    }
}

impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim, D5: Dim> Dims for (D1, D2, D3, D4, D5) {
    fn to_indexes_internal(self, shape: &Shape, op: &'static str) -> Result<RVec<usize>> {
        let d0 = self.0.to_index(shape, op)?;
        let d1 = self.1.to_index(shape, op)?;
        let d2 = self.2.to_index(shape, op)?;
        let d3 = self.3.to_index(shape, op)?;
        let d4 = self.4.to_index(shape, op)?;
        Ok(rvec![d0, d1, d2, d3, d4])
    }
}

impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim, D5: Dim, D6: Dim> Dims for (D1, D2, D3, D4, D5, D6) {
    fn to_indexes_internal(self, shape: &Shape, op: &'static str) -> Result<RVec<usize>> {
        let d0 = self.0.to_index(shape, op)?;
        let d1 = self.1.to_index(shape, op)?;
        let d2 = self.2.to_index(shape, op)?;
        let d3 = self.3.to_index(shape, op)?;
        let d4 = self.4.to_index(shape, op)?;
        let d5 = self.5.to_index(shape, op)?;
        Ok(rvec![d0, d1, d2, d3, d4, d5])
    }
}

extract_dims!(dims0, 0, |_: &[usize]| (), ());
extract_dims!(dims1, 1, |d: &[usize]| d[0], usize);
extract_dims!(dims2, 2, |d: &[usize]| (d[0], d[1]), (usize, usize));
extract_dims!(
    dims3,
    3,
    |d: &[usize]| (d[0], d[1], d[2]),
    (usize, usize, usize)
);
extract_dims!(
    dims4,
    4,
    |d: &[usize]| (d[0], d[1], d[2], d[3]),
    (usize, usize, usize, usize)
);
extract_dims!(
    dims5,
    5,
    |d: &[usize]| (d[0], d[1], d[2], d[3], d[4]),
    (usize, usize, usize, usize, usize)
);

pub trait ShapeWithOneHole {
    fn into_shape(self, el_count: usize) -> Result<Shape>;
}

impl<S: Into<Shape>> ShapeWithOneHole for S {
    fn into_shape(self, _el_count: usize) -> Result<Shape> {
        Ok(self.into())
    }
}

impl ShapeWithOneHole for ((),) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        Ok(el_count.into())
    }
}

pub fn hole_size(el_count: usize, prod_d: usize, s: &dyn std::fmt::Debug) -> Result<usize> {
    if prod_d == 0 {
        anyhow::bail!("cannot reshape tensor of {el_count} elements to {s:?}")
    }
    if !el_count.is_multiple_of(prod_d) {
        anyhow::bail!("cannot reshape tensor with {el_count} elements to {s:?}")
    }
    Ok(el_count / prod_d)
}

impl ShapeWithOneHole for ((), usize) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let ((), d1) = self;
        Ok((hole_size(el_count, d1, &self)?, d1).into())
    }
}

impl ShapeWithOneHole for (usize, ()) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let (d1, ()) = self;
        Ok((d1, hole_size(el_count, d1, &self)?).into())
    }
}

impl ShapeWithOneHole for ((), usize, usize) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let ((), d1, d2) = self;
        Ok((hole_size(el_count, d1 * d2, &self)?, d1, d2).into())
    }
}

impl ShapeWithOneHole for (usize, (), usize) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let (d1, (), d2) = self;
        Ok((d1, hole_size(el_count, d1 * d2, &self)?, d2).into())
    }
}

impl ShapeWithOneHole for (usize, usize, ()) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let (d1, d2, ()) = self;
        Ok((d1, d2, hole_size(el_count, d1 * d2, &self)?).into())
    }
}

impl ShapeWithOneHole for ((), usize, usize, usize) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let ((), d1, d2, d3) = self;
        let d = hole_size(el_count, d1 * d2 * d3, &self)?;
        Ok((d, d1, d2, d3).into())
    }
}

impl ShapeWithOneHole for (usize, (), usize, usize) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let (d1, (), d2, d3) = self;
        let d = hole_size(el_count, d1 * d2 * d3, &self)?;
        Ok((d1, d, d2, d3).into())
    }
}

impl ShapeWithOneHole for (usize, usize, (), usize) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let (d1, d2, (), d3) = self;
        let d = hole_size(el_count, d1 * d2 * d3, &self)?;
        Ok((d1, d2, d, d3).into())
    }
}

impl ShapeWithOneHole for (usize, usize, usize, ()) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let (d1, d2, d3, ()) = self;
        let d = hole_size(el_count, d1 * d2 * d3, &self)?;
        Ok((d1, d2, d3, d).into())
    }
}

impl ShapeWithOneHole for ((), usize, usize, usize, usize) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let ((), d1, d2, d3, d4) = self;
        let d = hole_size(el_count, d1 * d2 * d3 * d4, &self)?;
        Ok((d, d1, d2, d3, d4).into())
    }
}

impl ShapeWithOneHole for (usize, (), usize, usize, usize) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let (d1, (), d2, d3, d4) = self;
        let d = hole_size(el_count, d1 * d2 * d3 * d4, &self)?;
        Ok((d1, d, d2, d3, d4).into())
    }
}

impl ShapeWithOneHole for (usize, usize, (), usize, usize) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let (d1, d2, (), d3, d4) = self;
        let d = hole_size(el_count, d1 * d2 * d3 * d4, &self)?;
        Ok((d1, d2, d, d3, d4).into())
    }
}

impl ShapeWithOneHole for (usize, usize, usize, (), usize) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let (d1, d2, d3, (), d4) = self;
        let d = hole_size(el_count, d1 * d2 * d3 * d4, &self)?;
        Ok((d1, d2, d3, d, d4).into())
    }
}

impl ShapeWithOneHole for (usize, usize, usize, usize, ()) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let (d1, d2, d3, d4, ()) = self;
        let d = hole_size(el_count, d1 * d2 * d3 * d4, &self)?;
        Ok((d1, d2, d3, d4, d).into())
    }
}
