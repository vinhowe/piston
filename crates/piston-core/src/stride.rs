use std::ops::{Index, IndexMut, RangeFrom, RangeTo};
use std::slice::Iter;

use crate::{rvec, RVec, Shape};
use encase::impl_wrapper;

#[derive(Clone, PartialEq, Eq, Default, Hash)]
pub struct Stride(RVec<isize>);

impl_wrapper!(Stride; using);

impl Stride {
    pub fn to_vec(&self) -> Vec<isize> {
        self.0.to_vec()
    }

    pub fn iter(&self) -> Iter<'_, isize> {
        self.0.iter()
    }

    pub fn transpose(&mut self) {
        let rank = self.0.len();
        if rank < 2 {
            return;
        }
        self.0.swap(rank - 2, rank - 1);
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    pub fn as_slice(&self) -> &[isize] {
        &self.0
    }
}

impl std::fmt::Debug for Stride {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut shape = format!("[{}", self.0.first().unwrap_or(&0));
        for dim in self.0.iter().skip(1) {
            shape.push_str(&format!("x{dim}"));
        }
        write!(f, "{shape}]")
    }
}

impl core::ops::Deref for Stride {
    type Target = [isize];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Index<usize> for Stride {
    type Output = isize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for Stride {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl Index<RangeFrom<usize>> for Stride {
    type Output = [isize];

    fn index(&self, index: RangeFrom<usize>) -> &Self::Output {
        &self.0[index]
    }
}

impl Index<RangeTo<usize>> for Stride {
    type Output = [isize];

    fn index(&self, index: RangeTo<usize>) -> &Self::Output {
        &self.0[index]
    }
}

impl From<&Shape> for Stride {
    fn from(shape: &Shape) -> Self {
        let mut strides = rvec![];
        let mut stride = 1;
        for size in shape.inner().iter().rev() {
            strides.push(stride);
            stride *= *size as isize;
        }
        strides.reverse();
        Self(strides)
    }
}

impl From<Vec<isize>> for Stride {
    fn from(stride: Vec<isize>) -> Self {
        Self(stride.into())
    }
}

impl From<&[isize]> for Stride {
    fn from(stride: &[isize]) -> Self {
        Self(stride.into())
    }
}

impl From<&Stride> for [u32; 3] {
    fn from(stride: &Stride) -> Self {
        assert!(stride.0.len() <= 3);
        let mut array = [0; 3];
        for (i, &stride) in stride.0.iter().enumerate() {
            array[i] = stride as u32;
        }
        array
    }
}

impl From<&Stride> for glam::UVec3 {
    fn from(stride: &Stride) -> Self {
        let array: [u32; 3] = stride.into();
        glam::UVec3::from(array)
    }
}

impl From<&Stride> for [u32; 4] {
    fn from(stride: &Stride) -> Self {
        assert!(stride.0.len() <= 4);
        let mut array = [0; 4];
        for (i, &stride) in stride.0.iter().enumerate() {
            array[i] = stride as u32;
        }
        array
    }
}

impl From<&Stride> for [usize; 4] {
    fn from(stride: &Stride) -> Self {
        assert!(stride.0.len() <= 4);
        let mut array = [0; 4];
        for (i, &stride) in stride.0.iter().enumerate() {
            array[i] = stride as usize;
        }
        array
    }
}

impl From<&Stride> for glam::UVec4 {
    fn from(stride: &Stride) -> Self {
        let array: [u32; 4] = stride.into();
        glam::UVec4::from(array)
    }
}

impl From<Stride> for glam::IVec3 {
    fn from(stride: Stride) -> Self {
        (&stride).into()
    }
}

impl From<&Stride> for glam::IVec3 {
    fn from(stride: &Stride) -> Self {
        glam::IVec3::new(stride.0[0] as _, stride.0[1] as _, stride.0[2] as _)
    }
}

#[cfg(test)]
mod tests {
    use crate::shape;

    #[test]
    fn test_stride() {
        use super::*;
        let shape = shape![2, 3, 4];
        let stride = Stride::from(&shape);
        assert_eq!(stride.to_vec(), vec![12, 4, 1]);
    }
}
