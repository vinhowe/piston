use half::f16;
use std::fmt::{Debug, Display};

/// Supported data types in WGSL.
///
/// This can be mapped to and from the Piston DType.
pub trait WgslDType: Debug + Display + Default + Copy + num_traits::Num + num_traits::Zero {
    const DT: &'static str;
    const MIN: Self;

    fn render(&self) -> String;
}
//RENDER IS CONFUSING HERE

impl WgslDType for f32 {
    const DT: &'static str = "f32";
    const MIN: Self = -3e10; //ranges for wgsl and rust are diff

    fn render(&self) -> String {
        format!("{self}f")
    }
}

impl WgslDType for f16 {
    const DT: &'static str = "f16";
    const MIN: Self = f16::MIN;

    fn render(&self) -> String {
        format!("{self}h")
    }
}

impl WgslDType for i32 {
    const DT: &'static str = "i32";
    const MIN: Self = i32::MIN;

    fn render(&self) -> String {
        format!("{self}i")
    }
}

impl WgslDType for u32 {
    const DT: &'static str = "u32";
    const MIN: Self = u32::MIN;

    fn render(&self) -> String {
        format!("{self}u")
    }
}
