///
/// # Operators
///
/// This module define the ONNX operators structures.
///
/// Currently it only defines the subset of operators used to execute the selected neural networks
///
mod convolution;
mod minor;

pub use convolution::*;
pub use minor::*;

use ndarray::{Array0, Array1, Array2, Array4};

#[allow(dead_code)]
pub enum Operator {
    Convolution(ConvAttributes),
    Clip(ClipAttributes),
    Add,
    Shape,
    Gather(GatherAttributes),
    Unsqueeze(UnsqueezeAttributes),
    Concat(ConcatAttributes),
    GlobalAveragePool,
    Reshape,
    Gemm(GemmAttributes),
    MaxPool(MaxPoolAttributes),
    BatchNorm(BatchNormAttributes),
    ReLU,
}

pub enum Tensor{
    Array0Float32(Array0<f32>),
    Array1Float32(Array1<f32>),
    Array2Float32(Array2<f32>),
    Array4Float32(Array4<f32>),
}

