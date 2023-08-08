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

use thiserror::Error;

#[allow(dead_code)]
pub enum Operators {
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

#[derive(Error, Debug)]
pub enum OperationError {
    #[error("wrong tensor shape: expected `{0}`, found `{1}`")]
    WrongShape(String, String),
    #[error("wrong tensor dimensionality: expected `{0}`, found `{1}`")]
    WrongDim(usize, usize),
    #[error("input tensor shapes are supposed to match but they don't: `{0}` != `{1}`")]
    UnmatchingShape(String, String),
    #[error("the specified operator(s) are not supported")]
    UnsupportedOperator,
    #[error("the specified operator(s) are not valid")]
    InvalidOperator,
    #[error("unknwon operation error")]
    Unknown,
}
