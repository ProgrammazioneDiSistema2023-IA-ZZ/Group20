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
