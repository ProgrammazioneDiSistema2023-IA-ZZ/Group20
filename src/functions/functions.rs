use super::{ConvAttributes, ClipAttributes, GatherAttributes, UnsqueezeAttributes, ConcatAttributes, GemmAttributes, BatchNormAttributes, MaxPoolAttributes};

#[allow(dead_code)]
enum Function {
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