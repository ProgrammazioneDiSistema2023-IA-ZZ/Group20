use super::{ConvAttributes, ClipAttributes, GatherAttributes, UnsqueezeAttributes, ConcatAttributes, GemmAttributes};

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
    BatchNorm,
    ReLU,
}