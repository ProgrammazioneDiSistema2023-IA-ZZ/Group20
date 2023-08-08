///
/// # Operator
///
/// This module define the ONNX Operator structures.
///
/// Currently it only defines the subset of Operator used to execute the selected neural networks
///
mod convolution;
mod minor;

pub use convolution::*;
pub use minor::*;

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum Operator {
    None,
    Convolution(ConvInputs, ConvAttributes),
    Clip(ClipAttributes),
    Add,
    Shape,
    Gather(GatherInputs, GatherAttributes),
    Unsqueeze(UnsqueezeAttributes),
    Concat(ConcatAttributes),
    GlobalAveragePool,
    Reshape(ReshapeInputs),
    Gemm(GemmInputs, GemmAttributes),
    MaxPool(MaxPoolAttributes),
    BatchNorm(BatchNormInputs, BatchNormAttributes),
    ReLU,
}


impl Operator {
    pub fn name(&self) -> String {
        match self {
            Operator::Convolution(_, _) => "Convolution".to_string(),
            Operator::Clip(_) => "Clip".to_string(),
            Operator::Add => "Add".to_string(),
            Operator::Shape => "Shape".to_string(),
            Operator::Gather(_, _) => "Gather".to_string(),
            Operator::Unsqueeze(_) => "Unsqueeze".to_string(),
            Operator::Concat(_) => "Concat".to_string(),
            Operator::GlobalAveragePool => "GlobalAveragePool".to_string(),
            Operator::Reshape(_) => "Reshape".to_string(),
            Operator::Gemm(_, _) => "Gemm".to_string(),
            Operator::MaxPool(_) => "MaxPool".to_string(),
            Operator::BatchNorm(_, _) => "BatchNorm".to_string(),
            Operator::ReLU => "ReLU".to_string(),
            _ => "None".to_string(),
        }
    }
}
