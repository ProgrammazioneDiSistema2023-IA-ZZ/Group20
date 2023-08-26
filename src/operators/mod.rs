//! # Operator
//!
//! This module define the ONNX Operators structures.
//!
//! Currently it only defines the subset of operators used to execute two chosen neural networks
//! (ResNet18 and MobileNetV2).
//!

mod attributes;
mod initializers;

pub use attributes::*;
pub use initializers::*;

use crate::tensor::TensorParametrizedShape;
use thiserror::Error;

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum Operator {
    InputFeed(TensorParametrizedShape),
    OutputCollector(TensorParametrizedShape),

    Convolution(ConvInits, ConvAttributes),
    Clip(ClipAttributes),
    Add,
    Shape,
    Gather(GatherInits, GatherAttributes),
    Unsqueeze(UnsqueezeAttributes),
    Concat(ConcatAttributes),
    GlobalAveragePool,
    Reshape(ReshapeInits),
    Gemm(GemmInits, GemmAttributes),
    MaxPool(MaxPoolAttributes),
    BatchNorm(BatchNormInits, BatchNormAttributes),
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

#[derive(Error, Debug)]
pub enum OperationError {
    #[error("Wrong tensor shape: expected `{0}`, found `{1}`")]
    WrongShape(String, String),
    #[error("Wrong tensor dimensionality: expected `{0}`, found `{1}`")]
    WrongDim(usize, usize),
    #[error("The operation requires coherent tensor shapes, but they don't match: `{0}` != `{1}`")]
    UnexpectedShape(String, String),
    #[error("The specified operator(s) are not supported")]
    UnsupportedOperator,
    #[error("The specified operator(s) are not valid")]
    InvalidOperator,
    #[error("The tensor `{1}` type is not valid for the operator `{0}`")]
    InvalidTensorType(String, String),
    #[error("Unknwon operation error")]
    Unknown,
}
