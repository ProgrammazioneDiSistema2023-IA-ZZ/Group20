///
/// # Graph
///
/// This module defines the mapping between the ONNX standard and a Graph structure used to infer a ONNX model.
///

mod translator;
pub use translator::*;

mod ioreader;
pub use ioreader::*;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum GraphError {
    #[error("ConversionError: {0}")]
    ConversionError(String),

    #[error("{operand} for Operator {operator} of type {operator_type}")]
    MissingOperand{operand:String, operator:String, operator_type:String}, 

    #[error("")]
    UnsupportedOperator,

    #[error("for the child {child_name}")]
    ParentNotFound{child_name:String},

    #[error("")]
    UnexpectedError,

    #[error("DeconstructError: {0}")]
    DeconstructError(String),

    #[error("unknwon operation error")]
    Unknown,
}
