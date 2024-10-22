//! Contains the Graph utilities.
//!
//! This module defines the way to map an ONNX model to an execution graph structure to infer a ONNX model.

/// Strategy to use when converting an ONNX model to an execution graph.
mod translator;
pub use translator::*;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum GraphError {
    #[error("ConversionError: {0}")]
    ConversionError(String),

    #[error("{operand} for Operator {operator} of type {operator_type}")]
    MissingOperand {
        operand: String,
        operator: String,
        operator_type: String,
    },

    #[error("Unsupported operator: {0}")]
    UnsupportedOperator(String),

    #[error("for the child {child_name}")]
    ParentNotFound { child_name: String },

    #[error("Unexpected error")]
    UnexpectedError,

    #[error("DeconstructError: {0}")]
    DeconstructError(String),

    #[error("InputNodeParsingError")]
    InputNodeParsingError,

    #[error("OutputNodeParsingError")]
    OutputNodeParsingError,

    #[error("Unknwon operation error")]
    Unknown,
}
