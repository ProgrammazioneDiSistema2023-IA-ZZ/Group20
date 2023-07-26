///
/// # ONNX Format
///
/// This module contains the ONNX format related code.
///
/// The ONNX format is defined in the [ONNX specification](https://github.com/onnx/onnx/blob/main/docs/IR.md).
///
/// The .proto files are provided by the [official ONNX repository](https://github.com/onnx/onn).
///
mod onnx_model;

pub use onnx_model::*;
