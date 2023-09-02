//! # Encoding/Decoding structures for the ONNX format
//!
//! This module contains the ONNX format related code.
//!
//! The ONNX format is defined in the [ONNX specification](https://github.com/onnx/onnx/blob/main/docs/IR.md).
//!
//! The .proto files are provided by the [official ONNX repository](https://github.com/onnx/onn).

/// ONNX Model structures
///
/// This module contains the generated ONNX protobuf structures.
/// They can be used to encode/decode ONNX models, thanks to the `prost` crate.
mod onnx_model;

pub use onnx_model::*;
