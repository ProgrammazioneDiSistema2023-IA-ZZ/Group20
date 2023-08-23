//! # Validators
//! This are used for validating the inputs types of the operators.
//! The ONNX operators have a set of inputs that are used to configure the operator.
//! Their valid types are defined in the ONNX specification.
//!
//! Keep in mind that this just checks the types of the attributes according to the ONNX specs, not their values.
//! Each operator implementation (execution provider) is responsible for checking the values of the attributes and if they are applicable to the operator.

use crate::{
    operators::{ConvAttributes, ConvInits, Operator},
    tensor::TensorDataType,
};

use super::{BatchNormInits, GatherInits, GemmInits, ReshapeInits};

macro_rules! validate_tensor_data_type {
    ($tensor:expr, $($type:ident)|+) => {

        match $tensor.into() {
            $(
                TensorDataType::$type => Ok(()),
            )+
            _ => Err(format!(
                    "{} must be of type {}, found {:?}",
                    stringify!($tensor),
                    stringify!($($type)|+),
                    TensorDataType::from($tensor)
                ))
           ,
        }
    };
}

impl ConvInits {
    pub fn validate(&self) -> Result<(), String> {
        if let Some(bias) = &self.bias {
            validate_tensor_data_type!(bias, Float | Double)?;
        }
        validate_tensor_data_type!(&self.weights, Float | Double)
    }
}

impl GatherInits {
    pub fn validate(&self) -> Result<(), String> {
        validate_tensor_data_type!(&self.index, Int32 | Int64)
    }
}

impl ReshapeInits {
    pub fn validate(&self) -> Result<(), String> {
        validate_tensor_data_type!(&self.shape, Int32 | Int64)
    }
}

impl GemmInits {
    pub fn validate(&self) -> Result<(), String> {
        validate_tensor_data_type!(&self.b, Float | Double)?;
        validate_tensor_data_type!(&self.c, Float | Double)
    }
}

impl BatchNormInits {
    pub fn validate(&self) -> Result<(), String> {
        validate_tensor_data_type!(&self.scale, Float | Double)?;
        validate_tensor_data_type!(&self.bias, Float | Double)?;
        validate_tensor_data_type!(&self.mean, Float | Double)?;
        validate_tensor_data_type!(&self.var, Float | Double)
    }
}
