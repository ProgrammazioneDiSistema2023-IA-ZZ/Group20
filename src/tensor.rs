//! # Tensor utilities
//!
//! This module contains utilities to deal with dynamic ONNX tensors.
//!
//! The main struct is [`Tensor`], which contains the name of the tensor and its data.
//! The data is stored in the [`TensorData`] enum, which contains the actual array with generic element data type.
use ndarray::{ArrayBase, ArrayD, IxDyn, OwnedRepr};
use num_derive::{FromPrimitive, ToPrimitive};
use num_traits::FromPrimitive;

use crate::onnx_format::tensor_shape_proto::dimension::Value as TensorDimProto;
use crate::onnx_format::type_proto::Value as ValueTypeProto;

use crate::onnx_format::{TensorProto, TensorShapeProto, ValueInfoProto};

pub type TensorParametrizedShape = Vec<GraphDimension>;

/// Enum representing the different types of tensors in ONNX.
///
/// ONNX uses different types of tensors according to what the tensor is used for:
/// - **statically sized**: have a known shape prior to the model execution and known constant values prior to execution.
/// These are used for attributes and initializers tensors.
/// We'll refer to them as **constant tensors**.
/// - **graph input/output tensors**: *partially* know its shape and have an unknown data type prior to execution. Its values are fed or fetched during execution.
/// This is a tricky one, because the shape is not fully known on the model definition, but it is known before the model execution.
/// This is because the shape is defined by the input data. We'll refer to them as **in/out tensors**.
/// - **dynamically sized tensors**: have an unknown shape prior to execution.
/// Its shape and values are determined during execution.
/// These are used for intermediate values between nodes. We'll refer to them as **dynamic tensors**.
#[derive(Debug, Clone)]
pub enum Tensor {
    /// Tensor with a known shape and data type prior to execution.
    /// Its values are constant and also known before the model execution.
    ///
    /// These are used for attributes and initializers tensors.
    Constant(TensorData),

    /// Tensor with either a partially known shape on model definition,
    /// or a fully known shape on model definition,
    /// but is defined by a chosen user input/output before executing it.
    /// Its data type is unknown before the model execution.
    ///
    /// Its values are either given as input right before the model execution or
    /// fetched as output right after the model execution.
    /// These are used for graph input and output tensors.
    /// This tensor is defined to warn the programmer to check the shape of the TensorData when setted.
    InOut(TensorParametrizedShape, Option<TensorData>),

    /// Tensor with an unknown shape prior to execution. Its shape and values are determined during execution.
    ///
    /// These are used for intermediate values between nodes.
    Dynamic(Option<TensorData>),
}
/// This is a model input/output dimension.
/// It can be either a known value or a placeholder for a dimension that is not known
/// and should be provided before execution.
#[derive(Debug, Clone)]
pub enum GraphDimension {
    /// An actual dimension with a known value.
    Value(usize),

    /// This is a placeholder for a dimension that is not known and should be fed before execution.
    /// In ONNX this is known as `dim_param` inside a TensorShapeProto.
    Parameter(String),
}

impl GraphDimension {
    /// Returns `true` if the dimension is a placeholder for a dimension that is not known and should be fed before execution.
    pub fn is_parametrized(&self) -> bool {
        matches!(self, GraphDimension::Parameter(_))
    }
}

/// TensorProto completely define a tensor in ONNX (shape, data type and data values).
/// This is what we call a *constant tensor* (see [`Tensor`]).
impl From<TensorProto> for Tensor {
    fn from(proto: TensorProto) -> Self {
        let dimensions = proto
            .dims
            .iter()
            .map(|x| *x as usize)
            .collect::<Vec<usize>>();

        let data = convert_proto_to_tensor_data(proto, dimensions);

        Tensor::Constant(data)
    }
}

impl From<TensorShapeProto> for Tensor {
    fn from(proto: TensorShapeProto) -> Self {
        let dimensions = proto
            .dim
            .into_iter()
            .filter_map(|dim_proto| {
                let Some(dim) = dim_proto.value else {
                    return None;
                };

                match dim {
                    TensorDimProto::DimParam(param) => Some(GraphDimension::Parameter(param)),
                    TensorDimProto::DimValue(known_dim) => {
                        Some(GraphDimension::Value(known_dim as usize))
                    }
                }
            })
            .collect::<TensorParametrizedShape>();

        Tensor::InOut(dimensions, None)
    }
}

impl Tensor {
    pub fn is_parametrized_io(&self) -> bool {
        if let Tensor::InOut(dims, _) = self {
            dims.iter().any(|x| x.is_parametrized())
        } else {
            false
        }
    }
}

impl TryFrom<ValueInfoProto> for Tensor {
    type Error = &'static str;

    fn try_from(value: ValueInfoProto) -> Result<Self, Self::Error> {
        let value_type = value
            .r#type
            .ok_or("ValueInfoProto does not have a type")?
            .value
            .ok_or("ValueInfoProto does not have a tensor type")?;

        match value_type {
            ValueTypeProto::TensorType(tensor_value) => {
                let shape_proto = tensor_value
                    .shape
                    .ok_or("TensorTypeProto does not have a shape")?;

                // this will fail if the shape is not parametrized as expected for a graph tensor
                Ok(Tensor::from(shape_proto))
            }
            _ => Err("ValueInfoProto is not a tensor"),
        }
    }
}

/// Enum representing the different types of data that can be stored in a tensor
/// in ONNX.
/// This is a subset of the types defined in the ONNX protobuf specification.
/// The tags are the same as the ones defined in the protobuf specification.
#[derive(Debug, Clone, Copy, FromPrimitive, ToPrimitive, PartialEq, Eq)]
pub enum TensorDataType {
    /// 32-bit floating point, equivalent to Rust's `f32`
    Float = 1,
    /// Unsigned 8-bit int, equivalent to Rust's `u8`
    Uint8,
    /// Signed 8-bit int, equivalent to Rust's `i8`
    Int8,
    /// Unsigned 16-bit int, equivalent to Rust's `u16`
    Uint16,
    /// Signed 16-bit int, equivalent to Rust's `i16`
    Int16,
    /// Signed 32-bit int, equivalent to Rust's `i32`
    Int32,
    /// Signed 64-bit int, equivalent to Rust's `i64`
    Int64,
    /// String, equivalent to Rust's `String`
    String,
    /// 64-bit floating point, equivalent to Rust's `f64`
    Double = 11,
    /// Unsigned 32-bit int, equivalent to Rust's `u32`
    Uint32,
    /// Unsigned 64-bit int, equivalent to Rust's `u64`
    Uint64,
}
#[derive(Debug, Clone)]
pub enum TensorData {
    Float(ArrayD<f32>),
    Double(ArrayD<f64>),
    Uint8(ArrayD<u8>),
    Uint16(ArrayD<u16>),
    Uint32(ArrayD<u32>),
    Uint64(ArrayD<u64>),
    Int8(ArrayD<i8>),
    Int16(ArrayD<i16>),
    Int32(ArrayD<i32>),
    Int64(ArrayD<i64>),
    String(ArrayD<String>),
}

pub trait TensorDataIntoDimensionality<T>
where
    T: TypeToTensorDataType + Copy,
{
    fn into_dimensionality<D>(self) -> ArrayBase<OwnedRepr<T>, D>
    where
        D: ndarray::Dimension;
}

pub trait DynamicTensorData<T>
where
    T: TypeToTensorDataType + Copy,
{
    fn new_dyn(data: ArrayD<T>) -> TensorData;
}

/// Trait used to map Rust types to ONNX types (for example `f32` is mapped to `Float`)
pub trait TypeToTensorDataType {
    /// Return the ONNX type for a Rust type
    fn tensor_data_type() -> TensorDataType;
}

fn convert_proto_to_tensor_data(proto: TensorProto, dimensions: Vec<usize>) -> TensorData {
    let element_data_type: TensorDataType =
        FromPrimitive::from_i32(proto.data_type()).expect("Invalid tensor element data type");

    match element_data_type {
        TensorDataType::Float => {
            let data: Vec<f32> = {
                match proto.raw_data {
                    Some(raw_data) => raw_data
                        .chunks_exact(4)
                        .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
                        .collect(),
                    None => proto.float_data,
                }
            };
            TensorData::Float(ArrayD::from_shape_vec(IxDyn(&dimensions), data).unwrap())
        }
        TensorDataType::Uint8 => {
            let data: Vec<u8> = {
                match proto.raw_data {
                    Some(raw_data) => raw_data,
                    None => proto.int32_data.iter().map(|x| *x as u8).collect(),
                }
            };
            TensorData::Uint8(ArrayD::from_shape_vec(IxDyn(&dimensions), data).unwrap())
        }
        TensorDataType::Int8 => {
            let data: Vec<i8> = {
                match proto.raw_data {
                    Some(raw_data) => raw_data.iter().map(|x| *x as i8).collect(),
                    None => proto.int32_data.iter().map(|x| *x as i8).collect(),
                }
            };
            TensorData::Int8(ArrayD::from_shape_vec(IxDyn(&dimensions), data).unwrap())
        }
        TensorDataType::Uint16 => {
            let data: Vec<u16> = {
                match proto.raw_data {
                    Some(raw_data) => raw_data
                        .chunks_exact(2)
                        .map(|x| u16::from_le_bytes([x[0], x[1]]))
                        .collect(),
                    None => proto.int32_data.iter().map(|x| *x as u16).collect(),
                }
            };

            TensorData::Uint16(ArrayD::from_shape_vec(IxDyn(&dimensions), data).unwrap())
        }
        TensorDataType::Int16 => {
            let data: Vec<i16> = {
                match proto.raw_data {
                    Some(raw_data) => raw_data
                        .chunks_exact(2)
                        .map(|x| i16::from_le_bytes([x[0], x[1]]))
                        .collect(),
                    None => proto.int32_data.iter().map(|x| *x as i16).collect(),
                }
            };
            TensorData::Int16(ArrayD::from_shape_vec(IxDyn(&dimensions), data).unwrap())
        }
        TensorDataType::Int32 => {
            let data: Vec<i32> = {
                match proto.raw_data {
                    Some(raw_data) => raw_data
                        .chunks_exact(4)
                        .map(|x| i32::from_le_bytes([x[0], x[1], x[2], x[3]]))
                        .collect(),
                    None => proto.int32_data,
                }
            };
            TensorData::Int32(ArrayD::from_shape_vec(IxDyn(&dimensions), data).unwrap())
        }
        TensorDataType::Int64 => {
            let data: Vec<i64> = {
                match proto.raw_data {
                    Some(raw_data) => raw_data
                        .chunks_exact(8)
                        .map(|x| {
                            i64::from_le_bytes([x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]])
                        })
                        .collect(),
                    None => proto.int64_data,
                }
            };
            TensorData::Int64(ArrayD::from_shape_vec(IxDyn(&dimensions), data).unwrap())
        }
        TensorDataType::String => {
            let data: Vec<String> = {
                match proto.raw_data {
                    Some(raw_data) => raw_data
                        .split(|x| *x == 0)
                        .map(|x| String::from_utf8(x.to_vec()).expect("Invalid UTF-8 string"))
                        .collect(),
                    None => proto
                        .string_data
                        .into_iter()
                        .map(|x| String::from_utf8(x).expect("Invalid UTF-8 string"))
                        .collect(),
                }
            };

            TensorData::String(ArrayD::from_shape_vec(IxDyn(&dimensions), data).unwrap())
        }
        TensorDataType::Double => {
            let data: Vec<f64> = {
                match proto.raw_data {
                    Some(raw_data) => raw_data
                        .chunks_exact(8)
                        .map(|x| {
                            f64::from_le_bytes([x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]])
                        })
                        .collect(),
                    None => proto.double_data,
                }
            };

            TensorData::Double(ArrayD::from_shape_vec(IxDyn(&dimensions), data).unwrap())
        }
        TensorDataType::Uint32 => {
            let data: Vec<u32> = {
                match proto.raw_data {
                    Some(raw_data) => raw_data
                        .chunks_exact(4)
                        .map(|x| u32::from_le_bytes([x[0], x[1], x[2], x[3]]))
                        .collect(),
                    None => proto.int32_data.iter().map(|x| *x as u32).collect(),
                }
            };
            TensorData::Uint32(ArrayD::from_shape_vec(IxDyn(&dimensions), data).unwrap())
        }
        TensorDataType::Uint64 => {
            let data: Vec<u64> = {
                match proto.raw_data {
                    Some(raw_data) => raw_data
                        .chunks_exact(8)
                        .map(|x| {
                            u64::from_le_bytes([x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]])
                        })
                        .collect(),
                    None => proto.int64_data.iter().map(|x| *x as u64).collect(),
                }
            };
            TensorData::Uint64(ArrayD::from_shape_vec(IxDyn(&dimensions), data).unwrap())
        }
    }
}

/// This macro is used to map for each TensorData variant to a common action, that is independent of the actual data type.
macro_rules! dynamic_map {
    ($dyn_tensor_data:expr, |$data:pat_param| $action:expr) => {
        match $dyn_tensor_data {
            TensorData::Float($data) => $action,
            TensorData::Double($data) => $action,
            TensorData::Uint8($data) => $action,
            TensorData::Uint16($data) => $action,
            TensorData::Uint32($data) => $action,
            TensorData::Uint64($data) => $action,
            TensorData::Int8($data) => $action,
            TensorData::Int16($data) => $action,
            TensorData::Int32($data) => $action,
            TensorData::Int64($data) => $action,
            TensorData::String($data) => $action,
        }
    };
}

impl TensorData {
    pub fn shape(&self) -> &[usize] {
        dynamic_map!(self, |data| data.shape())
    }
}

macro_rules! impl_type_trait {
    ($type_:ty, $variant:ident) => {
        impl TypeToTensorDataType for $type_ {
            fn tensor_data_type() -> TensorDataType {
                TensorDataType::$variant
            }
        }

        // implement conversion from ndarray to TensorData
        impl From<ArrayD<$type_>> for TensorData {
            fn from(array: ArrayD<$type_>) -> Self {
                TensorData::$variant(array)
            }
        }

        //implement dimensionality conversion from TensorData to ndarray
        impl TensorDataIntoDimensionality<$type_> for TensorData {
            fn into_dimensionality<D>(self) -> ArrayBase<OwnedRepr<$type_>, D>
            where
                D: ndarray::Dimension,
            {
                match self {
                    TensorData::$variant(array) => array.into_dimensionality::<D>().unwrap(),
                    _ => panic!("Invalid conversion"),
                }
            }
        }
        //implement dynamic conversion from TensorData to ndarray
        impl DynamicTensorData<$type_> for TensorData {
            fn new_dyn(data: ArrayD<$type_>) -> TensorData {
                TensorData::$variant(data)
            }
        }
    };
}

impl_type_trait!(f32, Float);
impl_type_trait!(u8, Uint8);
impl_type_trait!(i8, Int8);
impl_type_trait!(u16, Uint16);
impl_type_trait!(i16, Int16);
impl_type_trait!(i32, Int32);
impl_type_trait!(i64, Int64);
impl_type_trait!(f64, Double);
impl_type_trait!(u32, Uint32);
impl_type_trait!(u64, Uint64);
