use std::{fs::File, io::Read, path::Path};

use prost::Message;

use crate::onnx_format::ModelProto;

use super::{prepare::batch_preprocessing, ServiceError};

/// Read and prepare images for the ImageNet dataset.
///
/// It will load the images from the given paths, preprocess them and return an array4.
pub fn read_and_prepare_images<P>(paths: &[P]) -> Result<ndarray::Array4<f32>, ServiceError>
where
    P: AsRef<Path>,
{
    let images = paths
        .iter()
        .map(image::open)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| ServiceError::InvalidInput(Box::new(e)))?;

    batch_preprocessing(images.as_slice()).map_err(|e| ServiceError::InvalidInput(Box::new(e)))
}

/// Read an ONNX model from the given path.
/// It will return a [`ModelProto`] that can be used by a [`Service`].
pub fn read_model_proto<P>(path: P) -> ModelProto
where
    P: AsRef<Path>,
{
    let mut buffer = Vec::new();
    let mut file = File::open(path).unwrap();
    file.read_to_end(&mut buffer).unwrap();

    ModelProto::decode(buffer.as_slice()).unwrap()
}
