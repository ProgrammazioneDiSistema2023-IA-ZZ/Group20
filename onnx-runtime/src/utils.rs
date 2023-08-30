use std::{fs::File, io::Read, path::PathBuf};

use prost::Message;

use crate::{onnx_format::ModelProto, prepare::preprocessing};
pub use image::ImageError;

pub fn read_model_proto(path: &str) -> ModelProto {
    let mut buffer = Vec::new();
    let mut file = File::open(path).unwrap();
    file.read_to_end(&mut buffer).unwrap();

    ModelProto::decode(buffer.as_slice()).unwrap()
}

pub fn read_and_prepare_image(path: PathBuf) -> Result<ndarray::Array4<f32>, ImageError> {
    let image = image::open(path)?;
    Ok(preprocessing(image))
}
