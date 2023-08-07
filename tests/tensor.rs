use std::{fs::File, io::Read};

use onnx_runtime::{onnx_format::ModelProto, tensor::Tensor};
use prost::Message;

#[test]
fn proto_to_tensor_data() {
    let mut buffer = Vec::new();
    let mut file = File::open("tests/models/mobilenetv2-10.onnx").unwrap();
    file.read_to_end(&mut buffer).unwrap();

    let parsed_model = ModelProto::decode(buffer.as_slice()).unwrap();

    // this should not panic for now
    let _tensors = parsed_model
        .graph
        .unwrap()
        .initializer
        .iter()
        .map(|x| Tensor::from(x.clone()))
        .collect::<Vec<Tensor>>();
}
