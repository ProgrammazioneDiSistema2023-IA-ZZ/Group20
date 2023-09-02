use ndarray::{Array2, Array4};
use prost::Message;
use std::{
    env,
    fs::File,
    io::{Read, Write},
    path::PathBuf,
};

use onnx_runtime::onnx_format::{
    type_proto::Tensor as TypeTensorProto, type_proto::Value as TypeValueProto, GraphProto,
    ModelProto, NodeProto, OperatorSetIdProto, TensorProto, TypeProto, ValueInfoProto,
};

#[test]
fn deserialize_prebuilt_model() {
    let mut buffer = Vec::new();
    let mut file = File::open("tests/models/mobilenetv2-7.onnx").unwrap();
    file.read_to_end(&mut buffer).unwrap();

    let parsed_model = ModelProto::decode(buffer.as_slice());

    assert!(parsed_model.is_ok());
}

#[test]
fn serialize_simple_model() {
    // example based on https://onnx.ai/onnx/intro/python.html
    let example_model = ModelProto {
        ir_version: Some(9),
        graph: Some(GraphProto {
            node: vec![
                NodeProto {
                    input: vec![String::from("X"), String::from("A")],
                    output: vec![String::from("XA")],
                    op_type: Some(String::from("MatMul")),
                    ..Default::default()
                },
                NodeProto {
                    input: vec![String::from("XA"), String::from("B")],
                    output: vec![String::from("Y")],
                    op_type: Some(String::from("Add")),
                    ..Default::default()
                },
            ],
            name: Some(String::from("lr")),
            input: vec![
                ValueInfoProto {
                    name: Some(String::from("X")),
                    r#type: Some(TypeProto {
                        value: Some(TypeValueProto::TensorType(TypeTensorProto {
                            elem_type: Some(1),
                            ..Default::default()
                        })),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                ValueInfoProto {
                    name: Some(String::from("A")),
                    r#type: Some(TypeProto {
                        value: Some(TypeValueProto::TensorType(TypeTensorProto {
                            elem_type: Some(1),
                            ..Default::default()
                        })),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                ValueInfoProto {
                    name: Some(String::from("B")),
                    r#type: Some(TypeProto {
                        value: Some(TypeValueProto::TensorType(TypeTensorProto {
                            elem_type: Some(1),
                            ..Default::default()
                        })),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            ],
            output: vec![ValueInfoProto {
                name: Some(String::from("Y")),
                r#type: Some(TypeProto {
                    value: Some(TypeValueProto::TensorType(TypeTensorProto {
                        elem_type: Some(1),
                        ..Default::default()
                    })),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        }),
        opset_import: vec![OperatorSetIdProto {
            version: Some(20),
            ..Default::default()
        }],
        ..Default::default()
    };

    //use a temporary file
    let mut tmp_path: PathBuf = env::temp_dir();
    tmp_path.push("my_model.onnx");

    let mut buffer = Vec::new();
    let serialization = example_model.encode(&mut buffer);

    let mut serialized_file = File::create(tmp_path).unwrap();
    serialized_file.write_all(&buffer).unwrap();

    assert!(serialization.is_ok())
}

#[test]
fn deserialize_test_data() {
    let mobilenet_input = load_input("tests/testset/mobilenet/input_0.pb");
    let resnet_input = load_input("tests/testset/resnet/input_0.pb");
    let mobilenet_output = load_output("tests/testset/mobilenet/output_0.pb");
    let resnet_output = load_output("tests/testset/resnet/output_0.pb");

    assert_eq!(mobilenet_input.shape(), [1, 3, 224, 224]);
    assert_eq!(resnet_input.shape(), [1, 3, 224, 224]);
    assert_eq!(mobilenet_output.shape(), [1, 1000]);
    assert_eq!(resnet_output.shape(), [1, 1000]);
}

pub fn load_input(path: &str) -> Array4<f32> {
    tensor_proto_to_arr4(load_tensor_from_pb(path))
}

pub fn load_output(path: &str) -> Array2<f32> {
    tensor_proto_to_arr2(load_tensor_from_pb(path))
}

fn tensor_proto_to_arr4(tensor: TensorProto) -> Array4<f32> {
    let [batch_size, chans, height, width] = *(tensor
        .dims
        .iter()
        .map(|&v| v as usize)
        .collect::<Vec<usize>>())
    else {
        todo!("Unexpected format")
    };
    let raw_data = tensor
        .raw_data
        .expect("Cannot extract raw data from Tensor");
    let mut f32_data: Vec<f32> = Vec::new();
    for chunk in raw_data.chunks_exact(4) {
        let bytes: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
        let num: u32 = u32::from_le_bytes(bytes);
        let float_num: f32 = f32::from_bits(num);
        f32_data.push(float_num);
    }
    Array4::from_shape_vec([batch_size, chans, height, width], f32_data).unwrap()
}

fn tensor_proto_to_arr2(tensor: TensorProto) -> Array2<f32> {
    let [batch_size, outputs] = *(tensor
        .dims
        .iter()
        .map(|&v| v as usize)
        .collect::<Vec<usize>>())
    else {
        todo!("Unexpected format")
    };
    let raw_data = tensor
        .raw_data
        .expect("Cannot extract raw data from Tensor");
    let mut f32_data: Vec<f32> = Vec::new();
    for chunk in raw_data.chunks_exact(4) {
        let bytes: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
        let num: u32 = u32::from_le_bytes(bytes);
        let float_num: f32 = f32::from_bits(num);
        f32_data.push(float_num);
    }
    Array2::from_shape_vec([batch_size, outputs], f32_data).unwrap()
}

fn load_tensor_from_pb(path: &str) -> TensorProto {
    let mut buffer = Vec::new();
    let mut file = File::open(path).unwrap();
    file.read_to_end(&mut buffer).unwrap();

    TensorProto::decode(buffer.as_slice()).expect("Unable to read TensorProto")
}
