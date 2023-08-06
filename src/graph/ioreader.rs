use std::{fs::File, io::Read};

use ndarray::{Array2, Array4};
use prost::Message;

use crate::onnx_format::TensorProto;

pub fn load_input(path: &str) -> Array4<f32> {
    tensor_proto_to_arr4(load_tensor_from_pb(path))
}

pub fn load_output(path: &str) -> Array2<f32> {
    tensor_proto_to_arr2(load_tensor_from_pb(path))
}

fn tensor_proto_to_arr4(tensor: TensorProto) -> Array4<f32> {
    let [batch_size, chans, height, width] = *(tensor.dims.iter().map(|&v| v as usize).collect::<Vec<usize>>()) else {todo!("Unexpected format")};
    // println!("{:?}", tensor);
    let raw_data = tensor.raw_data.expect("Cannot extract raw data from Tensor");
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
    let [batch_size, outputs] = *(tensor.dims.iter().map(|&v| v as usize).collect::<Vec<usize>>()) else {todo!("Unexpected format")};
    // println!("{:?}", tensor);
    let raw_data = tensor.raw_data.expect("Cannot extract raw data from Tensor");
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