use crate::onnx_format;
use crate::operators::{Operator, Tensor, BatchNormAttributes, ConvAttributes, MaxPoolAttributes, GemmAttributes};

use core::panic;
use std::{fs::File, io::Read};
use prost::Message;
use petgraph::Graph;

fn get_parsed_model(path: &str) -> onnx_format::ModelProto {
    let mut buffer = Vec::new();
    let mut file = File::open(path).unwrap();
    file.read_to_end(&mut buffer).unwrap();

    let parsed_model = onnx_format::ModelProto::decode(buffer.as_slice());
    parsed_model.expect("Failed to unwrap parsed_model in get_parsed_model()")
}

pub fn create_graph(){
    let path = "tests/models/resnet18-v2-7.onnx";

    let parsed_model = get_parsed_model(path);
    let nodes = parsed_model.graph.expect("Failed to unwrap graph in create_graph()").node;

    let mut graph: Graph<Operator, Tensor> = Graph::<Operator, Tensor>::new();

    for node in nodes{

        //let pg = deps.add_node("petgraph");

        let op_type = node.op_type.expect("Failed to unwrap node op_type in create_graph()");

        let operator: Operator = match op_type.as_str() {
            "BatchNormalitazion" => {

                let attrs = BatchNormAttributes::new(
                    node.attribute[0].f.expect("Failed to unwrap f in epsilon attribute"),
                    node.attribute[1].f.expect("Failed to unwrap f in momentum attribute"),
                    node.attribute[2].i.expect("Failed to unwrap i in momentum attribute")
                );

                Operator::BatchNorm(attrs)
            },
            "Conv" => {
                let attrs = ConvAttributes::new(
                    [node.attribute[0].ints[0] as usize, node.attribute[0].ints[1] as usize],
                    node.attribute[1].i.expect("Failed to unwrap i in group attribute") as usize,
                    [node.attribute[2].ints[0] as usize, node.attribute[2].ints[1] as usize],
                    [node.attribute[3].ints[0] as usize, node.attribute[3].ints[1] as usize, node.attribute[3].ints[2] as usize, node.attribute[3].ints[3] as usize ],
                    [node.attribute[4].ints[0] as usize, node.attribute[4].ints[1] as usize],
                );

                Operator::Convolution(attrs)
            },
            "Relu" => {
                Operator::ReLU
            },
            "MaxPool" => {
                let attrs: MaxPoolAttributes = MaxPoolAttributes::new(
                    [node.attribute[0].ints[0] as usize, node.attribute[0].ints[1] as usize],
                    [node.attribute[1].ints[0] as usize, node.attribute[1].ints[1] as usize, node.attribute[1].ints[2] as usize, node.attribute[1].ints[3] as usize ],
                    [node.attribute[2].ints[0] as usize, node.attribute[2].ints[1] as usize],
                );

                Operator::MaxPool(attrs)
            },
            "Add" => {
                Operator::Add
            },
            "GlobalAveragePool" => {
                Operator::GlobalAveragePool
            },
            "Reshape" => {
                Operator::Reshape
            },
            "Gemm" => {
                let attrs: GemmAttributes = GemmAttributes::new(
                    node.attribute[0].f.expect("Failed to unwrap f in alpha attribute"),
                    node.attribute[1].f.expect("Failed to unwrap f in beta attribute"),
                    node.attribute[2].i.expect("Failed to unwrap i in trans_a attribute"),
                    node.attribute[3].i.expect("Failed to unwrap i in trans_b attribute")
                );

                Operator::Gemm(attrs)
            },
            _ => panic!("No matched value for op_type"),
        };

    }

}



