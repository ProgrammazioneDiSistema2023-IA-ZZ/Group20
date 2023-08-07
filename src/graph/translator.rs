use crate::onnx_format;
use crate::onnx_format::TensorProto;
use crate::operators::{
    BatchNormAttributes, BatchNormInputs, ClipAttributes, ConcatAttributes, ConvAttributes,
    ConvInputs, GatherAttributes, GatherInputs, GemmAttributes, GemmInputs, MaxPoolAttributes,
    Operator, ReshapeInputs, TensorData, UnsqueezeAttributes,
};

use core::panic;
use petgraph::Graph;
use prost::Message;
use std::{fs::File, io::Read};

//da rimuovere
#[derive(Debug)]
pub struct Tensor {
    pub name: String,
    pub data: TensorData,
}
impl From<TensorProto> for Tensor {
    fn from(proto: TensorProto) -> Self {
        todo!();
    }
}

#[test]
fn print_parsed_model_test() {
    let mut buffer = Vec::new();
    // let mut file = File::open("tests/models/mobilenetv2-10.onnx").unwrap();
    let mut file = File::open("tests/models/resnet18-v2-7.onnx").unwrap();
    file.read_to_end(&mut buffer).unwrap();

    let parsed_model = onnx_format::ModelProto::decode(buffer.as_slice());
    let initializer = parsed_model.unwrap().graph.unwrap().initializer;

    // println!("\nlength of initializer: {}\n\n", initializer.len());

    // for tensor in initializer{
    //    println!("{:?}\n\n\n", tensor);
    //    break;
    // }

    // println!("{:?}\n\n\n", initializer[0]);

    // for node in parsed_model.unwrap().graph.unwrap().node{
    //     println!("{:?}\n", node );
    // }
}

fn get_parsed_model(path: &str) -> onnx_format::ModelProto {
    let mut buffer = Vec::new();
    let mut file = File::open(path).unwrap();
    file.read_to_end(&mut buffer).unwrap();

    let parsed_model = onnx_format::ModelProto::decode(buffer.as_slice());
    parsed_model.expect("Failed to unwrap parsed_model in get_parsed_model()")
}

pub fn create_graph() {
    let path_resnet = "tests/models/resnet18-v2-7.onnx";
    //let path_mobilenet = "tests/models/mobilenetv2-10.onnx";

    let parsed_model = get_parsed_model(path_resnet);
    let graph_proto = parsed_model
        .graph
        .expect("Failed to unwrap graph in create_graph()");
    let initializers = graph_proto.initializer;
    let nodes = graph_proto.node;

    let mut graph: Graph<Operator, Option<TensorData>> =
        Graph::<Operator, Option<TensorData>>::new();

    for node in nodes {
        //let pg = deps.add_node("petgraph");

        let op_type = node
            .op_type
            .expect("Failed to unwrap node op_type in create_graph()");
        let inputs = node.input;

        let operator: Operator = match op_type.as_str() {
            "BatchNormalitazion" => {
                let attrs = BatchNormAttributes::new(
                    node.attribute[0]
                        .f
                        .expect("Failed to unwrap f in epsilon attribute"),
                    node.attribute[1]
                        .f
                        .expect("Failed to unwrap f in momentum attribute"),
                    node.attribute[2]
                        .i
                        .expect("Failed to unwrap i in momentum attribute"),
                );

                let mut vec = Vec::<TensorData>::new();
                let inps;

                for inp in inputs {
                    let v = initializers
                        .iter()
                        .find(|tp| *tp.name() == inp)
                        .expect("Failed to filter initializers");
                    vec.push(Tensor::from(v.clone()).data);
                }

                if let [scale, b, mean, var] = &vec[..] {
                    inps =
                        BatchNormInputs::new(scale.clone(), b.clone(), mean.clone(), var.clone());
                } else {
                    panic!("inps uninitialized. Maybe initilizer is not defined");
                }

                Operator::BatchNorm(inps, attrs)
            }
            "Conv" => {
                let attrs = ConvAttributes::new(
                    [
                        node.attribute[0].ints[0] as usize,
                        node.attribute[0].ints[1] as usize,
                    ],
                    node.attribute[1]
                        .i
                        .expect("Failed to unwrap i in group attribute")
                        as usize,
                    [
                        node.attribute[2].ints[0] as usize,
                        node.attribute[2].ints[1] as usize,
                    ],
                    [
                        node.attribute[3].ints[0] as usize,
                        node.attribute[3].ints[1] as usize,
                        node.attribute[3].ints[2] as usize,
                        node.attribute[3].ints[3] as usize,
                    ],
                    [
                        node.attribute[4].ints[0] as usize,
                        node.attribute[4].ints[1] as usize,
                    ],
                );

                let mut vec = Vec::<TensorData>::new();
                let inps;

                for inp in inputs {
                    let v = initializers
                        .iter()
                        .find(|tp| *tp.name() == inp)
                        .expect("Failed to filter initializers");
                    vec.push(Tensor::from(v.clone()).data);
                }

                if let [weights, bias] = &vec[..] {
                    inps = ConvInputs::new(weights.clone(), Some(bias.clone()));
                } else if let [weights] = &vec[..] {
                    inps = ConvInputs::new(weights.clone(), None);
                } else {
                    panic!("inps uninitialized. Maybe initilizer is not defined");
                }

                Operator::Convolution(inps, attrs)
            }
            "Relu" => Operator::ReLU,
            "MaxPool" => {
                let attrs: MaxPoolAttributes = MaxPoolAttributes::new(
                    [
                        node.attribute[0].ints[0] as usize,
                        node.attribute[0].ints[1] as usize,
                    ],
                    [
                        node.attribute[1].ints[0] as usize,
                        node.attribute[1].ints[1] as usize,
                        node.attribute[1].ints[2] as usize,
                        node.attribute[1].ints[3] as usize,
                    ],
                    [
                        node.attribute[2].ints[0] as usize,
                        node.attribute[2].ints[1] as usize,
                    ],
                );

                Operator::MaxPool(attrs)
            }
            "Add" => Operator::Add,
            "GlobalAveragePool" => Operator::GlobalAveragePool,
            "Reshape" => {
                let mut vec = Vec::<TensorData>::new();
                let inps;

                for inp in inputs {
                    let v = initializers
                        .iter()
                        .find(|tp| *tp.name() == inp)
                        .expect("Failed to filter initializers");
                    vec.push(Tensor::from(v.clone()).data);
                }

                if let [shape] = &vec[..] {
                    inps = ReshapeInputs::new(shape.clone());
                } else {
                    panic!("inps uninitialized. Maybe initilizer is not defined");
                }

                Operator::Reshape(inps)
            }
            "Gemm" => {
                let attrs: GemmAttributes = GemmAttributes::new(
                    node.attribute[0]
                        .f
                        .expect("Failed to unwrap f in alpha attribute"),
                    node.attribute[1]
                        .f
                        .expect("Failed to unwrap f in beta attribute"),
                    node.attribute[2]
                        .i
                        .expect("Failed to unwrap i in trans_a attribute"),
                    node.attribute[3]
                        .i
                        .expect("Failed to unwrap i in trans_b attribute"),
                );

                let mut vec = Vec::<TensorData>::new();
                let inps;

                for inp in inputs {
                    let v = initializers
                        .iter()
                        .find(|tp| *tp.name() == inp)
                        .expect("Failed to filter initializers");
                    vec.push(Tensor::from(v.clone()).data);
                }

                if let [b, c] = &vec[..] {
                    inps = GemmInputs::new(b.clone(), c.clone());
                } else {
                    panic!("inps uninitialized. Maybe initilizer is not defined");
                }

                Operator::Gemm(inps, attrs)
            }
            "Clip" => {
                let attrs: ClipAttributes = ClipAttributes::new(
                    node.attribute[1]
                        .f
                        .expect("Failed to unwrap f in min attribute"),
                    node.attribute[0]
                        .f
                        .expect("Failed to unwrap f in max attribute"),
                );

                Operator::Clip(attrs)
            }
            "Shape" => Operator::Shape,
            "Gather" => {
                let attrs: GatherAttributes = GatherAttributes::new(
                    node.attribute[0]
                        .i
                        .expect("Failed to unwrap i in axes attribute")
                        as usize,
                );

                let mut vec = Vec::<TensorData>::new();
                let inps;

                for inp in inputs {
                    let v = initializers
                        .iter()
                        .find(|tp| *tp.name() == inp)
                        .expect("Failed to filter initializers");
                    vec.push(Tensor::from(v.clone()).data);
                }

                if let [index] = &vec[..] {
                    inps = GatherInputs::new(index.clone());
                } else {
                    panic!("inps uninitialized. Maybe initilizer is not defined");
                }

                Operator::Gather(inps, attrs)
            }
            "Unsqueeze" => {
                let attrs: UnsqueezeAttributes =
                    UnsqueezeAttributes::new(node.attribute[0].ints[0] as usize);

                Operator::Unsqueeze(attrs)
            }
            "Concat" => {
                let attrs: ConcatAttributes = ConcatAttributes::new(
                    node.attribute[0]
                        .i
                        .expect("Failed to unwrap i in axes attribute")
                        as usize,
                );

                Operator::Concat(attrs)
            }
            _ => panic!("No matched value for op_type"),
        };

        let n = graph.add_node(operator);
    }
}
