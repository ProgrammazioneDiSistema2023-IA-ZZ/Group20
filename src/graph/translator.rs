use crate::onnx_format;
use crate::operators::*;

use crate::tensor::{Tensor, TensorData};

use core::panic;
use petgraph::algo::toposort;
use petgraph::graph::NodeIndex;
use petgraph::Graph;
use prost::Message;
use std::collections::HashMap;
use std::{fs::File, io::Read};

#[allow(dead_code)]
#[derive(Debug)]
struct NodeInfo {
    parents_name: Vec<String>,
    node_index: usize,
}

impl NodeInfo {
    pub fn new(parents_name: Vec<String>, node_index: usize) -> Self {
        Self {
            parents_name,
            node_index,
        }
    }
}

pub fn create_graph() {
    let path_resnet = "tests/models/resnet18-v2-7.onnx";
    //let path_mobilenet = "tests/models/mobilenetv2-10.onnx";

    let parsed_model = get_parsed_model(path_resnet);

    let graph_proto = parsed_model
        .graph
        .expect("Failed to unwrap graph in create_graph()");

    let graph_input = graph_proto.input;
    let graph_output = graph_proto.output;

    let initializers = graph_proto.initializer;
    let nodes = graph_proto.node;

    let mut map: HashMap<String, NodeInfo> = HashMap::new();

    let mut graph: Graph<Operator, Option<TensorData>> =
        Graph::<Operator, Option<TensorData>>::new();

    for node in nodes {
        //let pg = deps.add_node("petgraph");

        let op_type = node
            .op_type
            .expect("Failed to unwrap node op_type in create_graph()");
        let mut inputs = node.input;
        let node_name = node.name.expect("Failed to recevor node name!");
        let parents_name: Vec<String>;

        let operator: Operator = match op_type.as_str() {
            "BatchNormalization" => {
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

                parents_name = vec![inputs.remove(0)];

                for inp in inputs {
                    let v = initializers
                        .iter()
                        .find(|tp| *tp.name().to_string() == inp)
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

                parents_name = vec![inputs.remove(0)];

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
            "Relu" => {
                parents_name = vec![inputs.remove(0)];

                Operator::ReLU
            }
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

                parents_name = vec![inputs.remove(0)];

                Operator::MaxPool(attrs)
            }
            "Add" => {
                parents_name = vec![inputs.remove(0), inputs.remove(0)];
                Operator::Add
            }
            "GlobalAveragePool" => {
                parents_name = vec![inputs.remove(0)];
                Operator::GlobalAveragePool
            }
            "Reshape" => {
                let mut vec = Vec::<TensorData>::new();
                let inps;

                parents_name = vec![inputs.remove(0)];

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

                parents_name = vec![inputs.remove(0)];

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

                parents_name = vec![inputs.remove(0)];

                Operator::Clip(attrs)
            }
            "Shape" => {
                parents_name = vec![inputs.remove(0)];
                Operator::Shape
            }
            "Gather" => {
                let attrs: GatherAttributes = GatherAttributes::new(
                    node.attribute[0]
                        .i
                        .expect("Failed to unwrap i in axes attribute")
                        as usize,
                );

                let mut vec = Vec::<TensorData>::new();
                let inps;

                parents_name = vec![inputs.remove(0)];

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

                parents_name = vec![inputs.remove(0)];

                Operator::Unsqueeze(attrs)
            }
            "Concat" => {
                let attrs: ConcatAttributes = ConcatAttributes::new(
                    node.attribute[0]
                        .i
                        .expect("Failed to unwrap i in axes attribute")
                        as usize,
                );

                parents_name = inputs.clone();

                Operator::Concat(attrs)
            }
            _ => panic!("No matched value for op_type"),
        };

        let n = graph.add_node(operator);
        let ni = NodeInfo::new(parents_name, n.index());
        map.insert(node_name, ni);
    }

    for (_n_name, n_info) in map.iter() {
        let parents = &n_info.parents_name;
        let n_index = n_info.node_index;
        for p_name in parents {
            let p_index = map.get(p_name).expect("Parent not found").node_index;
            graph.add_edge(NodeIndex::new(n_index), NodeIndex::new(p_index), None);
        }
    }

    // let _toposort = toposort(&graph, None).unwrap().into_iter().rev().for_each(|n| {
    //     println!("{:?}\n", graph[n].name());
    // });
}

#[test]
fn print_parsed_model_test() {
    let mut buffer = Vec::new();
    // let mut file = File::open("tests/models/mobilenetv2-10.onnx").unwrap();
    let mut file = File::open("tests/models/resnet18-v2-7.onnx").unwrap();
    file.read_to_end(&mut buffer).unwrap();

    let parsed_model = onnx_format::ModelProto::decode(buffer.as_slice());
    let _initializer = parsed_model.unwrap().graph.unwrap().initializer;

    // println!("\nlength of initializer: {}\n\n", initializer.len());

    // for tensor in initializer{
    //    println!("{:?}\n\n\n", tensor);
    //    break;
    // }

    // println!("{:?}\n\n\n", initializer[0]);

    // for node in parsed_model.unwrap().graph.unwrap().node{
    //     println!("{:?}\n", node );
    // }

    create_graph();
}

fn get_parsed_model(path: &str) -> onnx_format::ModelProto {
    let mut buffer = Vec::new();
    let mut file = File::open(path).unwrap();
    file.read_to_end(&mut buffer).unwrap();

    let parsed_model = onnx_format::ModelProto::decode(buffer.as_slice());
    parsed_model.expect("Failed to unwrap parsed_model in get_parsed_model()")
}
