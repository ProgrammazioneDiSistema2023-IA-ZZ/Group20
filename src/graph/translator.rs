use crate::onnx_format::ValueInfoProto;
use crate::onnx_format::{self, ModelProto};
use crate::operators::*;

use crate::tensor::{Tensor, TensorData};

use petgraph::algo::toposort;
use petgraph::dot::{Config, Dot};
use petgraph::graph::NodeIndex;
use petgraph::visit::GraphProp;
use petgraph::Graph;
use prost::Message;
use std::collections::HashMap;
use std::{fs::File, io::Read};

use super::GraphError;

type RuntimeGraph = Graph<Operator, Option<TensorData>>;

enum NodeInfo {
    Input(u32),
    Output(u32),
    Intermediate(u32, Vec<String>, Vec<String>),
}

impl NodeInfo {
    fn index(&self) -> u32 {
        match self {
            NodeInfo::Input(i) => *i,
            NodeInfo::Output(i) => *i,
            NodeInfo::Intermediate(i, _, _) => *i,
        }
    }
}

pub fn create_graph(model_proto: ModelProto) -> Result<RuntimeGraph, GraphError> {
    let mut parsed_nodes: HashMap<String, NodeInfo> = HashMap::new();

    let mut model_graph = RuntimeGraph::new();

    let graph_proto = match model_proto.graph {
        Some(g) => g,
        None => {
            return Err(GraphError::ConversionError(
                "Unable to retrieve graph from parsed model".to_string(),
            ))
        }
    };

    let graph_input = graph_proto.input;
    let graph_output = graph_proto.output;
    let initializers = graph_proto.initializer;
    let nodes = graph_proto.node;

    let (input_node_name, input_tensor) =
        parse_model_io_node(graph_input).expect("Unable to parse input node");
    let (output_node_name, output_tensor) =
        parse_model_io_node(graph_output).expect("Unable to parse output node");
    let input_node = model_graph.add_node(Operator::None);
    let output_node = model_graph.add_node(Operator::None);

    parsed_nodes.insert(input_node_name, NodeInfo::Input(input_node.index() as u32));
    println!("Input index: {}", input_node.index());
    for node in nodes {
        //let pg = deps.add_node("petgraph");

        let op_type = match node.op_type {
            Some(s) => s,
            None => {
                return Err(GraphError::ConversionError(
                    "Unable to convert op_type".to_string(),
                ))
            }
        };
        let mut inputs = node.input;
        let node_name = match node.name {
            Some(s) => s,
            None => {
                return Err(GraphError::ConversionError(
                    "Unable to recover node name".to_string(),
                ))
            }
        };
        let parents_names: Vec<String>;

        let operator: Operator = match op_type.as_str() {
            "BatchNormalization" => {
                let Some(epsilon) = node.attribute[0].f else{
                    return Err(GraphError::MissingOperand { operand: String::from("epsilon"), operator: node_name, operator_type:String::from("BatchNormalization")})
                };

                let Some(momentum) = node.attribute[1].f else{
                    return Err(GraphError::MissingOperand { operand: String::from("momentum"), operator: node_name, operator_type:String::from("BatchNormalization")})
                };

                let Some(spatial) = node.attribute[2].i else{
                    return Err(GraphError::MissingOperand { operand: String::from("spatial"), operator: node_name, operator_type:String::from("BatchNormalization")})
                };

                let attrs = BatchNormAttributes::new(epsilon, momentum, spatial);

                let mut useful_initializers: Vec<TensorData> = Vec::<TensorData>::new();
                let inps;

                parents_names = vec![inputs.remove(0)];

                for inp in inputs {
                    let v = match initializers.iter().find(|tp| *tp.name().to_string() == inp) {
                        Some(t) => t,
                        None => {
                            return Err(GraphError::ConversionError(
                                "Unable to filter initializers".to_string(),
                            ))
                        }
                    };

                    let Tensor::Constant(data) = Tensor::from(v.clone()) else{
                            return Err(GraphError::UnexpectedError);
                    };

                    useful_initializers.push(data);
                }

                if let [scale, b, mean, var] = &useful_initializers[..] {
                    inps =
                        BatchNormInputs::new(scale.clone(), b.clone(), mean.clone(), var.clone());
                } else {
                    return Err(GraphError::DeconstructError(
                        "Unable to retrieve inputs".to_string(),
                    ));
                }

                Operator::BatchNorm(inps, attrs)
            }
            "Conv" => {
                let attrs = ConvAttributes::new(
                    [
                        node.attribute[0].ints[0] as usize,
                        node.attribute[0].ints[1] as usize,
                    ],
                    match node.attribute[1].i {
                        Some(i) => i,
                        None => {
                            return Err(GraphError::ConversionError(
                                "Unable to unwrap i in group attribute".to_string(),
                            ))
                        }
                    } as usize,
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

                let mut useful_initializers = Vec::<TensorData>::new();
                let inps;

                parents_names = vec![inputs.remove(0)];

                for inp in inputs {
                    let v = match initializers.iter().find(|tp| *tp.name() == inp) {
                        Some(t) => t,
                        None => {
                            return Err(GraphError::ConversionError(
                                "Unable to filter initializers".to_string(),
                            ))
                        }
                    };

                    let Tensor::Constant(data) = Tensor::from(v.clone()) else{
                        return Err(GraphError::UnexpectedError);
                };

                    useful_initializers.push(data);
                }

                if let [weights, bias] = &useful_initializers[..] {
                    inps = ConvInputs::new(weights.clone(), Some(bias.clone()));
                } else if let [weights] = &useful_initializers[..] {
                    inps = ConvInputs::new(weights.clone(), None);
                } else {
                    return Err(GraphError::DeconstructError(
                        "Unable to retrieve inputs".to_string(),
                    ));
                }

                Operator::Convolution(inps, attrs)
            }
            "Relu" => {
                parents_names = vec![inputs.remove(0)];

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

                parents_names = vec![inputs.remove(0)];

                Operator::MaxPool(attrs)
            }
            "Add" => {
                parents_names = vec![inputs.remove(0), inputs.remove(0)];
                Operator::Add
            }
            "GlobalAveragePool" => {
                parents_names = vec![inputs.remove(0)];
                Operator::GlobalAveragePool
            }
            "Reshape" => {
                let mut useful_initializers = Vec::<TensorData>::new();
                let inps;

                parents_names = vec![inputs.remove(0)];

                for inp in inputs {
                    let v = match initializers.iter().find(|tp| *tp.name() == inp) {
                        Some(t) => t,
                        None => {
                            return Err(GraphError::ConversionError(
                                "Unable to filter initializers".to_string(),
                            ))
                        }
                    };

                    let Tensor::Constant(data) = Tensor::from(v.clone()) else{
                        return Err(GraphError::UnexpectedError);
                    };

                    useful_initializers.push(data);
                }

                if let [shape] = &useful_initializers[..] {
                    inps = ReshapeInputs::new(shape.clone());
                } else {
                    return Err(GraphError::DeconstructError(
                        "Unable to retrieve inputs".to_string(),
                    ));
                }

                Operator::Reshape(inps)
            }
            "Gemm" => {
                let Some(alpha) = node.attribute[0].f else{
                    return Err(GraphError::MissingOperand { operand: String::from("alpha"), operator: node_name, operator_type:String::from("Gemm")});
                };

                let Some(beta) = node.attribute[1].f else{
                    return Err(GraphError::MissingOperand { operand: String::from("beta"), operator: node_name, operator_type:String::from("Gemm")});
                };

                let Some(trans_a) = node.attribute[2].i else{
                    return Err(GraphError::MissingOperand { operand: String::from("trans_a"), operator: node_name, operator_type:String::from("Gemm")});
                };

                let Some(trans_b) = node.attribute[3].i else{
                    return Err(GraphError::MissingOperand { operand: String::from("trans_b"), operator: node_name, operator_type:String::from("Gemm")});
                };

                let attrs: GemmAttributes = GemmAttributes::new(alpha, beta, trans_a, trans_b);

                let mut useful_initializers = Vec::<TensorData>::new();
                let inps;

                parents_names = vec![inputs.remove(0)];

                for inp in inputs {
                    let v = match initializers.iter().find(|tp| *tp.name() == inp) {
                        Some(t) => t,
                        None => {
                            return Err(GraphError::ConversionError(
                                "Unable to filter initializers".to_string(),
                            ))
                        }
                    };

                    let Tensor::Constant(data) = Tensor::from(v.clone()) else{
                        return Err(GraphError::UnexpectedError);
                    };

                    useful_initializers.push(data);
                }

                if let [b, c] = &useful_initializers[..] {
                    inps = GemmInputs::new(b.clone(), c.clone());
                } else {
                    return Err(GraphError::DeconstructError(
                        "Unable to retrieve inputs".to_string(),
                    ));
                }

                Operator::Gemm(inps, attrs)
            }
            "Clip" => {
                let Some(min) = node.attribute[1].f else{
                    return Err(GraphError::MissingOperand { operand: String::from("min"), operator: node_name, operator_type:String::from("Clip")});
                };

                let Some(max) = node.attribute[0].f else{
                    return Err(GraphError::MissingOperand { operand: String::from("max"), operator: node_name, operator_type:String::from("Clip")});
                };

                let attrs: ClipAttributes = ClipAttributes::new(min, max);

                parents_names = vec![inputs.remove(0)];

                Operator::Clip(attrs)
            }
            "Shape" => {
                parents_names = vec![inputs.remove(0)];
                Operator::Shape
            }
            "Gather" => {
                let Some(axes) = node.attribute[0].i else{
                    return Err(GraphError::MissingOperand { operand: String::from("axes"), operator: node_name, operator_type:String::from("Gather")});
                };

                let attrs: GatherAttributes = GatherAttributes::new(axes as usize);

                let mut useful_initializers = Vec::<TensorData>::new();
                let inps;

                parents_names = vec![inputs.remove(0)];

                for inp in inputs {
                    let v = match initializers.iter().find(|tp| *tp.name() == inp) {
                        Some(t) => t,
                        None => {
                            return Err(GraphError::ConversionError(
                                "Unable to filter initializers".to_string(),
                            ))
                        }
                    };

                    let Tensor::Constant(data) = Tensor::from(v.clone()) else{
                        return Err(GraphError::UnexpectedError);
                    };

                    useful_initializers.push(data);
                }

                if let [index] = &useful_initializers[..] {
                    inps = GatherInputs::new(index.clone());
                } else {
                    return Err(GraphError::DeconstructError(
                        "Unable to retrieve inputs".to_string(),
                    ));
                }

                Operator::Gather(inps, attrs)
            }
            "Unsqueeze" => {
                let attrs: UnsqueezeAttributes =
                    UnsqueezeAttributes::new(node.attribute[0].ints[0] as usize);

                parents_names = vec![inputs.remove(0)];

                Operator::Unsqueeze(attrs)
            }
            "Concat" => {
                let Some(axes) = node.attribute[0].i else{
                    return Err(GraphError::MissingOperand { operand: String::from("axes"), operator: node_name, operator_type:String::from("Concat")});
                };
                let attrs: ConcatAttributes = ConcatAttributes::new(axes as usize);

                parents_names = inputs.clone();

                Operator::Concat(attrs)
            }
            _ => return Err(GraphError::UnsupportedOperator),
        };

        let n: NodeIndex = model_graph.add_node(operator);
        parsed_nodes.insert(
            node_name,
            NodeInfo::Intermediate(n.index() as u32, parents_names, node.output),
        );
    }

    for (node_name, parsed_node) in parsed_nodes.iter() {
        let NodeInfo::Intermediate(node_index, parents, children) = parsed_node else {
            continue;
        };
        for p_name in parents {
            let parent_node = parsed_nodes.get(p_name).ok_or(GraphError::ParentNotFound {
                child_name: (*node_name).clone(),
            })?;
            let parent_index = parent_node.index();
            model_graph.add_edge(
                NodeIndex::from(parent_index),
                NodeIndex::from(*node_index),
                None,
            );

            if children
                .iter()
                .any(|child| child.as_str() == output_node_name.as_str())
            {
                model_graph.add_edge(NodeIndex::from(*node_index), output_node, None);
            }
        }
    }

    Ok(model_graph)
}

fn parse_model_io_node(io_value_infos: Vec<ValueInfoProto>) -> Option<(String, Tensor)> {
    io_value_infos
        .into_iter()
        .filter_map(|value_info| {
            let node_name = value_info.name.clone().unwrap_or_default();
            let tensor = Tensor::try_from(value_info).ok()?;
            Some((node_name, tensor))
        })
        .next()
}

#[test]
fn print_parsed_model_test() {
    let path_resnet = "tests/models/resnet18-v2-7.onnx";
    //let path_mobilenet = "tests/models/mobilenetv2-10.onnx";

    let parsed_model = get_parsed_model(path_resnet);

    // println!("\nlength of initializer: {}\n\n", initializer.len());

    // for tensor in initializer{
    //    println!("{:?}\n\n\n", tensor);
    //    break;
    // }

    // println!("{:?}\n\n\n", initializer[0]);

    // for node in parsed_model.unwrap().new_graph.unwrap().node{
    //     println!("{:?}\n", node );
    // }

    let graph = create_graph(parsed_model).unwrap();
    //let pgraph = graph.map(|ni, n| n.name(), |ei, e| e.clone());
    //print_graph(&pgraph, 0.into());
    //println!("{:?}", Dot::with_config(&pgraph, &[Config::EdgeNoLabel]));

    toposort(&graph, None).unwrap().into_iter().for_each(|n| {
        println!("{}", graph[n].name());
    });
}

fn get_parsed_model(path: &str) -> onnx_format::ModelProto {
    let mut buffer = Vec::new();
    let mut file = File::open(path).unwrap();
    file.read_to_end(&mut buffer).unwrap();

    let parsed_model = onnx_format::ModelProto::decode(buffer.as_slice());
    parsed_model.expect("Failed to unwrap parsed_model in get_parsed_model()")
}