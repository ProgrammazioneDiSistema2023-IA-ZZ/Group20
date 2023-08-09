use crate::onnx_format;
use crate::operators::*;

use crate::tensor::{Tensor, TensorData};

use petgraph::algo::toposort;
use petgraph::graph::NodeIndex;
use petgraph::visit::GraphProp;
use petgraph::Graph;
use prost::Message;
use std::collections::HashMap;
use std::{fs::File, io::Read};

use super::GraphError;

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

pub fn create_graph() -> Result<(), GraphError> {
    let path_resnet = "tests/models/resnet18-v2-7.onnx";
    //let path_mobilenet = "tests/models/mobilenetv2-10.onnx";

    let parsed_model = get_parsed_model(path_resnet);

    let graph_proto = match parsed_model.graph {
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

    let mut input_node_name;
    let input_node = graph_input
        .into_iter()
        .filter_map(|vip| {
            input_node_name = vip.name.unwrap();
            Tensor::try_from(vip).ok()
        })
        .next();

    
    let output_node = graph_output
        .into_iter()
        .filter_map(|vip| Tensor::try_from(vip).ok())
        .next();

    println!("{:?}", input_node);

    let mut map: HashMap<String, NodeInfo> = HashMap::new();

    let mut new_graph: Graph<Operator, Option<TensorData>> =
        Graph::<Operator, Option<TensorData>>::new();

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
        let parents_name: Vec<String>;

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

                parents_name = vec![inputs.remove(0)];

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

                parents_name = vec![inputs.remove(0)];

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
                let mut useful_initializers = Vec::<TensorData>::new();
                let inps;

                parents_name = vec![inputs.remove(0)];

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

                parents_name = vec![inputs.remove(0)];

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

                parents_name = vec![inputs.remove(0)];

                Operator::Clip(attrs)
            }
            "Shape" => {
                parents_name = vec![inputs.remove(0)];
                Operator::Shape
            }
            "Gather" => {
                let Some(axes) = node.attribute[0].i else{
                    return Err(GraphError::MissingOperand { operand: String::from("axes"), operator: node_name, operator_type:String::from("Gather")});
                };

                let attrs: GatherAttributes = GatherAttributes::new(axes as usize);

                let mut useful_initializers = Vec::<TensorData>::new();
                let inps;

                parents_name = vec![inputs.remove(0)];

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

                parents_name = vec![inputs.remove(0)];

                Operator::Unsqueeze(attrs)
            }
            "Concat" => {
                let Some(axes) = node.attribute[0].i else{
                    return Err(GraphError::MissingOperand { operand: String::from("axes"), operator: node_name, operator_type:String::from("Concat")});
                };
                let attrs: ConcatAttributes = ConcatAttributes::new(axes as usize);

                parents_name = inputs.clone();

                Operator::Concat(attrs)
            }
            _ => return Err(GraphError::UnsupportedOperator),
        };

        let n = new_graph.add_node(operator);
        let ni = NodeInfo::new(parents_name, n.index());
        map.insert(node_name, ni);
    }

    for (n_name, n_info) in map.iter() {
        let parents = &n_info.parents_name;
        let n_index = n_info.node_index;
        for p_name in parents {
            let p_node_info = map.get(p_name).ok_or(GraphError::ParentNotFound {
                child_name: (*n_name).clone(),
            })?;
            let p_index = p_node_info.node_index;
            new_graph.add_edge(NodeIndex::new(n_index), NodeIndex::new(p_index), None);
        }
    }

    // let _toposort = toposort(&new_graph, None).unwrap().into_iter().rev().for_each(|n| {
    //     println!("{:?}\n", new_graph[n].name());
    // });

    Ok(())
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

    // for node in parsed_model.unwrap().new_graph.unwrap().node{
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
