use crate::onnx_format::ModelProto;
use crate::onnx_format::ValueInfoProto;
use crate::operators::*;

use crate::tensor::{Tensor, TensorData, TensorParametrizedShape};

use petgraph::graph::NodeIndex;
use petgraph::Graph;
use std::cell::RefCell;
use std::collections::HashMap;

use super::GraphError;

type RuntimeGraph = Graph<Operator, RefCell<Option<TensorData>>>;

/// This enum is used to store the information about the nodes of the graph.
/// We will find only Input and Intermediate (not Output) nodes in order to build the graph and to avoid cyclic dependencies.
enum NodeInfo {
    Input(u32),
    Intermediate(u32, Vec<String>, Vec<String>),
}

impl NodeInfo {
    fn index(&self) -> u32 {
        match self {
            NodeInfo::Input(i) => *i,
            NodeInfo::Intermediate(i, _, _) => *i,
        }
    }
}

/// This function creates an executable graph from a parsed ONNX model.
/// It returns a [`RuntimeGraph`] that can be used to execute the ONNX model with the [`Service`](crate::service::Service).
/// It returns a [`GraphError`] if the conversion fails.
pub fn to_exec_graph(model_proto: ModelProto) -> Result<RuntimeGraph, GraphError> {
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

    let (input_node_name, input_shape) = match parse_model_io_node(graph_input) {
        Some((name, shape)) => (name, shape),
        None => return Err(GraphError::InputNodeParsingError),
    };

    let (output_node_name, output_shape) = match parse_model_io_node(graph_output) {
        Some((name, shape)) => (name, shape),
        None => return Err(GraphError::OutputNodeParsingError),
    };

    let input_node = model_graph.add_node(Operator::InputFeed(input_shape));
    let output_node = model_graph.add_node(Operator::OutputCollector(output_shape));

    parsed_nodes.insert(input_node_name, NodeInfo::Input(input_node.index() as u32));
    for node in nodes {
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
                let Some(epsilon) = node.attribute[0].f else {
                    return Err(GraphError::MissingOperand {
                        operand: String::from("epsilon"),
                        operator: node_name,
                        operator_type: String::from("BatchNormalization"),
                    });
                };

                let Some(momentum) = node.attribute[1].f else {
                    return Err(GraphError::MissingOperand {
                        operand: String::from("momentum"),
                        operator: node_name,
                        operator_type: String::from("BatchNormalization"),
                    });
                };

                let Some(spatial) = node.attribute[2].i else {
                    return Err(GraphError::MissingOperand {
                        operand: String::from("spatial"),
                        operator: node_name,
                        operator_type: String::from("BatchNormalization"),
                    });
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

                    let Tensor::Constant(data) = Tensor::from(v.clone()) else {
                        return Err(GraphError::UnexpectedError);
                    };

                    useful_initializers.push(data);
                }

                if let [scale, b, mean, var] = &useful_initializers[..] {
                    inps = BatchNormInits::new(scale.clone(), b.clone(), mean.clone(), var.clone());
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

                    let Tensor::Constant(data) = Tensor::from(v.clone()) else {
                        return Err(GraphError::UnexpectedError);
                    };

                    useful_initializers.push(data);
                }

                if let [weights, bias] = &useful_initializers[..] {
                    inps = ConvInits::new(weights.clone(), Some(bias.clone()));
                } else if let [weights] = &useful_initializers[..] {
                    inps = ConvInits::new(weights.clone(), None);
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

                    let Tensor::Constant(data) = Tensor::from(v.clone()) else {
                        return Err(GraphError::UnexpectedError);
                    };

                    useful_initializers.push(data);
                }

                if let [shape] = &useful_initializers[..] {
                    inps = ReshapeInits::new(shape.clone());
                } else {
                    return Err(GraphError::DeconstructError(
                        "Unable to retrieve inputs".to_string(),
                    ));
                }

                Operator::Reshape(inps)
            }
            "Gemm" => {
                let Some(alpha) = node.attribute[0].f else {
                    return Err(GraphError::MissingOperand {
                        operand: String::from("alpha"),
                        operator: node_name,
                        operator_type: String::from("Gemm"),
                    });
                };

                let Some(beta) = node.attribute[1].f else {
                    return Err(GraphError::MissingOperand {
                        operand: String::from("beta"),
                        operator: node_name,
                        operator_type: String::from("Gemm"),
                    });
                };

                let Some(trans_a) = node.attribute[2].i else {
                    return Err(GraphError::MissingOperand {
                        operand: String::from("trans_a"),
                        operator: node_name,
                        operator_type: String::from("Gemm"),
                    });
                };

                let Some(trans_b) = node.attribute[3].i else {
                    return Err(GraphError::MissingOperand {
                        operand: String::from("trans_b"),
                        operator: node_name,
                        operator_type: String::from("Gemm"),
                    });
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

                    let Tensor::Constant(data) = Tensor::from(v.clone()) else {
                        return Err(GraphError::UnexpectedError);
                    };

                    useful_initializers.push(data);
                }

                if let [b, c] = &useful_initializers[..] {
                    inps = GemmInits::new(b.clone(), c.clone());
                } else {
                    return Err(GraphError::DeconstructError(
                        "Unable to retrieve inputs".to_string(),
                    ));
                }

                Operator::Gemm(inps, attrs)
            }
            "Clip" => {
                let Some(min) = node.attribute[1].f else {
                    return Err(GraphError::MissingOperand {
                        operand: String::from("min"),
                        operator: node_name,
                        operator_type: String::from("Clip"),
                    });
                };

                let Some(max) = node.attribute[0].f else {
                    return Err(GraphError::MissingOperand {
                        operand: String::from("max"),
                        operator: node_name,
                        operator_type: String::from("Clip"),
                    });
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
                let Some(axes) = node.attribute[0].i else {
                    return Err(GraphError::MissingOperand {
                        operand: String::from("axes"),
                        operator: node_name,
                        operator_type: String::from("Gather"),
                    });
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

                    let Tensor::Constant(data) = Tensor::from(v.clone()) else {
                        return Err(GraphError::UnexpectedError);
                    };

                    useful_initializers.push(data);
                }

                if let [index] = &useful_initializers[..] {
                    inps = GatherInits::new(index.clone());
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
                let Some(axes) = node.attribute[0].i else {
                    return Err(GraphError::MissingOperand {
                        operand: String::from("axes"),
                        operator: node_name,
                        operator_type: String::from("Concat"),
                    });
                };
                let attrs: ConcatAttributes = ConcatAttributes::new(axes as usize);

                parents_names = inputs.clone();

                Operator::Concat(attrs)
            }
            op => return Err(GraphError::UnsupportedOperator(String::from(op))),
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
                RefCell::new(None),
            );

            // Add edge to the model output node if the current node generates the output
            if children
                .iter()
                .any(|child| child.as_str() == output_node_name.as_str())
            {
                model_graph.add_edge(
                    NodeIndex::from(*node_index),
                    output_node,
                    RefCell::new(None),
                );
            }
        }
    }

    Ok(model_graph)
}

fn parse_model_io_node(
    io_value_infos: Vec<ValueInfoProto>,
) -> Option<(String, TensorParametrizedShape)> {
    io_value_infos
        .iter()
        .find_map(|value_info| {
            let node_name = value_info.name.clone().unwrap_or_default();
            let tensor = Tensor::try_from(value_info.clone()).ok()?;
            if !tensor.is_parametrized_io() {
                return None;
            }
            if let Tensor::InOut(shape, _) = tensor {
                return Some((node_name, shape));
            }
            None
        })
        .or_else(|| {
            let value_info = io_value_infos[0].clone();
            let node_name = value_info.name.clone().unwrap_or_default();
            let tensor = Tensor::try_from(value_info).ok()?;
            if let Tensor::InOut(shape, _) = tensor {
                return Some((node_name, shape));
            }
            None
        })
}
#[cfg(test)]
mod tests {

    use super::*;
    use crate::onnx_format::{self};
    use prost::Message;
    use std::{fs::File, io::Read};

    #[test]
    fn parsed_model_test_node_count_resnet() {
        let path_resnet = "tests/models/resnet18-v2-7.onnx";
        let node_count_resnet = run_parsed_model(path_resnet);
        assert_eq!(node_count_resnet, 71);
    }

    #[test]
    fn parsed_model_test_node_count_mobilenet() {
        let path_mobilenet = "tests/models/mobilenetv2-7.onnx";
        let node_count_mobilenet = run_parsed_model(path_mobilenet);
        assert_eq!(node_count_mobilenet, 157);
    }

    fn get_parsed_model(path: &str) -> onnx_format::ModelProto {
        let mut buffer = Vec::new();
        let mut file = File::open(path).unwrap();
        file.read_to_end(&mut buffer).unwrap();

        let parsed_model = onnx_format::ModelProto::decode(buffer.as_slice());
        parsed_model.expect("Failed to unwrap parsed_model in get_parsed_model()")
    }

    fn run_parsed_model(path: &str) -> usize {
        let parsed_model = get_parsed_model(path);
        let graph = to_exec_graph(parsed_model).unwrap();

        graph.node_count()
    }
}
