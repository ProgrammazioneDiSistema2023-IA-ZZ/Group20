use ndarray::{ArrayD, IxDyn};
use petgraph::{algo::toposort, Direction};
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::{borrow::BorrowMut, ops::ControlFlow};
use thiserror::Error;

use crate::{
    graph::{create_graph, GraphError},
    onnx_format::ModelProto,
    operators::{OperationError, Operator},
    providers::{DefaultProvider, Provider},
    tensor::{DynamicTensorData, GraphDimension, TensorData, TensorDataIntoDimensionality},
};

#[derive(Error, Debug)]
pub enum ServiceError {
    #[error("The model could not be translated into an executable graph: {0}")]
    CouldNotTranslateModel(GraphError),
    #[error("An operation failed while inferring the model: {0}")]
    CouldNotExecuteOperation(OperationError),
    #[error("The used model is invalid: {0}")]
    InvalidModel(&'static str),
    #[error("The output node was not found")]
    OutputNodeNotFound,
}

pub struct Service {
    model: ModelProto,
    config: Config,
    thread_pool: ThreadPool,
}

impl Service {
    pub fn new(model: ModelProto, config: Config) -> Self {
        let n_threads = config.num_threads;
        Self {
            model,
            config,
            thread_pool: ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()
                .expect("Unable to create ThreadPool"),
        }
    }

    pub fn current_config(&self) -> Config {
        self.config.clone()
    }

    /// Runs the service on the input data, using the input parameters and the default execution provider (naive).
    pub fn run(
        &self,
        input: ArrayD<f32>,
        input_parameters: Vec<(String, usize)>,
    ) -> Result<TensorData, ServiceError> {
        self.run_with_provider::<DefaultProvider>(input, input_parameters)
    }

    /// Runs the service on the input data, using the input parameters and the chosen execution provider.
    pub fn run_with_provider<P>(
        &self,
        input: ArrayD<f32>,
        input_parameters: Vec<(String, usize)>,
    ) -> Result<TensorData, ServiceError>
    where
        P: Provider,
    {
        let mut final_output = None;
        let mut operations_graph =
            create_graph(self.model.clone()).map_err(ServiceError::CouldNotTranslateModel)?;
        let ordered_operation_list = toposort(&operations_graph, None)
            .map_err(|_| ServiceError::InvalidModel("The model's graph is not a DAG"))?;

        let execution_result = ordered_operation_list.into_iter()
            .try_for_each(|node| {
                let incoming_data = operations_graph
                    .edges_directed(node, Direction::Incoming)
                    // TODO: wait for the input to be ready for execution in a multithreaded environment
                    .map(|e| {
                        e.weight()
                            .borrow()
                            .clone()
                            .expect("Trying to get data as an input for an operation, but the data is being used by another operation")
                    })
                    .collect::<Vec<TensorData>>();

                // if the incoming data is empty, it means that the current node is an input node
                // and we need to pass the input data to it
                let incoming_data = if incoming_data.is_empty() {
                    vec![TensorData::Float(input.clone())]
                } else {
                    incoming_data
                };

                let operation_result = execute_operation::<P>(incoming_data, &input_parameters, &operations_graph[node], &self.thread_pool);
                let outgoing_data = match operation_result {
                    Ok(res) => res,
                    Err(e) => return ControlFlow::Break(e),
                };

                // for each outgoing edge, set the data to the outgoing data
                operations_graph
                    .borrow_mut()
                    .edges_directed(node, Direction::Outgoing)
                    .for_each(|e| {
                        e.weight().replace(Some(outgoing_data.clone()));
                    });

                // check if the current node is an output node
                if let Operator::OutputCollector(_) = operations_graph[node] {
                    final_output = Some(outgoing_data);
                }

                ControlFlow::Continue(())
            });

        match execution_result {
            ControlFlow::Continue(_) => (),
            ControlFlow::Break(e) => return Err(ServiceError::CouldNotExecuteOperation(e)),
        };
        final_output.ok_or(ServiceError::OutputNodeNotFound)
    }
}

fn execute_operation<ChosenProvider>(
    inputs: Vec<TensorData>,
    input_parameters: &[(String, usize)],
    operator: &Operator,
    thread_pool: &ThreadPool,
) -> Result<TensorData, OperationError>
where
    ChosenProvider: Provider,
{
    match operator {
        Operator::InputFeed(required_shape) | Operator::OutputCollector(required_shape) => {
            let input_shape = inputs[0].shape();
            if required_shape.is_empty() {
                return Err(OperationError::WrongShape(
                    String::from("Empty"),
                    String::from("Non-empty"),
                ));
            }
            if required_shape.len() != input_shape.len() {
                return Err(OperationError::WrongDim(
                    required_shape.len(),
                    inputs[0].shape().len(),
                ));
            }
            //check if the required shape is parameterized and if so, replace the parameters with the values from the input_parameters
            let required_shape = required_shape
                .iter()
                .map(|dim| match dim {
                    GraphDimension::Parameter(name) => {
                        let param = input_parameters
                            .iter()
                            .find(|(param_name, _)| param_name == name)
                            .ok_or_else(|| {
                                OperationError::UnexpectedShape(
                                    format!("Input parameter `{}` not found", name),
                                    format!("Input parameter `{}` should be passed", name),
                                )
                            })?;
                        Ok(param.1)
                    }
                    GraphDimension::Value(dim) => Ok(*dim),
                })
                .collect::<Result<Vec<usize>, OperationError>>()?;

            //check if the input shape matches the required shape
            if required_shape != input_shape {
                return Err(OperationError::UnexpectedShape(
                    format!("{:?}", required_shape),
                    format!("{:?}", input_shape),
                ));
            }

            Ok(inputs[0].clone())
        }

        Operator::Convolution(inits, attrs) => {
            let TensorData::Float(operand) = inputs[0].clone() else {
                return Err(OperationError::InvalidTensorType(
                    operator.name(),
                    String::from("x"),
                ));
            };
            let TensorData::Float(weights) = inits.weights.clone() else {
                return Err(OperationError::InvalidTensorType(
                    operator.name(),
                    String::from("weights"),
                ));
            };

            let bias = inits
                .bias
                .clone()
                .map(|b| match b {
                    TensorData::Float(b) => {
                        if b.ndim() != 1 {
                            return Err(OperationError::WrongDim(1, b.ndim()));
                        };
                        Ok(b.into_dimensionality::<ndarray::Ix1>().unwrap())
                    }
                    _ => Err(OperationError::InvalidTensorType(
                        operator.name(),
                        String::from("bias"),
                    )),
                })
                .transpose()?;

            let result = ChosenProvider::conv(thread_pool, operand, weights, bias, attrs.clone())?;
            let tensor = TensorData::Float(result);
            Ok(tensor)
        }
        Operator::Clip(attrs) => {
            let TensorData::Float(operand) = inputs[0].clone() else {
                return Err(OperationError::InvalidTensorType(
                    operator.name(),
                    String::from("X"),
                ));
            };
            let result = ChosenProvider::clip(thread_pool, operand, attrs.clone());
            let tensor = TensorData::Float(result);
            Ok(tensor)
        }
        Operator::Add => {
            let TensorData::Float(lhs) = inputs[0].clone() else {
                return Err(OperationError::InvalidTensorType(
                    operator.name(),
                    String::from("A"),
                ));
            };
            let TensorData::Float(rhs) = inputs[1].clone() else {
                return Err(OperationError::InvalidTensorType(
                    operator.name(),
                    String::from("B"),
                ));
            };
            let result = ChosenProvider::add(thread_pool, lhs, rhs)?;
            let tensor = TensorData::Float(result);
            Ok(tensor)
        }
        Operator::Shape => {
            let TensorData::Float(operand) = inputs[0].clone() else {
                return Err(OperationError::InvalidTensorType(
                    operator.name(),
                    String::from("X"),
                ));
            };
            let result = ChosenProvider::shape(thread_pool, operand);
            let tensor = TensorData::Int64(result);
            Ok(tensor)
        }
        Operator::Gather(inits, attrs) => {
            //TODO: check for the input and output integer types according to the ONNX specs
            let operand = match &inputs[0] {
                TensorData::Int64(x) => x.map(|e| *e as usize),
                TensorData::Int32(x) => x.map(|e| *e as usize),
                _ => {
                    return Err(OperationError::InvalidTensorType(
                        operator.name(),
                        String::from("X"),
                    ))
                }
            };

            let index = match inits.index.clone() {
                TensorData::Int64(i) => i
                    .into_dimensionality::<ndarray::Ix0>()
                    .unwrap()
                    .into_scalar() as usize,
                TensorData::Int32(i) => i
                    .into_dimensionality::<ndarray::Ix0>()
                    .unwrap()
                    .into_scalar() as usize,
                _ => {
                    return Err(OperationError::InvalidTensorType(
                        operator.name(),
                        String::from("index"),
                    ))
                }
            };

            let result = ChosenProvider::gather(thread_pool, operand, index, attrs.clone())?;
            let tensor = TensorData::Int64(result.map(|e| *e as i64).into_dyn());
            Ok(tensor)
        }
        Operator::Unsqueeze(attrs) => {
            let operand = match &inputs[0] {
                TensorData::Int64(x) => x.map(|e| *e as usize),
                TensorData::Int32(x) => x.map(|e| *e as usize),
                _ => {
                    return Err(OperationError::InvalidTensorType(
                        operator.name(),
                        String::from("X"),
                    ))
                }
            };

            let result = ChosenProvider::unsqueeze(thread_pool, operand, attrs.clone())?;
            let tensor = TensorData::Int64(result.map(|e| *e as i64).into_dyn());
            Ok(tensor)
        }
        Operator::Concat(attrs) => {
            let operands = inputs
                .clone()
                .into_iter()
                .map(|input| input.into_dimensionality::<IxDyn>())
                .collect::<Vec<ArrayD<_>>>();
            //TODO: fix dynamic operation. Maybe use a byte array for all the operations like this that can take any input type
            let result = ChosenProvider::concat::<f32>(thread_pool, operands, attrs.clone())?;
            let tensor = TensorData::new_dyn(result);
            Ok(tensor)
        }
        Operator::GlobalAveragePool => {
            let TensorData::Float(operand) = inputs[0].clone() else {
                return Err(OperationError::InvalidTensorType(
                    operator.name(),
                    String::from("X"),
                ));
            };
            let result = ChosenProvider::global_average_pool(thread_pool, operand)?;
            let tensor = TensorData::Float(result);
            Ok(tensor)
        }
        Operator::Reshape(inits) => {
            let TensorData::Float(operand) = inputs[0].clone() else {
                return Err(OperationError::InvalidTensorType(
                    operator.name(),
                    String::from("X"),
                ));
            };
            let TensorData::Int64(shape) = inits.shape.clone() else {
                return Err(OperationError::InvalidTensorType(
                    operator.name(),
                    String::from("X"),
                ));
            };

            let result = ChosenProvider::reshape(thread_pool, operand, shape)?;
            let tensor = TensorData::Float(result);
            Ok(tensor)
        }
        Operator::Gemm(inits, attrs) => {
            let TensorData::Float(matrix_a) = inputs[0].clone() else {
                return Err(OperationError::InvalidTensorType(
                    operator.name(),
                    String::from("A"),
                ));
            };
            let TensorData::Float(matrix_b) = inits.b.clone() else {
                return Err(OperationError::InvalidTensorType(
                    operator.name(),
                    String::from("B"),
                ));
            };
            let TensorData::Float(matrix_c) = inits.c.clone() else {
                return Err(OperationError::InvalidTensorType(
                    operator.name(),
                    String::from("C"),
                ));
            };

            let result = ChosenProvider::gemm(thread_pool, matrix_a, matrix_b, matrix_c, attrs.clone())?;
            let tensor = TensorData::Float(result);
            Ok(tensor)
        }
        Operator::MaxPool(attrs) => {
            let TensorData::Float(operand) = inputs[0].clone() else {
                return Err(OperationError::InvalidTensorType(
                    operator.name(),
                    String::from("X"),
                ));
            };
            let result = ChosenProvider::max_pool(thread_pool, operand, attrs.clone())?;
            let tensor = TensorData::Float(result);
            Ok(tensor)
        }
        Operator::BatchNorm(inits, attrs) => {
            //TODO: check for doubles as well
            let TensorData::Float(operand) = inputs[0].clone() else {
                return Err(OperationError::InvalidTensorType(
                    operator.name(),
                    String::from("X"),
                ));
            };
            let TensorData::Float(scale) = inits.scale.clone() else {
                return Err(OperationError::InvalidTensorType(
                    operator.name(),
                    String::from("scale"),
                ));
            };
            let TensorData::Float(bias) = inits.bias.clone() else {
                return Err(OperationError::InvalidTensorType(
                    operator.name(),
                    String::from("bias"),
                ));
            };
            let TensorData::Float(mean) = inits.mean.clone() else {
                return Err(OperationError::InvalidTensorType(
                    operator.name(),
                    String::from("mean"),
                ));
            };
            let TensorData::Float(var) = inits.var.clone() else {
                return Err(OperationError::InvalidTensorType(
                    operator.name(),
                    String::from("var"),
                ));
            };

            let result =
                ChosenProvider::batch_norm(thread_pool, operand, scale, bias, mean, var, attrs.clone())?;
            let tensor = TensorData::Float(result);
            Ok(tensor)
        }
        Operator::ReLU => {
            let TensorData::Float(operand) = inputs[0].clone() else {
                return Err(OperationError::InvalidTensorType(
                    operator.name(),
                    String::from("X"),
                ));
            };
            let result = ChosenProvider::relu(thread_pool, operand);
            let tensor = TensorData::Float(result);
            Ok(tensor)
        }
    }
}

#[derive(Clone, Copy)]
pub struct Config {
    pub num_threads: usize,
}

#[cfg(test)]
mod tests {
    use crate::providers::ParProvider;
    use crate::tensor::{Tensor, TensorData};
    use image::codecs::jpeg::JpegEncoder;
    use prost::Message;

    use crate::onnx_format::TensorProto;
    use crate::prepare::{postprocessing, postprocessing_top_k};
    use crate::{
        onnx_format::ModelProto,
        service::{Config, Service},
    };
    use std::{fs::File, io::Read};

    #[test]
    fn run_mobilenet_with_testset() {
        test_service("mobilenetv2-7", "mobilenet");
    }

    #[test]
    fn run_resnet_with_testset() {
        test_service("resnet18-v2-7", "resnet");
    }

    #[test]
    fn run_resnet_with_siamese_cat_image() {
        let most_probable_class =
            run_with_image_input("resnet18-v2-7", "tests/images/siamese-cat.jpg");
        assert_eq!(most_probable_class, "Siamese cat, Siamese");
    }

    #[test]
    fn run_mobilenet_with_siamese_cat_image() {
        let most_probable_class =
            run_with_image_input("mobilenetv2-7", "tests/images/siamese-cat.jpg");
        assert_eq!(most_probable_class, "Siamese cat, Siamese");
    }

    fn test_service(model_name: &str, testset_folder: &str) {
        let parsed_model = read_model_proto(&format!("tests/models/{}.onnx", model_name));

        let config = Config { num_threads: 1 };
        let service = Service::new(parsed_model, config);
        let input_parameters = vec![(String::from("N"), 1_usize)];

        //read input from test.pb as an ArrayD of shape [1, 3, 224, 224]
        let input = read_testset(&format!("tests/testset/{}/input_0.pb", testset_folder));

        // convert the TensorData to an ArrayD
        let Tensor::Constant(TensorData::Float(input)) = input else {
            panic!("Invalid input type")
        };

        let result = service.run(input, input_parameters);
        let TensorData::Float(result) = result.unwrap() else {
            panic!("Invalid result type")
        };
        let result = result
            .into_dimensionality::<ndarray::Ix2>()
            .expect("Invalid result dimensionality");

        let mut expected_output_buffer = Vec::new();
        let mut expected_output_file =
            File::open(format!("tests/testset/{}//output_0.pb", testset_folder)).unwrap();
        expected_output_file
            .read_to_end(&mut expected_output_buffer)
            .unwrap();
        let expected_output = TensorProto::decode(expected_output_buffer.as_slice()).unwrap();

        let Tensor::Constant(TensorData::Float(expected_output)) = Tensor::from(expected_output)
        else {
            panic!("Invalid expected output type")
        };
        let expected_output = expected_output
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();

        //check if the result is the same as the expected output
        let err = (result - expected_output).mapv(|x| x.abs()).mean().unwrap();
        assert!(err < 1e-4);
    }

    fn run_with_image_input(model_name: &str, image_path: &str) -> String {
        use crate::prepare::preprocessing;

        let image = image::open(image_path).unwrap();
        let preprocessed_image = preprocessing(image);

        //save the preprocessed tensor as an image file
        //preprocessed_image_to_file(&preprocessed_image, "tests/preprocessed_cat.jpg");

        let model_proto = read_model_proto(&format!("tests/models/{}.onnx", model_name));
        let config = Config { num_threads: 4 };
        let service = Service::new(model_proto, config);
        let input_parameters = vec![(String::from("N"), 1_usize)];

        let result = service
            .run_with_provider::<ParProvider>(preprocessed_image.into_dyn(), input_parameters)
            .unwrap();
        let TensorData::Float(result) = result else {
            panic!("Invalid result type")
        };
        let result = result.into_dimensionality::<ndarray::Ix2>().unwrap();
        let result = postprocessing(result);

        //println!("Top 5 predictions:");
        //for (i, (class, prob)) in  postprocessing_top_k(result, 5).iter().enumerate() {
        //    println!("{}. class: {}, probability: {}", i + 1, class, prob);
        //}

        let result = postprocessing_top_k(result, 1);
        let [top_result, ..] = result.as_slice() else {
            panic!("The final output should have at least one element")
        };
        let (top_class, _probability) = top_result;
        top_class.clone()
    }

    fn read_model_proto(path: &str) -> ModelProto {
        let mut buffer = Vec::new();
        let mut file = File::open(path).unwrap();
        file.read_to_end(&mut buffer).unwrap();

        ModelProto::decode(buffer.as_slice()).unwrap()
    }

    fn read_testset(path: &str) -> Tensor {
        let mut buffer = Vec::new();
        let mut file = File::open(path).unwrap();
        file.read_to_end(&mut buffer).unwrap();

        Tensor::from(TensorProto::decode(buffer.as_slice()).unwrap())
    }

    #[allow(dead_code)]
    fn preprocessed_image_to_file(tensor: &ndarray::Array4<f32>, path: &str) {
        let output_file = File::create(path).unwrap();
        let mut encoder = JpegEncoder::new(output_file);
        encoder
            .encode(
                tensor
                    .clone()
                    .remove_axis(ndarray::Axis(0))
                    .permuted_axes([2, 0, 1])
                    .mapv(|x| (x * 255.0) as u8)
                    .into_raw_vec()
                    .as_slice(),
                224,
                224,
                image::ColorType::Rgb8,
            )
            .unwrap();
    }
}
