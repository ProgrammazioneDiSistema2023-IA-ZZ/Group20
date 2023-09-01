//! Contains the service that can be used to run inference on a model.
//!
//! The service is created using a [`ServiceBuilder`], which can be created using [`ServiceBuilder::new`].
//!
//! The service can be used to run inference on a model using the [`Service::run`] method.
//! if you want to preprocess and postprocess the input and output data yourself.
//! Otherwise, you can use the [`Service::prepare_and_run`] method, which will preprocess the input data and postprocess the output data for you.

mod labels;
pub mod prepare;
pub mod utility;

use ndarray::{ArrayD, IxDyn};
use petgraph::{algo::toposort, Direction};
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::{borrow::BorrowMut, error::Error, ops::ControlFlow, path::PathBuf};
use thiserror::Error;

use crate::{
    graph::{create_graph, GraphError},
    onnx_format::ModelProto,
    operators::{OperationError, Operator},
    providers::{DefaultProvider, Provider},
    tensor::{DynamicTensorData, GraphDimension, TensorData, TensorDataIntoDimensionality},
};

use self::prepare::postprocessing;

#[derive(Error, Debug)]
pub enum ServiceError {
    #[error("The input is invalid: {0}")]
    InvalidInput(Box<dyn Error>),
    #[error("The model could not be translated into an executable graph: {0}")]
    CouldNotTranslateModel(GraphError),
    #[error("An operation failed while inferring the model: {0}")]
    CouldNotExecuteOperation(OperationError),
    #[error("The used model is invalid: {0}")]
    InvalidModel(&'static str),
    #[error("The output node was not found")]
    OutputNodeNotFound,
    #[error("The output shape {actual} is different than expected {expected}")]
    InvalidOutputShape { expected: usize, actual: usize },
    #[error("The output type {actual} is different than expected {expected}")]
    UnexpectedOutputType { expected: String, actual: String },
}

#[derive(Clone, Debug)]
pub struct ServiceBuilder {
    model_path: PathBuf,
    config: Config,
}
pub struct Service {
    model: ModelProto,
    config: Config,
    thread_pool: ThreadPool,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub num_threads: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self { num_threads: 1 }
    }
}

impl ServiceBuilder {
    pub fn new(model_path: PathBuf) -> Self {
        Self {
            model_path,
            config: Config::default(),
        }
    }

    pub fn config(mut self, config: Config) -> Self {
        self.config = config;
        self
    }

    pub fn build(self) -> Result<Service, ServiceError> {
        let model = utility::read_model_proto(self.model_path.as_path());
        Ok(Service::new(model, self.config))
    }
}
pub struct Prediction {
    pub class: String,
    pub probability: f32,
}

pub struct InferenceOutput {
    batch_predictions: ndarray::Array2<f32>,
}

impl InferenceOutput {
    pub fn new(output_tensor: ArrayD<f32>) -> Result<Self, ServiceError> {
        if output_tensor.ndim() != 2 {
            return Err(ServiceError::InvalidOutputShape {
                expected: 2,
                actual: output_tensor.ndim(),
            });
        }

        let output_tensor = output_tensor.into_dimensionality::<ndarray::Ix2>().unwrap();
        let batch_predictions = postprocessing(output_tensor);
        Ok(Self { batch_predictions })
    }

    pub fn get_top_k_predictions(&self, k: usize) -> Vec<Vec<Prediction>> {
        // for each row in the tensor, get the top k predictions
        self.batch_predictions
            .outer_iter()
            .map(|row| self.get_batch_element_top_k_classes(row.to_owned(), k))
            .collect()
    }

    pub fn get_top_k_class_names(&self, k: usize) -> Vec<Vec<String>> {
        let top_classes = self.get_top_k_predictions(k);

        // for each batch element, get the top k classes
        top_classes
            .into_iter()
            .map(|batch_element_top_classes| {
                batch_element_top_classes
                    .into_iter()
                    .map(|prediction| prediction.class)
                    .take(k)
                    .collect()
            })
            .collect()
    }

    fn get_batch_element_top_k_classes(
        &self,
        tensor: ndarray::Array1<f32>,
        k: usize,
    ) -> Vec<Prediction> {
        let mut top_k_classes = tensor
            .iter()
            .enumerate()
            .map(|(i, x)| (i, x))
            .collect::<Vec<_>>();
        top_k_classes.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        top_k_classes.truncate(k);

        top_k_classes
            .into_iter()
            .map(|(i, x)| Prediction {
                class: String::from(labels::IMAGENET_LABELS[i]),
                probability: *x,
            })
            .collect()
    }
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

    /// Preprocesses multiple input data and runs the service on them, using the input parameters and the default execution provider.
    pub fn prepare_and_run(
        &self,
        inputs: Vec<PathBuf>,
        input_parameters: Vec<(String, usize)>,
    ) -> Result<InferenceOutput, ServiceError> {
        self.prepare_and_run_with_provider::<DefaultProvider>(inputs, input_parameters)
    }

    /// Preprocesses multiple input data and runs the service on them, using the input parameters and the given execution provider.
    pub fn prepare_and_run_with_provider<P: Provider>(
        &self,
        inputs: Vec<PathBuf>,
        input_parameters: Vec<(String, usize)>,
    ) -> Result<InferenceOutput, ServiceError> {
        let input_tensor = utility::read_and_prepare_images(inputs.as_slice())?.into_dyn();
        let output_tensor = self.run_with_provider::<P>(input_tensor, input_parameters)?;
        let TensorData::Float(output_tensor) = output_tensor else {
            return Err(ServiceError::UnexpectedOutputType {
                expected: String::from("Float"),
                actual: output_tensor.dtype().to_string(),
            });
        };

        let result = InferenceOutput::new(output_tensor)?;
        Ok(result)
    }

    /// Runs the service on the input data, using the input parameters and the default execution provider.
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
                                OperationError::MissingParamDimension(String::from(name))
                            })?;
                        Ok(param.1)
                    }
                    GraphDimension::Value(dim) => Ok(*dim),
                })
                .collect::<Result<Vec<usize>, OperationError>>()?;

            //check if the input shape matches the required shape
            if required_shape != input_shape {
                return Err(OperationError::UnexpectedInputShape(
                    required_shape.to_vec(),
                    input_shape.to_vec(),
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

            let result =
                ChosenProvider::gemm(thread_pool, matrix_a, matrix_b, matrix_c, attrs.clone())?;
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

            let result = ChosenProvider::batch_norm(
                thread_pool,
                operand,
                scale,
                bias,
                mean,
                var,
                attrs.clone(),
            )?;
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

#[cfg(test)]
mod tests {
    use crate::tensor::{Tensor, TensorData};
    use image::codecs::jpeg::JpegEncoder;
    use prost::Message;

    use crate::onnx_format::TensorProto;
    use crate::service::{Config, Service};
    use std::path::PathBuf;
    use std::{fs::File, io::Read};

    use super::utility::read_model_proto;
    use super::ServiceBuilder;

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
        let input_parameters = vec![(String::from("N"), 1_usize)];
        let most_probable_classes = run_with_image_input(
            "resnet18-v2-7",
            "tests/images/siamese-cat.jpg",
            input_parameters,
        );
        let most_probable_class = most_probable_classes.get(0).expect("No result found");
        assert_eq!(most_probable_class, "Siamese cat, Siamese");
    }

    #[test]
    fn run_mobilenet_with_siamese_cat_image() {
        let input_parameters = vec![];
        let most_probable_classes = run_with_image_input(
            "mobilenetv2-7",
            "tests/images/siamese-cat.jpg",
            input_parameters,
        );
        let most_probable_class = most_probable_classes.get(0).expect("No result found");
        assert_eq!(most_probable_class, "Siamese cat, Siamese");
    }

    #[test]
    fn run_resnet_with_siamese_cat_image_batch() {
        let batch_size = 2_usize;
        let input_parameters = vec![(String::from("N"), batch_size)];
        let most_probable_classes = run_with_image_input(
            "resnet18-v2-7",
            "tests/images/siamese-cat.jpg",
            input_parameters,
        );

        assert_eq!(most_probable_classes.len(), batch_size);

        most_probable_classes
            .iter()
            .take(batch_size)
            .for_each(|most_probable_class| {
                assert_eq!(most_probable_class, "Siamese cat, Siamese");
            });
    }

    fn test_service(model_name: &str, testset_folder: &str) {
        let parsed_model = read_model_proto(format!("tests/models/{}.onnx", model_name));

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
            File::open(format!("tests/testset/{}/output_0.pb", testset_folder)).unwrap();
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

    /// Runs the service on the input data, using the input parameters and the default execution provider.
    /// The input is an image file, which is preprocessed and then passed to the service.
    /// The output is the most probable class for the image.
    /// The batch size is passed as an input parameter, otherwise it defaults to 1.
    /// The output is a vector of strings, one for each batch element.
    fn run_with_image_input(
        model_name: &str,
        image_path: &str,
        input_parameters: Vec<(String, usize)>,
    ) -> Vec<String> {
        // assuming that the first parameter is the batch size
        let default_parameter = (String::from("N"), 1_usize);
        let (_, batch_size) = input_parameters.get(0).unwrap_or(&default_parameter);

        // Use the same image for all the batch elements
        let batch = vec![PathBuf::from(image_path); *batch_size];

        let config = Config { num_threads: 1 };
        let model_path = PathBuf::from(format!("tests/models/{}.onnx", model_name));
        let service = ServiceBuilder::new(model_path)
            .config(config)
            .build()
            .expect("Could not build service");

        let result = service
            .prepare_and_run(batch, input_parameters)
            .expect("Could not infer the model");

        // get the most probable class for each batch element
        result
            .get_top_k_class_names(1)
            .into_iter()
            .flatten()
            .collect()
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
