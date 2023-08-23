use ndarray::{ArrayD, IxDyn};
use petgraph::{algo::toposort, Direction};
use std::{borrow::BorrowMut, ops::ControlFlow};

use crate::{
    graph::create_graph,
    onnx_format::ModelProto,
    operators::{OperationError, Operator},
    providers::{DefaultProvider, Provider},
    tensor::{DynamicTensorData, TensorData, TensorDataIntoDimensionality},
};

pub struct Service {
    model: ModelProto,
    config: Config,
}

impl Service {
    pub fn new(model: ModelProto, config: Config) -> Self {
        Self { model, config }
    }

    /// Runs the service on the input data, using the input parameters and the default execution provider (naive).
    pub fn run(
        &self,
        input: ArrayD<f32>,
        input_parameters: Vec<(String, f32)>,
    ) -> Result<TensorData, &'static str> {
        self.run_with_provider::<DefaultProvider>(input, input_parameters)
    }

    /// Runs the service on the input data, using the input parameters and the chosen execution provider.
    pub fn run_with_provider<P>(
        &self,
        input: ArrayD<f32>,
        input_parameters: Vec<(String, f32)>,
    ) -> Result<TensorData, &'static str>
    where
        P: Provider,
    {
        let mut operations_graph = create_graph(self.model.clone()).unwrap();

        let mut final_output = None;
        let ordered_operation_list =
            toposort(&operations_graph, None).map_err(|_| "The graph is not a DAG")?;

        ordered_operation_list.into_iter()
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

                let operation_result = execute_operation::<P>(incoming_data, &operations_graph[node]).map_err(|e| e.to_string());
                let Ok(outgoing_data) = operation_result else { return ControlFlow::Break(operation_result)};

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
        final_output.ok_or("No output node found")
    }
}

fn execute_operation<ChosenProvider>(
    inputs: Vec<TensorData>,
    operator: &Operator,
) -> Result<TensorData, OperationError>
where
    ChosenProvider: Provider,
{
    match operator {
        //TODO: check for the shape correctness before passing the incoming data for InputFeed and OutputCollector
        Operator::InputFeed(_shape) => Ok(inputs[0].clone()),
        Operator::OutputCollector(_shape) => Ok(inputs[0].clone()),

        Operator::Convolution(inits, attrs) => {

            let tensor = ChosenProvider::conv(inputs[0].clone(), inits.weights.clone(), inits.bias.clone(), attrs.clone())?;
            Ok(tensor)
        }
        Operator::Clip(attrs) => {
            let TensorData::Float(operand) = inputs[0].clone() else {todo!("clip: invalid input tensor type")};
            let result = ChosenProvider::clip(operand, attrs.clone());
            let tensor = TensorData::Float(result);
            Ok(tensor)
        }
        Operator::Add => {
            let TensorData::Float(lhs) = inputs[0].clone() else {todo!("add: invalid input tensor type")};
            let TensorData::Float(rhs) = inputs[1].clone() else {todo!("add: invalid input tensor type")};
            let result = ChosenProvider::add(lhs, rhs)?;
            let tensor = TensorData::Float(result);
            Ok(tensor)
        }
        Operator::Shape => {
            let TensorData::Float(operand) = inputs[0].clone() else {todo!("shape: invalid input tensor type")};
            let result = ChosenProvider::shape(operand);
            let tensor = TensorData::Int64(result);
            Ok(tensor)
        }
        Operator::Gather(inits, attrs) => {
            //TODO: check for the input and output integer types according to the ONNX specs
            let operand = match &inputs[0] {
                TensorData::Int64(x) => x.map(|e| *e as usize),
                TensorData::Int32(x) => x.map(|e| *e as usize),
                _ => todo!("gather: invalid input tensor type"),
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
                _ => todo!("gather: invalid index tensor type"),
            };

            let result = ChosenProvider::gather(operand, index, attrs.clone())?;
            let tensor = TensorData::Int64(result.map(|e| *e as i64).into_dyn());
            Ok(tensor)
        }
        Operator::Unsqueeze(attrs) => {
            let operand = match &inputs[0] {
                TensorData::Int64(x) => x.map(|e| *e as usize),
                TensorData::Int32(x) => x.map(|e| *e as usize),
                _ => todo!("gather: invalid input tensor type"),
            };

            let result = ChosenProvider::unsqueeze(operand, attrs.clone())?;
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
            let result = ChosenProvider::concat::<f32>(operands, attrs.clone())?;
            let tensor = TensorData::new_dyn(result);
            Ok(tensor)
        }
        Operator::GlobalAveragePool => {
            let TensorData::Float(operand) = inputs[0].clone() else {todo!("global_average_pool: invalid input tensor type")};
            let result = ChosenProvider::global_average_pool(operand)?;
            let tensor = TensorData::Float(result);
            Ok(tensor)
        }
        Operator::Reshape(inits) => {
            let TensorData::Float(operand) = inputs[0].clone() else {todo!("reshape: invalid input tensor type")};
            let TensorData::Int64(shape) = inits.shape.clone() else {todo!("reshape: invalid shape tensor type")};

            let result = ChosenProvider::reshape(operand, shape)?;
            let tensor = TensorData::Float(result);
            Ok(tensor)
        }
        Operator::Gemm(inits, attrs) => {
            let TensorData::Float(matrix_a) = inputs[0].clone() else {todo!("gemm: invalid input tensor type")};
            let TensorData::Float(matrix_b) = inits.b.clone() else {todo!("gemm: invalid B matrix tensor type")};
            let TensorData::Float(matrix_c) = inits.c.clone() else {todo!("gemm: invalid C matrix tensor type")};

            let result = ChosenProvider::gemm(matrix_a, matrix_b, matrix_c, attrs.clone())?;
            let tensor = TensorData::Float(result);
            Ok(tensor)
        }
        Operator::MaxPool(attrs) => {
            let TensorData::Float(operand) = inputs[0].clone() else {todo!("max_pool: invalid input tensor type")};
            let result = ChosenProvider::max_pool(operand, attrs.clone())?;
            let tensor = TensorData::Float(result);
            Ok(tensor)
        }
        Operator::BatchNorm(inits, attrs) => {
            //TODO: check for doubles as well
            let TensorData::Float(operand) = inputs[0].clone() else {todo!("batch_norm: invalid input tensor type")};
            let TensorData::Float(scale) = inits.scale.clone() else {todo!("batch_norm: invalid scale tensor type")};
            let TensorData::Float(bias) = inits.bias.clone() else {todo!("batch_norm: invalid bias tensor type")};
            let TensorData::Float(mean) = inits.mean.clone() else {todo!("batch_norm: invalid mean tensor type")};
            let TensorData::Float(var) = inits.var.clone() else {todo!("batch_norm: invalid var tensor type")};

            let result =
                ChosenProvider::batch_norm(operand, scale, bias, mean, var, attrs.clone())?;
            let tensor = TensorData::Float(result);
            Ok(tensor)
        }
        Operator::ReLU => {
            let TensorData::Float(operand) = inputs[0].clone() else {todo!("relu: invalid input tensor type")};
            let result = ChosenProvider::relu(operand);
            let tensor = TensorData::Float(result);
            Ok(tensor)
        }
    }
}

pub struct Config {
    pub num_threads: usize,
}

#[cfg(test)]
mod tests {
    use csv::WriterBuilder;
    use image::codecs::jpeg::JpegEncoder;
    use ndarray_csv::Array2Writer;
    use prost::Message;

    use crate::onnx_format::TensorProto;
    use crate::prepare::{postprocessing, postprocessing_top_k};
    use crate::tensor::{Tensor, TensorData};
    use crate::{
        onnx_format::ModelProto,
        service::{Config, Service},
    };
    use std::{fs::File, io::Read};

    #[test]
    fn test_service() {
        use crate::tensor::{Tensor, TensorData};
        use prost::Message;
        use std::fs::File;
        use std::io::Read;

        let parsed_model = read_model_proto("tests/models/mobilenetv2-7.onnx");

        let config = Config { num_threads: 1 };
        let service = Service::new(parsed_model, config);

        //read input from test.pb as an ArrayD of shape [1, 3, 224, 224]
        let input = read_testset("tests/testset/mobilenet/input_0.pb");

        // convert the TensorData to an ArrayD
        let Tensor::Constant(TensorData::Float(input)) = input else {panic!("Invalid input type")};

        let input_parameters = vec![];
        let result = service.run(input, input_parameters);

        //print the result
        println!("{:?}", result);

        //and write it to a file
        let TensorData::Float(result) = result.unwrap() else {panic!("Invalid result type")};
        let result = result.into_dimensionality::<ndarray::Ix2>().unwrap();
        write_to_csv(&result, "tests/results.csv");

        let mut expected_input_file = File::open("tests/testset/mobilenet/output_0.pb").unwrap();
        let mut expected_input_buffer = Vec::new();
        expected_input_file
            .read_to_end(&mut expected_input_buffer)
            .unwrap();
        let expected_output =
            crate::onnx_format::TensorProto::decode(expected_input_buffer.as_slice()).unwrap();

        let Tensor::Constant(TensorData::Float(expected_output)) = Tensor::from(expected_output) else {panic!("Invalid expected output type")};
        let expected_output = expected_output
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();
        write_to_csv(&expected_output, "tests/results_expected.csv");

        //check if the result is the same as the expected output
        let err = (result - expected_output).mapv(|x| x.abs()).mean().unwrap();
        assert!(err < 1e-4);
    }

    #[test]
    fn run_on_cat_image() {
        use crate::prepare::preprocessing;

        let image = image::open("tests/images/siamese-cat.jpg").unwrap();
        let preprocessed_image = preprocessing(image);

        //save the preprocessed tensor as an image file
        //preprocessed_image_to_file(&preprocessed_image, "tests/preprocessed_cat.jpg");

        let model_proto = read_model_proto("tests/models/resnet18-v2-7.onnx");
        let config = Config { num_threads: 1 };
        let service = Service::new(model_proto, config);
        let input_parameters = vec![];

        let result = service
            .run(preprocessed_image.into_dyn(), input_parameters)
            .unwrap();
        let TensorData::Float(result) = result else {panic!("Invalid result type")};
        let result = result.into_dimensionality::<ndarray::Ix2>().unwrap();
        let result = postprocessing(result);

        write_to_csv(
            &result.clone().insert_axis(ndarray::Axis(0)),
            "tests/results_cat.csv",
        );

        let top_5_results = postprocessing_top_k(result, 5);
        println!("Top 5 predictions:");
        for (i, (class, prob)) in top_5_results.iter().enumerate() {
            println!("{}. class: {}, probability: {}", i + 1, class, prob);
        }
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

    fn write_to_csv(tensor: &ndarray::Array2<f32>, path: &str) {
        let output_file = File::create(path).unwrap();
        let mut writer = WriterBuilder::new()
            .has_headers(false)
            .from_writer(output_file);
        writer.serialize_array2(tensor).unwrap();
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
