use ndarray::{ArrayD, IxDyn};
use ndarray_csv::Array2Writer;
use petgraph::algo::toposort;
use std::{borrow::BorrowMut, fs::File, io::Write, ops::ControlFlow};

use crate::{
    graph::create_graph,
    onnx_format::ModelProto,
    operators::{OperationError, Operator},
    providers::{DefaultProvider, Provider},
    tensor::{DynamicTensorData, Tensor, TensorData, TensorDataIntoDimensionality},
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
    ///
    pub fn run(
        &self,
        input: ArrayD<f32>,
        input_parameters: Vec<(String, f32)>,
    ) -> Result<TensorData, &'static str> {
        self.run_with_provider::<DefaultProvider>(input, input_parameters)
    }

    pub fn run_with_provider<P>(
        &self,
        input: ArrayD<f32>,
        input_parameters: Vec<(String, f32)>,
    ) -> Result<TensorData, &'static str>
    where
        P: Provider,
    {
        println!("Running service with {} threads", self.config.num_threads);
        let mut debug_file = File::create("tests/log.txt").unwrap();

        let mut operations_graph = create_graph(self.model.clone()).unwrap();

        let n_operations = operations_graph.node_count();
        let mut executed_operations = 0;

        let mut final_output = None;
        toposort(&operations_graph, None)
            .unwrap()
            .into_iter()
            .try_for_each(|node| {
                let incoming_data = operations_graph
                    .borrow_mut()
                    .edges_directed(node, petgraph::Direction::Incoming)
                    .map(|e| {
                        //TODO: wait for the input to be ready for execution in a multithreaded environment
                        e.weight()
                            .borrow()
                            .clone()
                            .expect("Executing an operation without input")
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

                let Ok(outgoing_data) = operation_result else { return ControlFlow::Break(operation_result.map(|_| ()))};
                executed_operations += 1;
                println!(
                    "Executed {} out of {} operations",
                    executed_operations, n_operations
                );
                //append to a file the output of the current operation
                writeln!(
                    debug_file,
                    "Operation number: {:?} out of {:?}",
                    executed_operations, n_operations
                )
                .unwrap();
                writeln!(debug_file, "Operation: {:?}", operations_graph[node]).unwrap();
                writeln!(debug_file, "Intermediate output:").unwrap();
                writeln!(debug_file, "{:?}", outgoing_data).unwrap();

                // for each outgoing edge, set the data to the outgoing data
                operations_graph
                    .borrow_mut()
                    .edges_directed(node, petgraph::Direction::Outgoing)
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
            let TensorData::Float(operand) = inputs[0].clone() else {todo!("conv2d: invalid input tensor type")};
            let TensorData::Float(weights) = inits.weights.clone() else {todo!("conv2d: invalid weights tensor type")};

            //operand to Array4
            let bias = inits.bias.clone().map(|b| match b {
                TensorData::Float(b) => b.into_dimensionality::<ndarray::Ix1>().unwrap(),
                _ => todo!("conv2d: invalid bias tensor type"),
            });

            let result = ChosenProvider::conv(operand, weights, bias, attrs.clone())?;
            let tensor = TensorData::Float(result);
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

#[test]
fn test_service() {
    use crate::onnx_format::ModelProto;
    use csv::WriterBuilder;
    use prost::Message;
    use std::fs::File;
    use std::io::Read;

    let mut buffer = Vec::new();
    let mut file = File::open("tests/models/mobilenetv2-7.onnx").unwrap();
    file.read_to_end(&mut buffer).unwrap();

    let parsed_model = ModelProto::decode(buffer.as_slice()).unwrap();

    let output_file = File::create("tests/results.csv").unwrap();
    let mut writer = WriterBuilder::new()
        .has_headers(false)
        .from_writer(output_file);

    let config = Config { num_threads: 1 };
    let service = Service::new(parsed_model, config);

    //read input from test.pb as an ArrayD of shape [1, 3, 224, 224]
    let mut input_buffer = Vec::new();
    let mut input_file = File::open("tests/testset/mobilenet/input_0.pb").unwrap();

    input_file.read_to_end(&mut input_buffer).unwrap();
    // decode input as a TensorProto
    let input = crate::onnx_format::TensorProto::decode(input_buffer.as_slice()).unwrap();
    // convert the input to a TensorData
    let input = Tensor::from(input);
    // convert the TensorData to an ArrayD
    let Tensor::Constant(TensorData::Float(input)) = input else {panic!("Invalid input type")};

    let input_parameters = vec![];
    let result = service.run(input, input_parameters);

    //print the result
    println!("{:?}", result);

    //and write it to a file
    let TensorData::Float(result) = result.unwrap() else {panic!("Invalid result type")};
    let result = result.into_dimensionality::<ndarray::Ix2>().unwrap();
    writer.serialize_array2(&result).unwrap();

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

    let expected_output_file = File::create("tests/results_expected.csv").unwrap();
    let mut writer = WriterBuilder::new()
        .has_headers(false)
        .from_writer(expected_output_file);
    writer.serialize_array2(&expected_output).unwrap();

    //check if the result is the same as the expected output
    let err = (result - expected_output).mapv(|x| x.abs()).mean().unwrap();
    assert!(err < 1e-4);
}
