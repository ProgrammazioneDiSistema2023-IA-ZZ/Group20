use std::error::Error;

use clap::Parser;
use log::info;
use onnx_runtime::service::{Config, InferenceOutput, Prediction, ServiceBuilder, ServiceError};

mod cli;
use cli::Args;
use thiserror::Error;

#[derive(Debug, Error)]
enum AppError {
    #[error("Invalid model path")]
    InvalidModelPath,
    #[error("Could not decode image -> {0}")]
    CouldNotDecodeImage(Box<dyn Error>),
    #[error("Runtime failure -> {0}")]
    RuntimeFailure(ServiceError),
}

fn main() {
    if let Err(e) = exec_program() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

fn exec_program() -> Result<(), AppError> {
    env_logger::init();

    let args = Args::parse();

    let input_path = args.input;
    let model_name = args.model;
    let num_threads = args.threads.into();
    let ranking_len = args.show.into();
    let input_parameters = args
        .params
        .into_iter()
        .map(|p| (p.name, p.value))
        .collect::<Vec<_>>();
    let model_proto_path = format!("models/{}.onnx", model_name);

    // Default batch size is equal to the number of images and is called "N"
    let input_parameters = if input_parameters.is_empty() {
        vec![("N".to_string(), input_path.len())]
    } else {
        input_parameters
    };

    info!("Input image path: {:?}", input_path);
    info!("Model name: {}", model_name);
    info!(
        "Number of top infered class probabilities to show: {}",
        ranking_len
    );
    info!("Input parameters: {:?}", input_parameters);

    let config = Config { num_threads };
    let service = ServiceBuilder::new(model_proto_path.into())
        .config(config)
        .build()
        .map_err(|_| AppError::InvalidModelPath)?;

    info!("Service created successfully");

    let result = service
        .prepare_and_run(input_path, input_parameters)
        .map_err(|e| match e {
            ServiceError::InvalidInput(e) => AppError::CouldNotDecodeImage(e),
            e => AppError::RuntimeFailure(e),
        })?;

    print_top_k_batch_predictions(result, ranking_len);

    Ok(())
}

fn print_top_k_batch_predictions(output: InferenceOutput, k: usize) {
    println!("Top {} predictions:", k);
    for (i, image_prediction) in output.get_top_k_predictions(k).into_iter().enumerate() {
        println!("  Image #{}", i + 1);
        print_batch_element_prediction(image_prediction);
    }
}
fn print_batch_element_prediction(image_prediction: Vec<Prediction>) {
    for (rank, prediction) in image_prediction.iter().enumerate() {
        println!(
            "    {}. class: {}, probability: {} %",
            rank + 1,
            prediction.class,
            prediction.probability * 100_f32
        );
    }
}
