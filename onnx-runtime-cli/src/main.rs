use clap::Parser;
use log::info;
use onnx_runtime::{
    prepare::{postprocessing, postprocessing_top_k},
    service::{Config, Service, ServiceError},
    tensor::TensorData,
    utils::{read_and_prepare_image, read_model_proto, ImageError},
};

mod cli;
use cli::Args;

#[derive(Debug)]
enum AppError {
    CouldNotDecodeImage(ImageError),
    RuntimeFailure(ServiceError),
    WrongOutputType,
}

fn main() -> Result<(), AppError> {
    env_logger::init();

    let args = Args::parse();

    let input_path = args.input;
    let model_name = args.model;
    let num_threads = args.threads.into();
    let ranking_len = args.show.into();
    let input_parameters = args.params.into_iter().map(|p| (p.name, p.value)).collect();

    info!("Input image path: {:?}", input_path);
    info!("Model name: {}", model_name);
    info!(
        "Number of top infered class probabilities to show: {}",
        ranking_len
    );
    info!("Input parameters: {:?}", input_parameters);

    let model_proto = read_model_proto(&format!("models/{}.onnx", model_name));
    let preprocessed_image =
        read_and_prepare_image(input_path).map_err(AppError::CouldNotDecodeImage)?;

    info!("Model loaded successfully");
    info!("Image preprocessed successfully");

    let config = Config { num_threads };
    let service = Service::new(model_proto, config);

    let result = service
        .run(preprocessed_image.into_dyn(), input_parameters)
        .map_err(AppError::RuntimeFailure)?;
    let TensorData::Float(result) = result else {
        return Err(AppError::WrongOutputType);
    };
    let result = postprocessing(result);

    println!("Top {} predictions:", ranking_len);
    for (i, (class, prob)) in postprocessing_top_k(result, ranking_len)
        .into_iter()
        .enumerate()
    {
        println!(
            "{}. class: {}, probability: {} %",
            i + 1,
            class,
            prob * 100_f32
        );
    }

    Ok(())
}
