use criterion::{black_box, criterion_group, Criterion};
use onnx_runtime::onnx_format::ModelProto;
use onnx_runtime::service::prepare::postprocessing;
use onnx_runtime::service::prepare::preprocessing;
use onnx_runtime::service::Config;
use onnx_runtime::service::Service;
use onnx_runtime::tensor::TensorData;
use prost::Message;
use std::{fs::File, io::Read, time::Duration};

fn bench_with_cat_image(c: &mut Criterion) {
    let mut group = c.benchmark_group("Runtime");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(100));
    group.bench_function("Cat resnet", move |b| {
        b.iter(|| run_with_cat_image(black_box("resnet18-v2-7")))
    });
    group.bench_function("Cat mobilenet", move |b| {
        b.iter(|| run_with_cat_image(black_box("mobilenetv2-7")))
    });
    group.finish();
}

fn run_with_cat_image(model_name: &str) -> ndarray::Array2<f32> {
    let image = image::open("tests/images/siamese-cat.jpg").unwrap();
    let preprocessed_image = preprocessing(&image);

    let model_proto = read_model_proto(format!("tests/models/{}.onnx", model_name).as_str());
    let config = Config { num_threads: 1 };
    let service = Service::new(model_proto, config);
    let input_parameters = vec![];
    let result = service
        .run(preprocessed_image.into_dyn(), input_parameters)
        .unwrap();
    let TensorData::Float(result) = result else {
        panic!("Invalid result type")
    };
    let result = result.into_dimensionality::<ndarray::Ix2>().unwrap();
    postprocessing(result)
}

fn read_model_proto(path: &str) -> ModelProto {
    let mut buffer = Vec::new();
    let mut file = File::open(path).unwrap();
    file.read_to_end(&mut buffer).unwrap();

    ModelProto::decode(buffer.as_slice()).unwrap()
}

criterion_group!(runtime, bench_with_cat_image,);
