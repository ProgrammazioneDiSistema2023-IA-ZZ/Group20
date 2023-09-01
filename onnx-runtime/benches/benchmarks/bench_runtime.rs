use criterion::{black_box, criterion_group, Criterion};
use onnx_runtime::providers::ParNaiveProvider;
use onnx_runtime::service::prepare::{postprocessing, preprocessing};
use onnx_runtime::service::utility::read_model_proto;
use onnx_runtime::service::{Config, Service};
use onnx_runtime::tensor::TensorData;
use std::time::Duration;

fn bench_with_cat_image(c: &mut Criterion) {
    let mut group = c.benchmark_group("Runtime");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(100));
    group.bench_function("Cat resnet", move |b| {
        b.iter(|| run_with_cat_image(black_box("resnet18-v2-7"), 1))
    });
    group.bench_function("Cat mobilenet", move |b| {
        b.iter(|| run_with_cat_image(black_box("mobilenetv2-7"), 1))
    });
    group.finish();
}

fn bench_with_cat_image_4(c: &mut Criterion) {
    let mut group = c.benchmark_group("Runtime");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(100));
    group.bench_function("Cat resnet 4", move |b| {
        b.iter(|| run_with_cat_image(black_box("resnet18-v2-7"), 4))
    });
    group.bench_function("Cat mobilenet 4", move |b| {
        b.iter(|| run_with_cat_image(black_box("mobilenetv2-7"), 4))
    });
    group.finish();
}

fn bench_with_cat_image_2(c: &mut Criterion) {
    let mut group = c.benchmark_group("Runtime");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(100));
    group.bench_function("Cat resnet 2", move |b| {
        b.iter(|| run_with_cat_image(black_box("resnet18-v2-7"), 2))
    });
    group.bench_function("Cat mobilenet 2", move |b| {
        b.iter(|| run_with_cat_image(black_box("mobilenetv2-7"), 2))
    });
    group.finish();
}

fn run_with_cat_image(model_name: &str, num_threads: usize) -> ndarray::Array2<f32> {
    let image = image::open("tests/images/siamese-cat.jpg").unwrap();
    let preprocessed_image = preprocessing(&image);

    let model_proto = read_model_proto(format!("tests/models/{}.onnx", model_name).as_str());
    let config = Config { num_threads };
    let service = Service::new(model_proto, config);
    let input_parameters = vec![("N".to_string(), 1_usize)];
    let result = service
        .run_with_provider::<ParNaiveProvider>(preprocessed_image.into_dyn(), input_parameters)
        .unwrap();
    let TensorData::Float(result) = result else {
        panic!("Invalid result type")
    };
    let result = result.into_dimensionality::<ndarray::Ix2>().unwrap();
    postprocessing(result)
}

criterion_group!(
    runtime,
    bench_with_cat_image,
    bench_with_cat_image_2,
    bench_with_cat_image_4
);
