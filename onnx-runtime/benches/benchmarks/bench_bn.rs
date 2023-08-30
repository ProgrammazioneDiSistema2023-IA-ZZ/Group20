use criterion::{criterion_group, Criterion};
use lazy_static::lazy_static;
use ndarray::{ArrayD, IxDyn};
use npy::NpyData;
use onnx_runtime::{
    operators::*,
    providers::{Provider, ParNaiveProvider, NaiveProvider},
};
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::{fs::File, io::Read, ops::Sub, time::Duration};

lazy_static! {
    static ref THREAD_POOL_4: ThreadPool = ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .expect("Unable to create ThreadPool");
}

fn load(path: &str, shape: &[usize]) -> ArrayD<f32> {
    let mut buf = vec![];
    File::open(path).unwrap().read_to_end(&mut buf).unwrap();
    let array_data: NpyData<'_, f32> = NpyData::from_bytes(&buf).expect("Failed from_bytes");
    ArrayD::<f32>::from_shape_vec(IxDyn(shape), array_data.to_vec()).unwrap()
}

fn batchnorm_big() {
    let shape_x = [1, 256, 224, 224];
    let shape_mean = [256];
    let shape_b = [256];
    let shape_scale = [256];
    let shape_var = [256];
    let shape_y = [1, 256, 224, 224];

    let x: ArrayD<f32> = load("tests/tensors/bn/big/x.npy", &shape_x);
    let mean: ArrayD<f32> = load("tests/tensors/bn/big/mean.npy", &shape_mean);
    let b: ArrayD<f32> = load("tests/tensors/bn/big/b.npy", &shape_b);
    let scale: ArrayD<f32> = load("tests/tensors/bn/big/scale.npy", &shape_scale);
    let var: ArrayD<f32> = load("tests/tensors/bn/big/var.npy", &shape_var);
    let y: ArrayD<f32> = load("tests/tensors/bn/big/y.npy", &shape_y);
    let attrs = BatchNormAttributes::new(1e-5, 0.9, 1);

    let my_y = NaiveProvider::batch_norm(&THREAD_POOL_4, x, scale, b, mean, var, attrs).unwrap();

    let err = y.sub(my_y).mapv(|x| x.abs()).mean().unwrap();
    // println!("avg error = {}", err);
    assert!(err < 1e-5);
}

fn bench_bn(c: &mut Criterion) {
    let mut group = c.benchmark_group("BatchNorms");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(100));
    group.bench_function("BnBig", move |b| b.iter(batchnorm_big));
    group.finish();
}

criterion_group!(batchnorms, bench_bn,);
