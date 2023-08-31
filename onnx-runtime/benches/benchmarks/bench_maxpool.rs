use criterion::{criterion_group, Criterion};
use lazy_static::lazy_static;
use ndarray::{ArrayD, IxDyn};
use npy::NpyData;
use onnx_runtime::{
    operators::*,
    providers::{Provider, ParNaiveProvider},
};
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::{fs::File, io::Read, ops::Sub, time::Duration};

lazy_static! {
    static ref THREAD_POOL_4: ThreadPool = ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("Unable to create ThreadPool");
}

fn load(path: &str, shape: &[usize]) -> ArrayD<f32> {
    let mut buf = vec![];
    File::open(path).unwrap().read_to_end(&mut buf).unwrap();
    let array_data: NpyData<'_, f32> = NpyData::from_bytes(&buf).expect("Failed from_bytes");
    ArrayD::<f32>::from_shape_vec(IxDyn(shape), array_data.to_vec()).unwrap()
}

fn maxpool_big() {
    let shape_x = [1, 256, 224, 224];
    let shape_y = [1, 256, 112, 112];
    let x: ArrayD<f32> = load("tests/tensors/maxpool/big/x.npy", &shape_x);
    let y: ArrayD<f32> = load("tests/tensors/maxpool/big/y.npy", &shape_y);
    let attrs = MaxPoolAttributes::new([3, 3], [1, 1, 1, 1], [2, 2]);
    let my_y = ParNaiveProvider::max_pool(&THREAD_POOL_4, x, attrs).unwrap();
    let err = y.sub(my_y).mapv(|x| x.abs()).mean().unwrap();
    // println!("avg error = {}", err);
    assert!(err < 1e-5);
}

fn bench_maxpool(c: &mut Criterion) {
    let mut group = c.benchmark_group("MaxPools");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(100));
    group.bench_function("MaxPoolBig", move |b| b.iter(maxpool_big));
    group.finish();
}

criterion_group!(maxpools, bench_maxpool,);
