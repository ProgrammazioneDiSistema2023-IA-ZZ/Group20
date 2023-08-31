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

fn gemm_big() {
    let a_shape = [128, 256];
    let b_shape = [256, 512];
    let c_shape = [512];
    let y_shape = [128, 512];
    let a = load("tests/tensors/gemm/big/a.npy", &a_shape);
    let b = load("tests/tensors/gemm/big/b.npy", &b_shape);
    let c = load("tests/tensors/gemm/big/c.npy", &c_shape);
    let y = load("tests/tensors/gemm/big/y.npy", &y_shape);

    let attrs = GemmAttributes::new(1.8, 0.2, 0, 0);
    let my_y = ParNaiveProvider::gemm(&THREAD_POOL_4, a, b, c, attrs).unwrap();
    let err = y.sub(my_y).mapv(|x| x.abs()).mean().unwrap();
    // println!("avg error = {}", err);
    assert!(err < 1e-5);
}

fn bench_gemm(c: &mut Criterion) {
    let mut group = c.benchmark_group("Gemms");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(100));
    group.bench_function("GemmBig", move |b| b.iter(gemm_big));
    group.finish();
}

criterion_group!(gemms, bench_gemm,);
