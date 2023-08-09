use std::{fs::File, io::Read, ops::Sub, time::Duration};
use ndarray::{ArrayD, IxDyn, Ix1};
use npy::NpyData;
use onnx_runtime::operators::*;
use criterion::{criterion_group, Criterion};

fn load(path: &str, shape: &[usize]) -> ArrayD<f32> {
    let mut buf = vec![];
    File::open(path).unwrap().read_to_end(&mut buf).unwrap();
    let array_data: NpyData<'_, f32> = NpyData::from_bytes(&buf).expect("Failed from_bytes");
    ArrayD::<f32>::from_shape_vec(IxDyn(shape), array_data.to_vec()).unwrap()
}

fn convolution_big() {
    let x_shape = [1, 64, 224, 224];
    let w_shape = [64, 64, 3, 3];
    let b_shape = [64];
    let y_shape = [1, 64, 224, 224];
    let x = load("tests/tensors/convolution/big/x.npy", &x_shape);
    let w = load("tests/tensors/convolution/big/w.npy", &w_shape);
    let b = load("tests/tensors/convolution/big/b.npy", &b_shape).into_dimensionality::<Ix1>().unwrap();
    let y = load("tests/tensors/convolution/big/y.npy", &y_shape);
    let attrs = ConvAttributes::new([1, 1], 1, [3, 3], [1, 1, 1, 1], [1, 1]);
    let my_y = conv(x, w, Some(b), attrs).unwrap();
    let err = y.sub(my_y).mapv(|x| x.abs()).mean().unwrap();
    println!("avg error = {}", err);
    assert!(err < 1e-4);
}

#[allow(dead_code)]
fn convolution_huge() {
    let x_shape = [1, 128, 224, 224];
    let w_shape = [256, 128, 3, 3];
    let b_shape = [256];
    let y_shape = [1, 256, 224, 224];
    let x = load("tests/tensors/convolution/huge/x.npy", &x_shape);
    let w = load("tests/tensors/convolution/huge/w.npy", &w_shape);
    let b = load("tests/tensors/convolution/huge/b.npy", &b_shape).into_dimensionality::<Ix1>().unwrap();
    let y = load("tests/tensors/convolution/huge/y.npy", &y_shape);
    let attrs = ConvAttributes::new([1, 1], 1, [3, 3], [1, 1, 1, 1], [1, 1]);
    let my_y = conv(x, w, Some(b), attrs).unwrap();
    let err = y.sub(my_y).mapv(|x| x.abs()).mean().unwrap();
    println!("avg error = {}", err);
    assert!(err < 1e-4);
}

fn bench_convolution(c: &mut Criterion) {
    let mut group = c.benchmark_group("Convolution");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(100));
    group.bench_function("ConvBig", move|b| b.iter(|| convolution_big()));
    // group.bench_function("ConvHuge", move|b| b.iter_batched(||{}, |_| convolution_huge(), BatchSize::NumBatches(1)));
    group.finish();
}

/*
fn bench_convolution_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("Fibonacci3");
    for i in 20..=21 {
        group.bench_with_input(BenchmarkId::new("Recursive", i), &i, |b, i| {
            b.iter(|| fibonacci_slow(*i))
        });
        group.bench_with_input(BenchmarkId::new("Iterative", i), &i, |b, i| {
            b.iter(|| fibonacci_fast(*i))
        });
    }
    group.finish()
}*/

criterion_group!(convolutions, bench_convolution,);