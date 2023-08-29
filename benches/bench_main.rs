use benchmarks::{bench_convolution::convolutions, bench_runtime::runtime};
use criterion::criterion_main;
mod benchmarks;

criterion_main!(convolutions, runtime);
