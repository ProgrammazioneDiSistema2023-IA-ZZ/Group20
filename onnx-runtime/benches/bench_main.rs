use benchmarks::{
    bench_bn::batchnorms, bench_convolution::convolutions, bench_gemm::gemms,
    bench_maxpool::maxpools, bench_runtime::runtime,
};
use criterion::criterion_main;
mod benchmarks;

criterion_main!(convolutions, runtime, batchnorms, gemms, maxpools);
