use benchmarks::{bench_convolution::convolutions, bench_runtime::runtime, bench_bn::batchnorms, bench_gemm::gemms, bench_maxpool::maxpools};
use criterion::criterion_main;
mod benchmarks;

criterion_main!(convolutions, runtime, batchnorms, gemms, maxpools);
