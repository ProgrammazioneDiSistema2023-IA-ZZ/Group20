# ONNX Runtime - Group20

This repository contains the group project for an ONNX Runtime implementation using Rust. The project is part of the Italian “System and device programming” course ([02GRSOV](https://didattica.polito.it/pls/portal30/gap.pkg_guide.viewGap?p_cod_ins=02GRSOV&p_a_acc=2023&p_header=S&p_lang=IT&multi=N)) at Polytechnic University of Turin.

The requirements for the project are collected in [issue #1](https://github.com/ProgrammazioneDiSistema2023-IA-ZZ/Group20/issues/1).

Nearly all the development process is documented in the issues and in the pull requests. You can start from the requirements and then follow the development process to understand how the project was developed.

## Project structure

It is organized using a [cargo workspace](https://doc.rust-lang.org/book/ch14-03-cargo-workspaces.html) with the following structure:

- [onnx-runtime](onnx-runtime/README.md): the main library that implements our ONNX Runtime

- [onnx-runtime-cli](onnx-runtime-cli/README.md): a command line interface app that uses the library

- [onnx-binding](onnx-binding/README.md): Python bindings for the library
- [onnx-runtime-pyapp](onnx-runtime-pyapp/README.md): a Python app that uses our Python bindings

## Code documentation (cargo doc)

If you want to generate the documentation for the project, you can run the following command:

```bash

cargo doc --no-deps

```

Modern browsers won't allow you to open the documentation if it is not served through a web server. If you have python installed, you can use the following command to quickly serve the documentation:

```bash

python -m http.server --directory target/doc/

```

Then you can open our library documentation at the following address: http://localhost:8000/onnx_runtime/

## Testing (cargo test)

If you want to run all the tests for the project, you can run the following command:

```bash

cargo test --release

```

**NOTE**: the tests are run in release mode to speed up the execution. Otherwise, the tests would take too much time to complete.

## Benchmarking

We use [criterion](https://docs.rs/criterion/latest/criterion/) to run benchmarks for our project.

If you want to run all the benchmarks, you can run the following command:

```bash

cargo bench

```

They **can take a while to complete**.

In alternative you can run a single benchmark with the following command:

```bash

cargo bench --bench <bench_name>

```

For example, this will run the benchmark for the mobilenet model with 4 threads and a cat image as input:

```bash

cargo bench --bench "Cat mobilenet 4"

```

The interesting benchmarks are the ones that have the following names:

- `Cat mobilenet`: it runs the mobilenet model with a cat image as input in single thread mode

- `Cat mobilenet 2`: it runs the mobilenet model with a cat image as input in multi thread mode (2 threads)

- `Cat mobilenet 4`: it runs the mobilenet model with a cat image as input in multi thread mode (4 threads)

- `Cat resnet`: it runs the resnet model with a cat image as input in single thread mode

- `Cat resnet 2`: it runs the resnet model with a cat image as input in multi thread mode (2 threads)

- `Cat resnet 4`: it runs the resnet model with a cat image as input in multi thread mode (4 threads)

### Benchmark reports

The nice thing about criterion is that it generates a report with the results of the benchmarks. You can find the report in the `target/criterion/report/index.html` file.

Again, modern browsers won't allow you to open the HTML report if it is not served through a web server. If you have python installed, and want to use http.server module this time as well, you may have some problems because of the relative paths used in the report. To avoid that, you can host the criterion directory as a root folder and navigate to the report folder:

```bash

python -m http.server --directory target/criterion/

```

Then you can open the report at the following address: http://localhost:8000/report/index.html
