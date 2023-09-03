# Runtime library

This is the main library that implements our ONNX Runtime.

We chose to support the **mobilenetv2-7** and **resnet18-v2-7** models. These models are available in the [tests/models](../tests/models) folder.

As [required for the project](https://github.com/ProgrammazioneDiSistema2023-IA-ZZ/Group20/issues/1), we have implemented the following features:

- [Encoding and decoding of ONNX models](#encoding-and-decoding-of-onnx-models)
- [Execution of ONNX models](#execution-of-onnx-models)
- [Parallel execution of ONNX models](#parallel-execution)

Python **bindings** for the library are provided by [onnx-runtime-bindings](../onnx-runtime-bindings/README.md).

An example of how to use the library can be seen in the [onnx-runtime-cli](../onnx-runtime-cli/README.md) app.
However, almost each library feature has tests that can be used as examples.

Check out the [README.md](../README.md) of the workspace project for more general information about the project. You can also find information about [docs generation](../README.md#code-documentation-cargo-doc), [benchmarking](../README.md#benchmarking) and [testing](../README.md#testing-cargo-test).

## Encoding and decoding of ONNX models

PR [#2](https://github.com/ProgrammazioneDiSistema2023-IA-ZZ/Group20/pull/2) implements this feature and explains how it works.

Briefly, we take advantage of the Protobuf code generation and the [prost](https://docs.rs/prost/latest/prost/) crate to encode and decode ONNX models. Please check out [#2](https://github.com/ProgrammazioneDiSistema2023-IA-ZZ/Group20/pull/2) for more details, like how to generate the Rust code and why we picked this approach.

### Example

The library provides the `onnx_format` module that can be used to encode and decode ONNX models.
The module offers the ModelProto structure that can be used to encode and decode ONNX models. It relies on the [prost](https://docs.rs/prost/latest/prost/) crate.

An example could be:

```rust
use onnx_format::ModelProto;
use prost::Message;
use std::{
   fs::File,
   io::{Read, Write},
};

let mut in_buffer = Vec::new();
let mut out_buffer = Vec::new();

let mut in_file = File::open("tests/models/mobilenetv2-7.onnx").unwrap();
let mut out_file = File::create("tests/models/cloned-mobilenet.onnx").unwrap();

in_file.read_to_end(&mut buffer).unwrap();

let parsed_model = ModelProto::decode(buffer.as_slice()).expect("Failed to decode model");
parsed_model.encode(&mut out_buffer).expect("Failed to encode model");

out_file.write_all(&out_buffer).unwrap();
```

## Execution of ONNX models
This is such a big feature that it has its own tracking issue: [#3](https://github.com/ProgrammazioneDiSistema2023-IA-ZZ/Group20/issues/3).

### Data structure for model execution
The ModelProto structure is inconvenient to use for model execution. Thus, we had to create a data structure that is more convenient for model execution. We opted for a graph (using [petgraph](https://docs.rs/petgraph/latest/petgraph/)) and a [toposort](https://docs.rs/petgraph/latest/petgraph/algo/fn.toposort.html) algorithm to arrange the correct execution order. This was done in the `translator` module.

### Operators execution implementation
The next step was to implement the execution of the operators. ONNX supports a lot of operators, but the project requirements states to chose them in order to support at least two models. We decided to support the **mobilenetv2** and **resnet18v2** models. Furthermore, ONNX specs have a lot of versions. We chose to support **version 7**. The implementations rely on the [ndarray](https://docs.rs/ndarray/latest/ndarray/) crate and use very simple algorithms. For example, the convolution operator is implemented using a naive algorithm. Hence, poor performance are expected (at least ~10 times slower than smarter implementations). Simpler algorithms were used to keep the implementation simple and to focus on the overall architecture of the project.
The implementation of the operators can be found in the `provider` module. Every operator has been thoroughly tested in [tests/operators.rs](tests/operators.rs) by comparing the results with the ones obtained with python PyTorch.

**NOTE**: floating point operations are not bit-exact. Thus, we had to use a tolerance up to 1e-4 to compare the results. This is not a problem for the project, but it is something to keep in mind when we'll compare the inferred results with other frameworks.

## Parallel execution
PR [#11](https://github.com/ProgrammazioneDiSistema2023-IA-ZZ/Group20/pull/11) implements this feature and explains how it was done.

In a nutshell, we used the [rayon](https://docs.rs/rayon/latest/rayon/) crate to parallelize the execution of the operators in our only Execution Provider (`NaiveProvider`) and put them in the `ParNaiveProvider`.

Rather than executing multiple operations in parallel, we chose to execute each operation in parallel, relegating to their implementation the choice on how to generate and aggregate sub-tasks. They may even decide not to use this capability. This is because some operations are very fast and parallelizing them would be a waste of resources.

We decided to parallelize the execution inside an operator and follow a sequential execution order for the model because we think that it is easier to implement and, in our case, it was the most convenient solution. This is because our models do not expect to have a lot of branches in the execution graph, so that kind of parallelization would not be very useful and would require a lot of work to synchronize the threads correctly.

Of course, the project can be extended to support parallel execution of the operators in the graph. This would require implementing a more complex synchronization mechanism between the threads and implementing a more complex graph traversal algorithm. However, we think that this is not a priority for the project.

