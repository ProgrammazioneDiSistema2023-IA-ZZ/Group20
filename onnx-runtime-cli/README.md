# Rust CLI app for the ONNX Runtime library

This is a simple CLI app that uses the [onnx-runtime](../onnx-runtime) library.

## Usage
You can run the app directly with cargo:

```bash
cargo run --release -- -i <input_file> -m <model_name> -t <num_threads> -s <num_results>
```


A short description of the arguments can be found with:
```bash
cargo run --release -- -h
```

More detailed info about the arguments can be found with:
```bash
cargo run --release -- --help
```

### Logging
Basics logging can be enabled with the `RUST_LOG` environment variable:

```bash
RUST_LOG=info cargo run --release -- -i <input_file> -m <model_name> -t <num_threads> -s <num_results>
```

### Build
You can build the app and run it directly with:

```bash
cargo build --release

./target/release/onnx-runtime-cli -i <input_file> -m <model_name> -t <num_threads> -s <num_results>
```

### Example

A simple case scenario would be to use the app to run the mobilenet model on a given image with 4 threads and you want the app to show the top 3 results:

```bash
onnx-runtime-cli -i my-image.jpeg -m mobilenet -t 4 -s 3
```

This will print something like this:
```bash
Top 3 predictions:
  Image #1
    1. class: Siamese cat, Siamese, probability: 99.09454 %
    2. class: Egyptian cat, probability: 0.772422 %
    3. class: lynx, catamount, probability: 0.016159737 %
```

Resnet support an image batch as input. Thus, you can specify a list of images to run the model on:

```bash
onnx-runtime-cli -i img1.jpeg -i img2.png -i img3.jpg -m resnet -t 4 -s 3
```

This will print something like this:
```bash
Top 3 predictions:
  Image #1
    1. class: Siamese cat, Siamese, probability: 99.79126 %
    2. class: Egyptian cat, probability: 0.10404349 %
    3. class: cougar, puma, catamount, mountain lion, painter, panther, Felis concolor, probability: 0.031973638 %
  Image #2
    1. class: candle, taper, wax light, probability: 99.99932 %
    2. class: lighter, light, igniter, ignitor, probability: 0.0004174863 %
    3. class: matchstick, probability: 0.00022958497 %
  Image #3
    1. class: flagpole, flagstaff, probability: 98.14253 %
    2. class: parachute, chute, probability: 0.9296734 %
    3. class: bow tie, bow-tie, bowtie, probability: 0.21822464 %
```
