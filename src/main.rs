use ndarray::Array4;
use onnx_runtime::functions::{convolution, ConvAttributes};

fn main() {
    let batch_size = 2; // Example batch size
    let input_channels = 512; // Example number of input channels
    let height = 64; // Height of the 2D input data
    let width = 64; // Width of the 2D input data

    let x: Array4<f32> = Array4::from_shape_fn(
        (batch_size, input_channels, height, width),
        |(_, _, h, w)| {
            // Create some example values for the input tensor (replace with your own data)
            (h * width + w) as f32
        },
    );

    let num_features = 512; // Example number of features (number of filters)
    let kernel_height = 3; // Height of the convolutional kernel (filter)
    let kernel_width = 3; // Width of the convolutional kernel (filter)

    let w: Array4<f32> = Array4::from_shape_fn(
        (num_features, input_channels, kernel_height, kernel_width),
        |(_, _, h, w)| {
            // Create some example values for the weight tensor (replace with your own data)
            ((h * kernel_width + w) as f32) * 0.1
        },
    );

    let attrs = ConvAttributes::new([1, 1], 1, [3, 3], [1, 1, 1, 1], [1, 1]);
    println!("{:?}", convolution(x, w, attrs));
}
