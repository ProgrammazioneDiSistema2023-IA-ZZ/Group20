use ndarray::{Array1, Array4, arr0, Array2, arr2};
use npy::NpyData;
use onnx_runtime::functions::*;
use std::fs::File;
use std::io::Read;
use std::ops::Sub;

fn load4d(path: &str, shape: [usize; 4]) -> Array4<f32> {
    let mut buf = vec![];
    File::open(path).unwrap().read_to_end(&mut buf).unwrap();
    let array_data: NpyData<'_, f32> = NpyData::from_bytes(&buf).expect("Failed from_bytes");
    Array4::<f32>::from_shape_vec(shape, array_data.to_vec()).unwrap()
}
fn load1d(path: &str, shape: [usize; 1]) -> Array1<f32> {
    let mut buf = vec![];
    File::open(path).unwrap().read_to_end(&mut buf).unwrap();
    let array_data: NpyData<'_, f32> = NpyData::from_bytes(&buf).expect("Failed from_bytes");
    Array1::<f32>::from_shape_vec(shape, array_data.to_vec()).unwrap()
}
fn load2d(path: &str, shape: [usize; 2]) -> Array2<f32> {
    let mut buf = vec![];
    File::open(path).unwrap().read_to_end(&mut buf).unwrap();
    let array_data: NpyData<'_, f32> = NpyData::from_bytes(&buf).expect("Failed from_bytes");
    Array2::<f32>::from_shape_vec(shape, array_data.to_vec()).unwrap()
}

#[allow(dead_code, unused_variables)]
fn test_convolution() {
    let batch_size = 2; // Example batch size
    let input_channels = 4; // Example number of input channels
    let height = 5; // Height of the 2D input data
    let width = 5; // Width of the 2D input data

    let x: Array4<f32> = Array4::from_shape_fn(
        (batch_size, input_channels, height, width),
        |(_, _, h, w)| {
            // Create some example values for the input tensor (replace with your own data)
            (h * width + w) as f32
        },
    );

    let num_features = 4; // Example number of features (number of filters)
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
    // println!("{:?}", convolution(x, w, None, attrs));
    assert!(true);
}

#[test]
fn test_convolution_basic() {
    let x_shape = [1, 4, 5, 5];
    let w_shape = [4, 4, 3, 3];
    let y_shape = [1, 4, 3, 3];
    let x = load4d("tests/tensors/convolution/basic/x.npy", x_shape);
    let w = load4d("tests/tensors/convolution/basic/w.npy", w_shape);
    let y = load4d("tests/tensors/convolution/basic/y.npy", y_shape);
    let attrs = ConvAttributes::new([1, 1], 1, [3, 3], [0, 0, 0, 0], [1, 1]);
    let my_y = convolution(x, w, None, attrs);
    let err = y.sub(my_y).mapv(|x| x.abs()).mean().unwrap();
    // println!("avg error = {}", err);
    assert!(err < 1e-5);
}

#[test]
fn test_convolution_stride2() {
    let x_shape = [1, 4, 5, 5];
    let w_shape = [4, 4, 3, 3];
    let y_shape = [1, 4, 2, 2];
    let x = load4d("tests/tensors/convolution/stride2/x.npy", x_shape);
    let w = load4d("tests/tensors/convolution/stride2/w.npy", w_shape);
    let y = load4d("tests/tensors/convolution/stride2/y.npy", y_shape);
    let attrs = ConvAttributes::new([1, 1], 1, [3, 3], [0, 0, 0, 0], [2, 2]);
    let my_y = convolution(x, w, None, attrs);
    let err = y.sub(my_y).mapv(|x| x.abs()).mean().unwrap();
    // println!("avg error = {}", err);
    assert!(err < 1e-5);
}

#[test]
fn test_convolution_pad1() {
    let x_shape = [1, 4, 5, 5];
    let w_shape = [4, 4, 3, 3];
    let y_shape = [1, 4, 5, 5];
    let x = load4d("tests/tensors/convolution/pad1/x.npy", x_shape);
    let w = load4d("tests/tensors/convolution/pad1/w.npy", w_shape);
    let y = load4d("tests/tensors/convolution/pad1/y.npy", y_shape);
    let attrs = ConvAttributes::new([1, 1], 1, [3, 3], [1, 1, 1, 1], [1, 1]);
    let my_y = convolution(x, w, None, attrs);
    let err = y.sub(my_y).mapv(|x| x.abs()).mean().unwrap();
    // println!("avg error = {}", err);
    assert!(err < 1e-5);
}

#[test]
fn test_convolution_dil2() {
    let x_shape = [1, 4, 5, 5];
    let w_shape = [4, 4, 3, 3];
    let y_shape = [1, 4, 1, 1];
    let x = load4d("tests/tensors/convolution/dil2/x.npy", x_shape);
    let w = load4d("tests/tensors/convolution/dil2/w.npy", w_shape);
    let y = load4d("tests/tensors/convolution/dil2/y.npy", y_shape);
    let attrs = ConvAttributes::new([2, 2], 1, [3, 3], [0, 0, 0, 0], [1, 1]);
    let my_y = convolution(x, w, None, attrs);
    let err = y.sub(my_y).mapv(|x| x.abs()).mean().unwrap();
    // println!("avg error = {}", err);
    assert!(err < 1e-5);
}

#[test]
fn test_convolution_dil2big() {
    let x_shape = [1, 4, 9, 9];
    let w_shape = [4, 4, 3, 3];
    let y_shape = [1, 4, 5, 5];
    let x = load4d("tests/tensors/convolution/dil2big/x.npy", x_shape);
    let w = load4d("tests/tensors/convolution/dil2big/w.npy", w_shape);
    let y = load4d("tests/tensors/convolution/dil2big/y.npy", y_shape);
    let attrs = ConvAttributes::new([2, 2], 1, [3, 3], [0, 0, 0, 0], [1, 1]);
    let my_y = convolution(x, w, None, attrs);
    let err = y.sub(my_y).mapv(|x| x.abs()).mean().unwrap();
    // println!("avg error = {}", err);
    assert!(err < 1e-5);
}

#[test]
fn test_convolution_dil2big_stride2() {
    let x_shape = [1, 4, 9, 9];
    let w_shape = [4, 4, 3, 3];
    let y_shape = [1, 4, 3, 3];
    let x = load4d("tests/tensors/convolution/dil2big_stride2/x.npy", x_shape);
    let w = load4d("tests/tensors/convolution/dil2big_stride2/w.npy", w_shape);
    let y = load4d("tests/tensors/convolution/dil2big_stride2/y.npy", y_shape);
    let attrs = ConvAttributes::new([2, 2], 1, [3, 3], [0, 0, 0, 0], [2, 2]);
    let my_y = convolution(x, w, None, attrs);
    let err = y.sub(my_y).mapv(|x| x.abs()).mean().unwrap();
    // println!("avg error = {}", err);
    assert!(err < 1e-5);
}

#[test]
fn test_convolution_dil2big_stride2_pad2() {
    let x_shape = [1, 4, 9, 9];
    let w_shape = [4, 4, 3, 3];
    let y_shape = [1, 4, 5, 5];
    let x = load4d(
        "tests/tensors/convolution/dil2big_stride2_pad2/x.npy",
        x_shape,
    );
    let w = load4d(
        "tests/tensors/convolution/dil2big_stride2_pad2/w.npy",
        w_shape,
    );
    let y = load4d(
        "tests/tensors/convolution/dil2big_stride2_pad2/y.npy",
        y_shape,
    );
    let attrs = ConvAttributes::new([2, 2], 1, [3, 3], [2, 2, 2, 2], [2, 2]);
    let my_y = convolution(x, w, None, attrs);
    let err = y.sub(my_y).mapv(|x| x.abs()).mean().unwrap();
    // println!("avg error = {}", err);
    assert!(err < 1e-5);
}

#[test]
fn test_convolution_complete() {
    let x_shape = [16, 4, 9, 9];
    let w_shape = [8, 2, 3, 3];
    let y_shape = [16, 8, 5, 5];
    let x = load4d("tests/tensors/convolution/complete/x.npy", x_shape);
    let w = load4d("tests/tensors/convolution/complete/w.npy", w_shape);
    let y = load4d("tests/tensors/convolution/complete/y.npy", y_shape);
    let attrs = ConvAttributes::new([2, 2], 2, [3, 3], [2, 2, 2, 2], [2, 2]);
    let my_y = convolution(x, w, None, attrs);
    let err = y.sub(my_y).mapv(|x| x.abs()).mean().unwrap();
    // println!("avg error = {}", err);
    assert!(err < 1e-5);
}

#[test]
fn test_convolution_complete_bias() {
    let x_shape = [16, 4, 9, 9];
    let w_shape = [8, 2, 3, 3];
    let y_shape = [16, 8, 5, 5];
    let b_shape = [8];
    let x = load4d("tests/tensors/convolution/complete_bias/x.npy", x_shape);
    let w = load4d("tests/tensors/convolution/complete_bias/w.npy", w_shape);
    let b = load1d("tests/tensors/convolution/complete_bias/b.npy", b_shape);
    let y = load4d("tests/tensors/convolution/complete_bias/y.npy", y_shape);
    let attrs = ConvAttributes::new([2, 2], 2, [3, 3], [2, 2, 2, 2], [2, 2]);
    let my_y = convolution(x, w, Some(b), attrs);
    let err = y.sub(my_y).mapv(|x| x.abs()).mean().unwrap();
    // println!("avg error = {}", err);
    assert!(err < 1e-5);
}

#[test]
fn test_clip() {
    let x_shape = [2, 4, 5, 5];
    let y_shape = [2, 4, 5, 5];
    let x = load4d("tests/tensors/clip/x.npy", x_shape);
    let y = load4d("tests/tensors/clip/y.npy", y_shape);
    let attrs = ClipAttributes::new(0.2, 0.8);
    let my_y = clip(x, attrs);
    let err = y.sub(my_y).mapv(|x| x.abs()).mean().unwrap();
    // println!("avg error = {}", err);
    assert!(err < 1e-5);
}

#[test]
fn test_add() {
    let x_shape = [2, 4, 5, 5];
    let y_shape = [2, 4, 5, 5];
    let z_shape = [2, 4, 5, 5];
    let x = load4d("tests/tensors/add/x.npy", x_shape);
    let y = load4d("tests/tensors/add/y.npy", y_shape);
    let z = load4d("tests/tensors/add/z.npy", z_shape);
    let my_z = add(x, y);
    let err = z.sub(my_z).mapv(|x| x.abs()).mean().unwrap();
    // println!("avg error = {}", err);
    assert!(err < 1e-5);
}

#[test]
fn test_shape() {
    let x_shape = [64, 32, 5, 5];
    let x = Array4::<f32>::from_shape_fn(x_shape, |_| 0.0);
    let my_x_shape = shape(x);
    assert_eq!(my_x_shape.to_vec(), x_shape);
}

#[test]
fn test_gather() {
    let x = Array1::from_vec(vec![64, 32, 5, 5]);
    let attrs = GatherAttributes::new(0);
    let gathered = gather(x, 0, attrs);
    let expected = arr0(64);
    assert_eq!(gathered, expected);
}

#[test]
fn test_unqueeze() {
    let x = arr0(64);
    let attrs = UnsqueezeAttributes::new(0);
    let unsqueezed = unsqueeze(x, attrs);
    let expected = Array1::<usize>::from_vec(vec![64]);
    assert_eq!(unsqueezed, expected);
}

#[test]
fn test_concat() {
    let x1 = Array1::from_vec(vec![64]);
    let x2 = Array1::from_vec(vec![-1]);
    let attrs = ConcatAttributes::new(0);
    let concated = concat(vec![x1, x2], attrs);
    let expected = Array1::<i64>::from_vec(vec![64, -1]);
    assert_eq!(concated, expected);
}

#[test]
fn test_global_average_pool() {
    let x_shape = [8, 4, 5, 5];
    let y_shape = [8, 4, 1, 1];
    let x = load4d("tests/tensors/glob_avg_pool/x.npy", x_shape);
    let y = load4d("tests/tensors/glob_avg_pool/y.npy", y_shape);
    let my_y = global_average_pool(x);
    let err = y.sub(my_y).mapv(|x| x.abs()).mean().unwrap();
    // println!("avg error = {}", err);
    assert!(err < 1e-5);
}

#[test]
fn test_reshape() {
    let x_shape = [8, 4, 1, 1];
    let y_shape = [8, 4];
    let shape = Array1::from_vec(vec![8, -1]);
    let x = load4d("tests/tensors/reshape/x.npy", x_shape);
    let y = load2d("tests/tensors/reshape/y.npy", y_shape);
    let my_y = reshape(x, shape);
    let err = y.sub(my_y).mapv(|x| x.abs()).mean().unwrap();
    // println!("avg error = {}", err);
    assert!(err < 1e-5);
}

#[test]
fn test_gemm() {
    /*
        A -> (2, 3)
        B -> (3, 4)
        C -> (4)
        Y -> (2, 4)
     */
    let a: Array2<f32> = arr2(&[[2.0, 3.0, 4.0], [4.0, 5.0, 6.0]]);
    let b: Array2<f32> = arr2(&[[0.1, 1.0, 10.0, 100.0], [0.2, 2.0, 20.0, 200.0], [0.3, 3.0, 30.0, 300.0]]);
    let c: Array2<f32> = arr2(&[[0.5, -0.5, 0.5, -0.5]]);
    let attrs = GemmAttributes::new(2.0, 0.1, 0, 0);
    let y_shape = [2, 4];
    let y = load2d("tests/tensors/gemm/y.npy", y_shape);
    let my_y = gemm(a, b, c, attrs);
    let err = y.sub(my_y).mapv(|x| x.abs()).mean().unwrap();
    // println!("avg error = {}", err);
    assert!(err < 1e-5);
}
