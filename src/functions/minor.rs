use ndarray::{Array0, Array1, Array2, Array4};
use std::ops::Add;

pub struct ClipAttributes {
    min: f32,
    max: f32,
}

impl ClipAttributes {
    pub fn new(min: f32, max: f32) -> Self {
        Self { min, max }
    }
}

pub fn clip(x: Array4<f32>, attrs: ClipAttributes) -> Array4<f32> {
    let ClipAttributes {
        min: min_v,
        max: max_v,
    } = attrs;
    x.mapv(|x| x.max(min_v).min(max_v))
}

pub fn add(x: Array4<f32>, y: Array4<f32>) -> Array4<f32> {
    x.add(y)
}

pub fn shape(x: Array4<f32>) -> Array1<usize> {
    Array1::<usize>::from_vec(x.shape().to_vec())
}

#[derive(Default)]
pub struct GatherAttributes {
    axes: usize,
}

impl GatherAttributes {
    pub fn new(axes: usize) -> Self {
        Self { axes }
    }
}

pub fn gather(x: Array1<usize>, index: usize, attrs: GatherAttributes) -> Array0<usize> {
    assert!(attrs.axes == 0); // this is the only use case we are interested in
    Array0::<usize>::from_shape_fn([], |_| x[[index]])
    //    Array0::<usize>::from_vec(vec![x[[index]]])
}

pub type UnsqueezeAttributes = GatherAttributes;

pub fn unsqueeze(x: Array0<usize>, attrs: UnsqueezeAttributes) -> Array1<usize> {
    assert!(attrs.axes == 0); // this is the only use case we are interested in
    Array1::<usize>::from_shape_vec([1], vec![x.into_scalar()]).expect("Unsqueeze failed")
}

pub type ConcatAttributes = GatherAttributes;

pub fn concat(x: Vec<Array1<i64>>, attrs: ConcatAttributes) -> Array1<i64> {
    assert!(!x.is_empty());
    assert!(attrs.axes == 0); // this is the only use case we are interested in
    Array1::from_shape_fn([x.len()], |i| x[i][[0]])
}

pub fn global_average_pool(x: Array4<f32>) -> Array4<f32> {
    let [batch_size, channels, height, width] = *x.shape() else {todo!("Failed global average pool")};
    Array4::from_shape_fn([batch_size, channels, 1, 1], |(bs, c, _, _)| {
        let mut accumulator = 0.0;
        for i in 0..height {
            for j in 0..width {
                accumulator += x[[bs, c, i, j]];
            }
        }
        accumulator / (height * width) as f32
    })
}

pub fn reshape(x: Array4<f32>, shape: Array1<i64>) -> Array2<f32> {
    let mut myshape: [usize; 2] = [0, 0];
    let xshape = x.shape();
    for i in 0..shape.len() {
        if myshape[i] == 0 {
            myshape[i] = xshape[i] as usize;
        } else if shape[i] == -1 {
            myshape[i] = xshape[i..].iter().product::<usize>();
        } else {
            myshape[i] = shape[i] as usize;
        }
    }
    x.into_shape(myshape).unwrap()
}

pub struct GemmAttributes {
    alpha: f32,
    beta: f32,
    trans_a: i64,
    trans_b: i64,
}

impl GemmAttributes {
    pub fn new(alpha: f32, beta: f32, trans_a: i64, trans_b: i64) -> Self {
        Self {
            alpha,
            beta,
            trans_a,
            trans_b,
        }
    }
}

pub fn gemm(a: Array2<f32>, b: Array2<f32>, c: Array2<f32>, attrs: GemmAttributes) -> Array2<f32> {
    let GemmAttributes {
        alpha,
        beta,
        trans_a,
        trans_b,
    } = attrs;
    let act_a = if trans_a == 0 { a } else { a.t().to_owned() };
    let act_b = if trans_b == 0 { b } else { b.t().to_owned() };
    let ab = alpha * act_a.dot(&act_b);
    ab + beta * c
}

#[allow(dead_code)]
pub struct BatchNormAttributes {
    epsilon: f32,
    momentum: f32, // not used during inference
    spatial: i64,
}

impl BatchNormAttributes {
    pub fn new(epsilon: f32, momentum: f32, spatial: i64) -> Self {
        Self {
            epsilon,
            momentum,
            spatial,
        }
    }
}

pub fn batch_norm(
    x: Array4<f32>,
    scale: Array1<f32>,
    b: Array1<f32>,
    mean: Array1<f32>,
    var: Array1<f32>,
    attrs: BatchNormAttributes,
) -> Array4<f32> {
    let BatchNormAttributes {
        epsilon,
        momentum: _,
        spatial,
    } = attrs;
    assert!(spatial != 0); // this is the only use case we are interested in
    let mean = Array4::from_shape_vec([1, x.shape()[1], 1, 1], mean.to_vec()).unwrap();
    let b = Array4::from_shape_vec([1, x.shape()[1], 1, 1], b.to_vec()).unwrap();
    let scale = Array4::from_shape_vec([1, x.shape()[1], 1, 1], scale.to_vec()).unwrap();
    let var = Array4::from_shape_vec([1, x.shape()[1], 1, 1], var.to_vec()).unwrap();

    let x_normalized = (x - mean) / (var + epsilon).mapv(|v| v.sqrt());
    let y = scale * x_normalized + b;
    y
}

pub fn relu(x: Array4<f32>) -> Array4<f32> {
    x.mapv(|v| v.max(0.0))
}
