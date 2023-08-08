use ndarray::{ArrayD, Ix0, Ix2, IxDyn};
use std::ops::Add;

use super::OperationError;

pub struct ClipAttributes {
    min: f32,
    max: f32,
}

impl ClipAttributes {
    pub fn new(min: f32, max: f32) -> Self {
        Self { min, max }
    }
}

pub fn clip(x: ArrayD<f32>, attrs: ClipAttributes) -> ArrayD<f32> {
    let ClipAttributes {
        min: min_v,
        max: max_v,
    } = attrs;
    x.mapv(|x| x.max(min_v).min(max_v))
}

pub fn add(x: ArrayD<f32>, y: ArrayD<f32>) -> Result<ArrayD<f32>, OperationError> {
    if x.shape() == y.shape() {
        Ok(x.add(y))
    } else {
        Err(OperationError::UnmatchingShape(format!("{:?}", x.shape()), format!("{:?}", y.shape())))
    }
}

pub fn shape(x: ArrayD<f32>) -> ArrayD<usize> {
    ArrayD::<usize>::from_shape_vec(IxDyn(&[x.shape().len()]), x.shape().to_vec()).unwrap()
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

pub fn gather(x: ArrayD<usize>, index: usize, attrs: GatherAttributes) -> ArrayD<usize> {
    assert!(attrs.axes == 0); // this is the only use case we are interested in
    ArrayD::<usize>::from_shape_fn(IxDyn(&[]), |_| x[[index]])
}

pub type UnsqueezeAttributes = GatherAttributes;

pub fn unsqueeze(x: ArrayD<usize>, attrs: UnsqueezeAttributes) -> ArrayD<usize> {
    assert!(attrs.axes == 0); // this is the only use case we are interested in
    ArrayD::<usize>::from_shape_vec(
        IxDyn(&[1]),
        vec![x.into_dimensionality::<Ix0>().unwrap().into_scalar()],
    )
    .expect("Unsqueeze failed")
}

pub type ConcatAttributes = GatherAttributes;

pub fn concat(x: Vec<ArrayD<i64>>, attrs: ConcatAttributes) -> ArrayD<i64> {
    assert!(!x.is_empty());
    assert!(attrs.axes == 0); // this is the only use case we are interested in
    ArrayD::from_shape_fn(IxDyn(&[x.len()]), |i| x[i[0]][[0]])
}

pub fn global_average_pool(x: ArrayD<f32>) -> ArrayD<f32> {
    let [batch_size, channels, height, width] = *x.shape() else {todo!("Failed global average pool")};
    ArrayD::from_shape_fn(IxDyn(&[batch_size, channels, 1, 1]), |idx| {
        let mut accumulator = 0.0;
        for i in 0..height {
            for j in 0..width {
                accumulator += x[[idx[0], idx[1], i, j]];
            }
        }
        accumulator / (height * width) as f32
    })
}

pub fn reshape(x: ArrayD<f32>, shape: ArrayD<i64>) -> ArrayD<f32> {
    let mut myshape: [usize; 2] = [0, 0];
    let xshape = x.shape();
    for i in 0..shape.len() {
        if myshape[i] == 0 {
            myshape[i] = xshape[i];
        } else if shape[i] == -1 {
            myshape[i] = xshape[i..].iter().product::<usize>();
        } else {
            myshape[i] = shape[i] as usize;
        }
    }
    x.into_shape(IxDyn(&myshape)).unwrap()
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

pub fn gemm(a: ArrayD<f32>, b: ArrayD<f32>, c: ArrayD<f32>, attrs: GemmAttributes) -> ArrayD<f32> {
    let GemmAttributes {
        alpha,
        beta,
        trans_a,
        trans_b,
    } = attrs;
    let act_a = if trans_a == 0 {
        a.into_dimensionality::<Ix2>().unwrap()
    } else {
        a.into_dimensionality::<Ix2>().unwrap().t().to_owned()
    };
    let act_b = if trans_b == 0 {
        b.into_dimensionality::<Ix2>().unwrap()
    } else {
        b.into_dimensionality::<Ix2>().unwrap().t().to_owned()
    };

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
    x: ArrayD<f32>,
    scale: ArrayD<f32>,
    b: ArrayD<f32>,
    mean: ArrayD<f32>,
    var: ArrayD<f32>,
    attrs: BatchNormAttributes,
) -> ArrayD<f32> {
    let BatchNormAttributes {
        epsilon,
        momentum: _,
        spatial,
    } = attrs;
    assert!(spatial != 0); // this is the only use case we are interested in
    let mean = mean
        .to_shape(IxDyn(&[1, x.shape()[1], 1, 1]))
        .unwrap()
        .to_owned();
    let b = b
        .to_shape(IxDyn(&[1, x.shape()[1], 1, 1]))
        .unwrap()
        .to_owned();
    let scale = scale
        .to_shape(IxDyn(&[1, x.shape()[1], 1, 1]))
        .unwrap()
        .to_owned();
    let var = var
        .to_shape(IxDyn(&[1, x.shape()[1], 1, 1]))
        .unwrap()
        .to_owned();

    let x_normalized = (x - mean) / (var + epsilon).mapv(|v| v.sqrt());
    scale * x_normalized + b
}

pub fn relu(x: ArrayD<f32>) -> ArrayD<f32> {
    x.mapv(|v| v.max(0.0))
}
