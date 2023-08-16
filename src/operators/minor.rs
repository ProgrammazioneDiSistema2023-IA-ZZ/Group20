use ndarray::{ArrayD, Ix0, Ix2, IxDyn};
use std::ops::Add;

use crate::tensor::TensorData;

use super::OperationError;

#[derive(Debug, Clone)]
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
        Err(OperationError::UnexpectedShape(
            format!("{:?}", x.shape()),
            format!("{:?}", y.shape()),
        ))
    }
}

pub fn shape(x: ArrayD<f32>) -> ArrayD<usize> {
    ArrayD::<usize>::from_shape_vec(IxDyn(&[x.ndim()]), x.shape().to_vec()).unwrap()
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct GatherInputs {
    index: TensorData,
}

impl GatherInputs {
    pub fn new(index: TensorData) -> Self {
        Self { index }
    }
}

#[derive(Default, Debug, Clone)]
pub struct GatherAttributes {
    axes: usize,
}

impl GatherAttributes {
    pub fn new(axes: usize) -> Self {
        Self { axes }
    }
}

pub fn gather(
    x: ArrayD<usize>,
    index: usize,
    attrs: GatherAttributes,
) -> Result<ArrayD<usize>, OperationError> {
    assert!(attrs.axes == 0); // this is the only use case we are interested in
    if attrs.axes != 0 {
        Err(OperationError::UnsupportedOperator)
    } else if x.ndim() != 1 {
        Err(OperationError::WrongDim(1, x.ndim()))
    } else if index >= x.shape()[0] {
        Err(OperationError::InvalidOperator)
    } else {
        Ok(ArrayD::<usize>::from_shape_fn(IxDyn(&[]), |_| x[[index]]))
    }
}

pub type UnsqueezeAttributes = GatherAttributes;

pub fn unsqueeze(
    x: ArrayD<usize>,
    attrs: UnsqueezeAttributes,
) -> Result<ArrayD<usize>, OperationError> {
    if attrs.axes != 0 {
        Err(OperationError::UnsupportedOperator)
    } else if x.ndim() != 0 {
        Err(OperationError::WrongDim(0, x.ndim()))
    } else {
        Ok(ArrayD::<usize>::from_shape_vec(
            IxDyn(&[1]),
            vec![x.into_dimensionality::<Ix0>().unwrap().into_scalar()],
        )
        .expect("Unsqueeze failed"))
    }
}

pub type ConcatAttributes = GatherAttributes;

pub fn concat(x: Vec<ArrayD<i64>>, attrs: ConcatAttributes) -> Result<ArrayD<i64>, OperationError> {
    if attrs.axes != 0 {
        Err(OperationError::UnsupportedOperator)
    } else if x.is_empty() {
        Err(OperationError::InvalidOperator)
    } else {
        Ok(ArrayD::from_shape_fn(IxDyn(&[x.len()]), |i| x[i[0]][[0]]))
    }
}

pub fn global_average_pool(x: ArrayD<f32>) -> Result<ArrayD<f32>, OperationError> {
    let [batch_size, channels, height, width] = *x.shape() else {
        return Err(OperationError::WrongDim(4, x.ndim()));
    };
    Ok(ArrayD::from_shape_fn(
        IxDyn(&[batch_size, channels, 1, 1]),
        |idx| {
            let mut accumulator = 0.0;
            for i in 0..height {
                for j in 0..width {
                    accumulator += x[[idx[0], idx[1], i, j]];
                }
            }
            accumulator / (height * width) as f32
        },
    ))
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ReshapeInputs {
    shape: TensorData,
}

impl ReshapeInputs {
    pub fn new(shape: TensorData) -> Self {
        Self { shape }
    }
}

pub fn reshape(x: ArrayD<f32>, shape: ArrayD<i64>) -> Result<ArrayD<f32>, OperationError> {
    if shape.len() != 2 {
        return Err(OperationError::WrongShape(
            "[2]".to_string(),
            format!("[{}]", shape.len()),
        ));
    }
    let mut myshape: [usize; 2] = [0, 0];
    let xshape = x.shape();
    for i in 0..shape.len() {
        if shape[i] == 0 {
            myshape[i] = xshape[i];
        } else if shape[i] == -1 {
            myshape[i] = xshape[i..].iter().product::<usize>();
        } else {
            myshape[i] = shape[i] as usize;
        }
    }
    if xshape.iter().product::<usize>() != myshape.iter().product::<usize>() {
        Err(OperationError::InvalidOperator)
    } else {
        Ok(x.into_shape(IxDyn(&myshape)).unwrap())
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct GemmInputs {
    b: TensorData,
    c: TensorData,
}

impl GemmInputs {
    pub fn new(b: TensorData, c: TensorData) -> Self {
        Self { b, c }
    }
}

#[derive(Debug, Clone)]
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

pub fn gemm(
    a: ArrayD<f32>,
    b: ArrayD<f32>,
    c: ArrayD<f32>,
    attrs: GemmAttributes,
) -> Result<ArrayD<f32>, OperationError> {
    let GemmAttributes {
        alpha,
        beta,
        trans_a,
        trans_b,
    } = attrs;
    if a.ndim() > 2 {
        return Err(OperationError::WrongDim(2, a.ndim()));
    }
    if b.ndim() > 2 {
        return Err(OperationError::WrongDim(2, b.ndim()));
    }
    if c.ndim() > 2 {
        return Err(OperationError::WrongDim(2, c.ndim()));
    }
    let act_c = if c.ndim() == 2 {
        c.into_dimensionality::<Ix2>().unwrap()
    } else {
        let n = c.len();
        c.into_shape(IxDyn(&[1, n]))
            .unwrap()
            .into_dimensionality::<Ix2>()
            .unwrap()
    };

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

    if act_a.shape()[1] != act_b.shape()[0] {
        return Err(OperationError::UnexpectedShape(
            format!("[{}, *]", act_a.shape()[1]),
            format!("[{}, *]", act_b.shape()[0]),
        ));
    }
    if act_b.shape()[1] != act_c.shape()[1] {
        return Err(OperationError::UnexpectedShape(
            format!("[*, {}]", act_b.shape()[1]),
            format!("[*, {}]", act_c.shape()[1]),
        ));
    }
    Ok((alpha * act_a.dot(&act_b) + beta * act_c)
        .into_dimensionality::<IxDyn>()
        .unwrap())
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct BatchNormInputs {
    scale: TensorData,
    b: TensorData,
    mean: TensorData,
    var: TensorData,
}

impl BatchNormInputs {
    pub fn new(scale: TensorData, b: TensorData, mean: TensorData, var: TensorData) -> Self {
        Self {
            scale,
            b,
            mean,
            var,
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
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
) -> Result<ArrayD<f32>, OperationError> {
    // checks
    let dims = vec![
        (1, scale.ndim()),
        (1, b.ndim()),
        (1, mean.ndim()),
        (1, var.ndim()),
        (4, x.ndim()),
    ];
    for dim in dims {
        if dim.0 != dim.1 {
            return Err(OperationError::WrongDim(dim.0, dim.1));
        }
    }
    let dims = vec![
        scale.shape()[0],
        b.shape()[0],
        mean.shape()[0],
        var.shape()[0],
    ];
    for dim in dims {
        if x.shape()[1] != dim {
            return Err(OperationError::WrongShape(
                format!("[{}]", x.shape()[1]),
                format!("[{}]", dim),
            ));
        }
    }

    let BatchNormAttributes {
        epsilon,
        momentum: _,
        spatial,
    } = attrs;
    assert!(spatial != 0); // this is the only use case we are interested in
    let mean = mean.into_shape(IxDyn(&[1, x.shape()[1], 1, 1])).unwrap();
    let b = b.into_shape(IxDyn(&[1, x.shape()[1], 1, 1])).unwrap();
    let scale = scale.into_shape(IxDyn(&[1, x.shape()[1], 1, 1])).unwrap();
    let var = var.into_shape(IxDyn(&[1, x.shape()[1], 1, 1])).unwrap();

    let x_normalized = (x - mean) / (var + epsilon).mapv(|v| v.sqrt());
    Ok(scale * x_normalized + b)
}

pub fn relu(x: ArrayD<f32>) -> ArrayD<f32> {
    x.mapv(|v| v.max(0.0))
}
