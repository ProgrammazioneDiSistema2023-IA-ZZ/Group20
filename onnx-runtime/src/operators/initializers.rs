use crate::tensor::TensorData;

#[derive(Debug, Clone)]
pub struct ConvInits {
    pub weights: TensorData,
    pub bias: Option<TensorData>,
}
#[derive(Debug, Clone)]
pub struct GatherInits {
    pub index: TensorData,
}

#[derive(Debug, Clone)]
pub struct ReshapeInits {
    pub shape: TensorData,
}

#[derive(Debug, Clone)]
pub struct GemmInits {
    pub b: TensorData,
    pub c: TensorData,
}

#[derive(Debug, Clone)]
pub struct BatchNormInits {
    pub scale: TensorData,
    pub bias: TensorData,
    pub mean: TensorData,
    pub var: TensorData,
}

impl ConvInits {
    pub fn new(weights: TensorData, bias: Option<TensorData>) -> Self {
        Self { weights, bias }
    }
}
impl GatherInits {
    pub fn new(index: TensorData) -> Self {
        Self { index }
    }
}

impl ReshapeInits {
    pub fn new(shape: TensorData) -> Self {
        Self { shape }
    }
}

impl GemmInits {
    pub fn new(b: TensorData, c: TensorData) -> Self {
        Self { b, c }
    }
}

impl BatchNormInits {
    pub fn new(scale: TensorData, bias: TensorData, mean: TensorData, var: TensorData) -> Self {
        Self {
            scale,
            bias,
            mean,
            var,
        }
    }
}
