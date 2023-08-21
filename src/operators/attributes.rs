use crate::tensor::TensorData;

#[derive(Debug, Clone)]
pub struct ConvAttributes {
    // assuming 4D tensors
    pub dilations: [usize; 2],
    pub group: usize,
    pub kernel_shape: [usize; 2],
    pub pads: [usize; 4],
    pub strides: [usize; 2],
}

#[derive(Debug, Clone)]
pub struct MaxPoolAttributes {
    pub kernel_shape: [usize; 2],
    pub pads: [usize; 4],
    pub strides: [usize; 2],
}
#[derive(Debug, Clone)]
pub struct ClipAttributes {
    pub min: f32,
    pub max: f32,
}

#[derive(Default, Debug, Clone)]
pub struct GatherAttributes {
    pub axes: usize,
}

#[derive(Debug, Clone)]
pub struct BatchNormAttributes {
    pub epsilon: f32,
    pub momentum: f32, // not used during inference
    pub spatial: i64,
}

pub type UnsqueezeAttributes = GatherAttributes;

pub type ConcatAttributes = GatherAttributes;

impl ConvAttributes {
    pub fn new(
        dilations: [usize; 2],
        group: usize,
        kernel_shape: [usize; 2],
        pads: [usize; 4],
        strides: [usize; 2],
    ) -> Self {
        Self {
            dilations,
            group,
            kernel_shape,
            pads,
            strides,
        }
    }
}

impl Default for ConvAttributes {
    fn default() -> Self {
        ConvAttributes {
            dilations: [1, 1],
            group: 1,
            kernel_shape: [3, 3],
            pads: [0, 0, 0, 0],
            strides: [1, 1],
        }
    }
}

impl MaxPoolAttributes {
    pub fn new(kernel_shape: [usize; 2], pads: [usize; 4], strides: [usize; 2]) -> Self {
        Self {
            kernel_shape,
            pads,
            strides,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GemmAttributes {
    pub alpha: f32,
    pub beta: f32,
    pub trans_a: i64,
    pub trans_b: i64,
}

impl ClipAttributes {
    pub fn new(min: f32, max: f32) -> Self {
        Self { min, max }
    }
}

impl GatherAttributes {
    pub fn new(axes: usize) -> Self {
        Self { axes }
    }
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

impl BatchNormAttributes {
    pub fn new(epsilon: f32, momentum: f32, spatial: i64) -> Self {
        Self {
            epsilon,
            momentum,
            spatial,
        }
    }
}
