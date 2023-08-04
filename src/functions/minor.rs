use ndarray::{Array1, Array2, Array4};
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

pub fn gather(x: Array1<usize>, index: usize, attrs: GatherAttributes) -> Array1<usize> {
    assert!(attrs.axes == 0); // this is the only use case we are interested in
    Array1::<usize>::from_vec(vec![x[[index]]])
}

type UnsqueezeAttributes = GatherAttributes;

pub fn unsqueeze(x: Array1<usize>, attrs: UnsqueezeAttributes) -> Array2<usize> {
    assert!(attrs.axes == 0); // this is the only use case we are interested in
    let v = x.to_vec();
    Array2::<usize>::from_shape_vec([0, v.len()], v).expect("Unsqueeze failed")
}

type ConcatAttributes = GatherAttributes;

pub fn concat(x: Vec<Array2<usize>>, attrs: ConcatAttributes) -> Array2<usize> {
    assert!(!x.is_empty());
    assert!(attrs.axes == 0); // this is the only use case we are interested in
                              //stack![Axis(attrs.axes), x[0], x[1]]
    let mut y = Array2::from_shape_fn([x.len(), x[0].shape()[0]], |_| 0);
    for i in 0..x.len() {
        for j in 0..x[i].shape()[1] {
            y[[i, j]] = x[i][[0, j]];
        }
    }
    y
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
        accumulator / (height*width) as f32
    })
}
