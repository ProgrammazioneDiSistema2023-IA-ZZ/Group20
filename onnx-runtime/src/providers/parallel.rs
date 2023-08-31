use ndarray::Axis;
use ndarray::{Array1, ArrayD, Ix2, IxDyn};
use rayon::prelude::*;
use rayon::ThreadPool;
use std::sync::Mutex;

use crate::{
    operators::{
        BatchNormAttributes, ClipAttributes, ConcatAttributes, ConvAttributes, GatherAttributes,
        GemmAttributes, MaxPoolAttributes, UnsqueezeAttributes,
    },
    tensor::TypeToTensorDataType,
};

use super::{NaiveProvider, OperationError, Provider};

pub struct ParNaiveProvider;
impl Provider for ParNaiveProvider {
    fn name(&self) -> &str {
        "ParNaive"
    }

    fn version(&self) -> u64 {
        7
    }

    fn add(
        _thread_pool: &ThreadPool,
        x: ArrayD<f32>,
        y: ArrayD<f32>,
    ) -> Result<ArrayD<f32>, OperationError> {
        NaiveProvider::add(_thread_pool, x, y)
    }

    fn relu(_thread_pool: &ThreadPool, x: ArrayD<f32>) -> ArrayD<f32> {
        NaiveProvider::relu(_thread_pool, x)
    }

    fn clip(_thread_pool: &ThreadPool, x: ArrayD<f32>, attrs: ClipAttributes) -> ArrayD<f32> {
        NaiveProvider::clip(_thread_pool, x, attrs)
    }

    fn shape(_thread_pool: &ThreadPool, x: ArrayD<f32>) -> ArrayD<i64> {
        NaiveProvider::shape(_thread_pool, x)
    }

    fn gather(
        _thread_pool: &ThreadPool,
        x: ArrayD<usize>,
        index: usize,
        attrs: GatherAttributes,
    ) -> Result<ArrayD<usize>, OperationError> {
        NaiveProvider::gather(_thread_pool, x, index, attrs)
    }

    fn unsqueeze(
        _thread_pool: &ThreadPool,
        x: ArrayD<usize>,
        attrs: UnsqueezeAttributes,
    ) -> Result<ArrayD<usize>, OperationError> {
        NaiveProvider::unsqueeze(_thread_pool, x, attrs)
    }
    fn concat<T>(
        _thread_pool: &ThreadPool,
        x: Vec<ArrayD<T>>,
        attrs: ConcatAttributes,
    ) -> Result<ArrayD<T>, OperationError>
    where
        T: TypeToTensorDataType + Copy,
    {
        NaiveProvider::concat(_thread_pool, x, attrs)
    }
    fn global_average_pool(
        _thread_pool: &ThreadPool,
        x: ArrayD<f32>,
    ) -> Result<ArrayD<f32>, OperationError> {
        NaiveProvider::global_average_pool(_thread_pool, x)
    }

    fn reshape(
        _thread_pool: &ThreadPool,
        x: ArrayD<f32>,
        shape: ArrayD<i64>,
    ) -> Result<ArrayD<f32>, OperationError> {
        NaiveProvider::reshape(_thread_pool, x, shape)
    }

    fn gemm(
        thread_pool: &ThreadPool,
        a: ArrayD<f32>,
        b: ArrayD<f32>,
        c: ArrayD<f32>,
        attrs: GemmAttributes,
    ) -> Result<ArrayD<f32>, OperationError> {
        if thread_pool.current_num_threads() == 1 {
            return NaiveProvider::gemm(thread_pool, a, b, c, attrs);
        }
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
        let mut term1 = None;
        let mut term2 = None;
        thread_pool.scope(|s| {
            s.spawn(|_| term1 = Some(alpha * act_a.dot(&act_b)));
            s.spawn(|_| term2 = Some(beta * act_c));
        });
        Ok(
            (term1.expect("term1 unavailable") + term2.expect("term2 unavailable"))
                .into_dimensionality::<IxDyn>()
                .unwrap(),
        )
    }

    fn batch_norm(
        thread_pool: &ThreadPool,
        x: ArrayD<f32>,
        scale: ArrayD<f32>,
        b: ArrayD<f32>,
        mean: ArrayD<f32>,
        var: ArrayD<f32>,
        attrs: BatchNormAttributes,
    ) -> Result<ArrayD<f32>, OperationError> {
        NaiveProvider::batch_norm(thread_pool, x, scale, b, mean, var, attrs)
    }

    fn max_pool(
        thread_pool: &ThreadPool,
        x: ArrayD<f32>,
        attrs: MaxPoolAttributes,
    ) -> Result<ArrayD<f32>, OperationError> {
        if thread_pool.current_num_threads() == 1 {
            return NaiveProvider::max_pool(thread_pool, x, attrs);
        }
        // checks
        let [batch_size, in_chans, height, width] = *x.shape() else {
            return Err(OperationError::WrongDim(4, x.shape().len()));
        };
        let MaxPoolAttributes {
            kernel_shape: [kern_h, kern_w],
            pads: [pad_hs, pad_ws, pad_he, pad_we],
            strides: [stride_h, stride_w],
        } = attrs;
        let out_height = 1 + ((height + pad_hs + pad_he) - kern_h) / stride_h;
        let out_width = 1 + ((width + pad_ws + pad_we) - kern_w) / stride_w;
        // let out_shape = [batch_size, in_chans, out_height, out_width];

        // declaration of tensor bounds considering padding
        let tens_hs: i64 = 0i64 - (pad_hs as i64);
        let tens_ws: i64 = 0i64 - (pad_ws as i64);
        let tens_he: i64 = ((height + pad_he) - kern_h + 1) as i64; // subtracting kernel size to consider valid windows only
        let tens_we: i64 = ((width + pad_we) - kern_w + 1) as i64;

        // result tensor
        let output = thread_pool.install(|| {
            (0..batch_size)
                .into_iter()
                .map(|batch| {
                    (0..in_chans)
                        .into_par_iter()
                        .map(|channel| {
                            let mut subview = ArrayD::zeros(IxDyn(&[1, out_height, out_width]));
                            for ext_row in (tens_hs..tens_he).step_by(stride_h) {
                                for ext_col in (tens_ws..tens_we).step_by(stride_w) {
                                    // declaration of kernel window bounds
                                    let win_sh = ext_row;
                                    let win_sw = ext_col;
                                    let win_eh = ext_row + kern_h as i64; // actual kernel size takes into account the dilation
                                    let win_ew = ext_col + kern_w as i64;

                                    let mut result: f32 = f32::MIN;
                                    // iterate over the window defined by the kernel
                                    for input_row in win_sh.max(0)..win_eh.min(height as i64) {
                                        for input_col in win_sw.max(0)..win_ew.min(width as i64) {
                                            result = result.max(
                                                x[[
                                                    batch,
                                                    channel,
                                                    input_row as usize,
                                                    input_col as usize,
                                                ]],
                                            )
                                        }
                                    }
                                    // compute output tensor indexes and update the corresponding value
                                    let out_row = (ext_row + pad_hs as i64) as usize / stride_h;
                                    let out_col = (ext_col + pad_ws as i64) as usize / stride_w;
                                    subview[[0, out_row, out_col]] = result;
                                }
                            }
                            subview
                        })
                        .collect::<Vec<_>>()
                        .iter()
                        .fold(
                            ArrayD::<f32>::zeros(IxDyn(&[0, out_height, out_width])),
                            |mut acc, x| {
                                acc.append(Axis(0), x.view()).expect("Cannot append 1");
                                acc
                            },
                        )
                        .insert_axis(Axis(0))
                })
                .fold(
                    ArrayD::<f32>::zeros(IxDyn(&[0, in_chans, out_height, out_width])),
                    |mut acc, x| {
                        acc.append(Axis(0), x.view()).expect("Cannot append 2");
                        acc
                    },
                )
        });
        Ok(output)
    }

    fn conv(
        thread_pool: &ThreadPool,
        x: ArrayD<f32>,
        weights: ArrayD<f32>,
        bias: Option<Array1<f32>>,
        attrs: ConvAttributes,
    ) -> Result<ArrayD<f32>, OperationError> {
        if thread_pool.current_num_threads() == 1 {
            return NaiveProvider::conv(thread_pool, x, weights, bias, attrs);
        }
        // checks
        let [batch_size, in_chans, height, width] = *x.shape() else {
            return Err(OperationError::WrongDim(4, x.shape().len()));
        };
        if weights.shape().len() != 4 {
            return Err(OperationError::WrongDim(4, weights.shape().len()));
        }
        if weights.shape()[2..] != attrs.kernel_shape {
            return Err(OperationError::WrongShape(
                format!(
                    "[*, *, {}, {}]",
                    attrs.kernel_shape[0], attrs.kernel_shape[1]
                ),
                format!("[*, *, {}, {}]", weights.shape()[2], weights.shape()[3]),
            ));
        }
        let n_featmaps = weights.shape()[0];
        let bias = bias.unwrap_or(Array1::from_vec(vec![0.0; n_featmaps]));
        if bias.shape()[0] != n_featmaps {
            return Err(OperationError::WrongShape(
                format!("[{}]", n_featmaps),
                format!("[{}]", bias.shape()[0]),
            ));
        }

        let ConvAttributes {
            // w = width, h = height; s = start, e = end
            dilations: [dilat_h, dilat_w],
            group: n_groups,
            kernel_shape: [kern_h, kern_w],
            pads: [pad_hs, pad_ws, pad_he, pad_we],
            strides: [stride_h, stride_w],
        } = attrs;
        let output_group_size = n_featmaps / n_groups;
        let input_group_size = in_chans / n_groups;
        let out_height = 1 + ((height + pad_hs + pad_he) - (dilat_h * (kern_h - 1) + 1)) / stride_h;
        let out_width = 1 + ((width + pad_ws + pad_we) - (dilat_w * (kern_w - 1) + 1)) / stride_w;
        let out_shape = [batch_size, n_featmaps, out_height, out_width];

        // compute actual kernel size, i.e. kernel size considering the dilation
        let act_kern_h = (dilat_h * (kern_h - 1) + 1) as i64;
        let act_kern_w = (dilat_w * (kern_w - 1) + 1) as i64;

        // declaration of tensor bounds considering padding
        let tens_hs: i64 = 0_i64 - (pad_hs as i64);
        let tens_ws: i64 = 0_i64 - (pad_ws as i64);
        let tens_he: i64 = (height + pad_he) as i64 - act_kern_h + 1; // subtracting kernel size to consider valid windows only
        let tens_we: i64 = (width + pad_we) as i64 - act_kern_w + 1;

        // result tensor
        let output: Mutex<ArrayD<f32>> =
            Mutex::new(ArrayD::<f32>::from_elem(IxDyn(&out_shape), 0.0));
        thread_pool.install(|| {
            for batch in 0..batch_size {
                (0..n_featmaps).into_par_iter().for_each(|featmap| {
                    // get the group index of the feature map and compute input channel group bounds
                    let group: usize = featmap / output_group_size;
                    let group_s = group * input_group_size;
                    let group_e = group_s + input_group_size;

                    // iterate over the input tensor with the specified stride
                    for ext_row in (tens_hs..tens_he).step_by(stride_h) {
                        for ext_col in (tens_ws..tens_we).step_by(stride_w) {
                            // declaration of kernel window bounds
                            let win_hs = ext_row;
                            let win_ws = ext_col;
                            let win_he = ext_row + act_kern_h; // actual kernel size takes into account the dilation
                            let win_we = ext_col + act_kern_w;

                            // kern_row and kern_col used to access the kernel
                            let mut accumulator: f32 = bias[[featmap]];
                            // iterate over all input channels
                            for channel in group_s..group_e {
                                let group_channel = channel % input_group_size;
                                // iterate over the window defined by the kernel with the specified dilation
                                for (kern_row, input_row) in
                                    (win_hs..win_he).step_by(dilat_h).enumerate()
                                {
                                    if input_row < 0 || input_row >= height as i64 {
                                        continue;
                                    }
                                    for (kern_col, input_col) in
                                        (win_ws..win_we).step_by(dilat_w).enumerate()
                                    {
                                        if input_col < 0 || input_col >= width as i64 {
                                            continue;
                                        }
                                        accumulator += x[[
                                            batch,
                                            channel,
                                            input_row as usize,
                                            input_col as usize,
                                        ]] * weights
                                            [[featmap, group_channel, kern_row, kern_col]];
                                    }
                                }
                            }
                            // compute output tensor indexes and update the corresponding value
                            let out_row = (ext_row + pad_hs as i64) as usize / stride_h;
                            let out_col = (ext_col + pad_ws as i64) as usize / stride_w;
                            let mut guard = output.lock().expect("Failed to obtain lock");
                            guard[[batch, featmap, out_row, out_col]] = accumulator;
                        }
                    }
                });
            }
        });
        let result = output.into_inner().expect("Failed to obtain lock");
        Ok(result)
    }
}
