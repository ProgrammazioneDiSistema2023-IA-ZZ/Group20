use ndarray::{Array1, ArrayD, Ix0, Ix2, IxDyn};
use std::ops::Add;

use crate::{
    operators::{
        BatchNormAttributes, ClipAttributes, ConcatAttributes, ConvAttributes, GatherAttributes,
        GemmAttributes, MaxPoolAttributes, UnsqueezeAttributes,
    },
    tensor::TypeToTensorDataType,
};

use super::{OperationError, Provider};

pub struct NaiveProvider;
impl Provider for NaiveProvider {
    fn name(&self) -> &str {
        "Naive"
    }

    fn version(&self) -> u64 {
        7
    }

    fn add(x: ArrayD<f32>, y: ArrayD<f32>) -> Result<ArrayD<f32>, OperationError> {
        if x.shape() == y.shape() {
            Ok(x.add(y))
        } else {
            Err(OperationError::UnexpectedShape(
                format!("{:?}", x.shape()),
                format!("{:?}", y.shape()),
            ))
        }
    }

    fn relu(x: ArrayD<f32>) -> ArrayD<f32> {
        x.mapv(|v| v.max(0.0))
    }

    fn clip(x: ArrayD<f32>, attrs: ClipAttributes) -> ArrayD<f32> {
        let ClipAttributes {
            min: min_v,
            max: max_v,
        } = attrs;
        x.mapv(|x| x.max(min_v).min(max_v))
    }

    fn shape(x: ArrayD<f32>) -> ArrayD<i64> {
        ArrayD::<i64>::from_shape_vec(
            IxDyn(&[x.ndim()]),
            x.shape().iter().map(|e| *e as i64).collect(),
        )
        .unwrap()
    }

    fn gather(
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

    fn unsqueeze(
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
    fn concat<T>(x: Vec<ArrayD<T>>, attrs: ConcatAttributes) -> Result<ArrayD<T>, OperationError>
    where
        T: TypeToTensorDataType + Copy,
    {
        if attrs.axes != 0 {
            Err(OperationError::UnsupportedOperator)
        } else if x.is_empty() {
            Err(OperationError::InvalidOperator)
        } else {
            Ok(ArrayD::from_shape_fn(IxDyn(&[x.len()]), |i| x[i[0]][[0]]))
        }
    }
    fn global_average_pool(x: ArrayD<f32>) -> Result<ArrayD<f32>, OperationError> {
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

    fn reshape(x: ArrayD<f32>, shape: ArrayD<i64>) -> Result<ArrayD<f32>, OperationError> {
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

    fn gemm(
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

    fn batch_norm(
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

    fn max_pool(x: ArrayD<f32>, attrs: MaxPoolAttributes) -> Result<ArrayD<f32>, OperationError> {
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
        let out_shape = [batch_size, in_chans, out_height, out_width];

        // declaration of tensor bounds considering padding
        let tens_hs: i64 = 0i64 - (pad_hs as i64);
        let tens_ws: i64 = 0i64 - (pad_ws as i64);
        let tens_he: i64 = ((height + pad_he) - kern_h + 1) as i64; // subtracting kernel size to consider valid windows only
        let tens_we: i64 = ((width + pad_we) - kern_w + 1) as i64;

        // result tensor
        let mut output: ArrayD<f32> = ArrayD::<f32>::from_elem(IxDyn(&out_shape), 0.0);
        for batch in 0..batch_size {
            for channel in 0..in_chans {
                // iterate over the input tensor with the specified stride
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
                                    x[[batch, channel, input_row as usize, input_col as usize]],
                                )
                            }
                        }
                        // compute output tensor indexes and update the corresponding value
                        let out_row = (ext_row + pad_hs as i64) as usize / stride_h;
                        let out_col = (ext_col + pad_ws as i64) as usize / stride_w;
                        output[[batch, channel, out_row, out_col]] = result;
                    }
                }
            }
        }
        Ok(output)
    }

    fn conv(
        x: ArrayD<f32>,
        weights: ArrayD<f32>,
        bias: Option<Array1<f32>>,
        attrs: ConvAttributes,
    ) -> Result<ArrayD<f32>, OperationError> {
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
        let mut output: ArrayD<f32> = ArrayD::<f32>::from_elem(IxDyn(&out_shape), 0.0);

        for batch in 0..batch_size {
            for featmap in 0..n_featmaps {
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
                                    accumulator += x
                                        [[batch, channel, input_row as usize, input_col as usize]]
                                        * weights[[featmap, group_channel, kern_row, kern_col]];
                                }
                            }
                        }
                        // compute output tensor indexes and update the corresponding value
                        let out_row = (ext_row + pad_hs as i64) as usize / stride_h;
                        let out_col = (ext_col + pad_ws as i64) as usize / stride_w;
                        output[[batch, featmap, out_row, out_col]] = accumulator;
                    }
                }
            }
        }
        Ok(output)
    }
}
