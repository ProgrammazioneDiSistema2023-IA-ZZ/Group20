use ndarray::{Array1, ArrayD, IxDyn};

use super::OperationError;

pub struct ConvAttributes {
    // assuming 4D tensors
    dilations: [usize; 2],
    group: usize,
    kernel_shape: [usize; 2],
    pads: [usize; 4],
    strides: [usize; 2],
}

pub fn conv(
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
                        for (kern_row, input_row) in (win_hs..win_he).step_by(dilat_h).enumerate() {
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

pub struct MaxPoolAttributes {
    kernel_shape: [usize; 2],
    pads: [usize; 4],
    strides: [usize; 2],
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

pub fn max_pool(x: ArrayD<f32>, attrs: MaxPoolAttributes) -> Result<ArrayD<f32>, OperationError> {
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
                            result = result
                                .max(x[[batch, channel, input_row as usize, input_col as usize]])
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
