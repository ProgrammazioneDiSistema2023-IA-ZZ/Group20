use ndarray::Array1;
use ndarray::Array4;

pub struct ConvAttributes {
    // assuming 4D tensors
    dilations: [usize; 2],
    group: usize,
    kernel_shape: [usize; 2],
    pads: [usize; 4],
    strides: [usize; 2],
}

pub fn convolution(
    x: Array4<f32>,
    w: Array4<f32>,
    b: Option<Array1<f32>>,
    attrs: ConvAttributes,
) -> Array4<f32> {
    let [batch_size, in_chans, height, width] = *x.shape() else {todo!("conv2d: invalid input tensor shape")};
    let n_featmaps = w.shape()[0];
    let ConvAttributes {
        dilations: [dh, dw],
        group: ngroups,
        kernel_shape: [kh, kw],
        pads: [phs, pws, phe, pwe],
        strides: [strh, strw],
    } = attrs;
    let output_group_size = n_featmaps / ngroups;
    let input_group_size = in_chans / ngroups;
    let out_height = 1 + ((height + phs + phe) - (dh * (kh - 1) + 1)) / strh;
    let out_width = 1 + ((width + pws + pwe) - (dw * (kw - 1) + 1)) / strw;
    let out_shape = [batch_size, n_featmaps, out_height, out_width];

    // compute actual kernel size, i.e. kernel size considering the dilation
    let act_kh = (dh * (kh - 1) + 1) as i64;
    let act_kw = (dw * (kw - 1) + 1) as i64;

    // result tensor
    let mut y: Array4<f32> = Array4::<f32>::from_elem(out_shape, 0.0);
    let bias = b.unwrap_or(Array1::from_shape_fn([n_featmaps], |_| 0.0));

    for bidx in 0..batch_size {
        for m in 0..n_featmaps {
            // get the group index of the feature map and compute input channel group bounds
            let g = m / output_group_size;
            let igs = g * input_group_size;
            let ige = igs + input_group_size;

            // declaration of tensor bounds considering padding
            let tens_sh: i64 = 0i64 - (phs as i64);
            let tens_sw: i64 = 0i64 - (pws as i64);
            let tens_eh: i64 = (height + phe) as i64 - act_kh + 1; // subtracting kernel size to consider valid windows only
            let tens_ew: i64 = (width + pwe) as i64 - act_kw + 1;

            // iterate over the input tensor with the specified stride
            for ext_i in (tens_sh..tens_eh).step_by(strh) {
                for ext_j in (tens_sw..tens_ew).step_by(strw) {
                    // declaration of kernel window bounds
                    let win_sh = ext_i;
                    let win_sw = ext_j;
                    let win_eh = ext_i + act_kh; // actual kernel size takes into account the dilation
                    let win_ew = ext_j + act_kw;

                    // ki and kj used to access the kernel
                    let mut ki = 0;
                    let mut accumulator: f32 = bias[[m]];
                    // iterate over the window defined by the kernel with the specified dilation
                    for i in (win_sh..win_eh).step_by(dh) {
                        if i >= 0 && i < height as i64 {
                            let mut kj = 0;
                            for j in (win_sw..win_ew).step_by(dw) {
                                if j >= 0 && j < width as i64 {
                                    // iterate also along all channels and increment accumulator
                                    for c in igs..ige {
                                        accumulator += x[[bidx, c, i as usize, j as usize]]
                                            * w[[m, c % input_group_size, ki, kj]];
                                    }
                                }
                                kj += 1;
                            }
                        }
                        ki += 1;
                    }
                    // compute output tensor indexes and update the corresponding value
                    let out_i = (ext_i + phs as i64) as usize / strh;
                    let out_j = (ext_j + pws as i64) as usize / strw;
                    y[[bidx, m, out_i, out_j]] = accumulator;
                }
            }
        }
    }
    y
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

pub fn max_pool(x: Array4<f32>, attrs: MaxPoolAttributes) -> Array4<f32> {
    let MaxPoolAttributes {
        kernel_shape: [kh, kw],
        pads: [phs, pws, phe, pwe],
        strides: [strh, strw],
    } = attrs;
    let [batch_size, in_chans, height, width] = *x.shape() else {todo!("Error in max_pool")};
    let out_height = 1 + ((height + phs + phe) - kh) / strh;
    let out_width = 1 + ((width + pws + pwe) - kw) / strw;
    let out_shape = [batch_size, in_chans, out_height, out_width];
    
    // result tensor
    let mut y: Array4<f32> = Array4::<f32>::from_elem(out_shape, 0.0);
    for bidx in 0..batch_size {
        for c in 0..in_chans {
            // declaration of tensor bounds considering padding
            let tens_sh: i64 = 0i64 - (phs as i64);
            let tens_sw: i64 = 0i64 - (pws as i64);
            let tens_eh: i64 = ((height + phe) - kh + 1) as i64; // subtracting kernel size to consider valid windows only
            let tens_ew: i64 = ((width + pwe) - kw + 1) as i64;

            // iterate over the input tensor with the specified stride
            for ext_i in (tens_sh..tens_eh).step_by(strh) {
                for ext_j in (tens_sw..tens_ew).step_by(strw) {
                    // declaration of kernel window bounds
                    let win_sh = ext_i;
                    let win_sw = ext_j;
                    let win_eh = ext_i + kh as i64; // actual kernel size takes into account the dilation
                    let win_ew = ext_j + kw as i64;

                    let mut res: f32 = f32::MIN;
                    // iterate over the window defined by the kernel
                    for i in win_sh..win_eh {
                        if i >= 0 && i < height as i64 {
                            for j in win_sw..win_ew {
                                if j >= 0 && j < width as i64 {
                                    res = res.max(x[[bidx, c, i as usize, j as usize]])
                                }
                            }
                        }
                    }
                    // compute output tensor indexes and update the corresponding value
                    let out_i = (ext_i + phs as i64) as usize / strh;
                    let out_j = (ext_j + pws as i64) as usize / strw;
                    y[[bidx, c, out_i, out_j]] = res;
                }
            }
        }
    }
    y
}