use image::GenericImageView;

pub fn preprocessing(image: image::DynamicImage) -> ndarray::Array4<f32> {
    // resize image to 256x256
    let image = image.resize_exact(256, 256, image::imageops::FilterType::Triangle);
    // crop image to 224x224
    let cropped_image_view = image::imageops::crop_imm(&image, 16, 16, 224, 224);
    // convert image to an Array3<f32>
    let tensor = ndarray::Array3::from_shape_fn((224, 224, 3), |(y, x, c)| {
        cropped_image_view.get_pixel(x as u32, y as u32)[c] as f32
    });

    //transpose the image from [224, 224, 3] to [3, 224, 224]
    let tensor = tensor.permuted_axes([2, 0, 1]);

    //normalize the image using the mean and std of the ImageNet dataset
    let mean = ndarray::arr1(&[0.485, 0.456, 0.406]);
    let std = ndarray::arr1(&[0.229, 0.224, 0.225]);

    // subtract the mean and divide by the standard deviation individually for each color channel without broadcasting
    let tensor = tensor.mapv(|x| x / 255.0)
        - &mean
            .insert_axis(ndarray::Axis(1))
            .insert_axis(ndarray::Axis(1));
    let tensor = tensor
        / &std
            .insert_axis(ndarray::Axis(1))
            .insert_axis(ndarray::Axis(1));

    //add a batch dimension
    tensor.insert_axis(ndarray::Axis(0))
}

// Post processing for the ImageNet dataset using mxnet gluon
pub fn postprocessing(tensor: ndarray::Array2<f32>) -> ndarray::Array1<f32> {
    // softmax on the last axis of the tensor
    let tensor = tensor.mapv(|x| x.exp());
    let tensor = tensor.clone()
        / tensor
            .sum_axis(ndarray::Axis(1))
            .insert_axis(ndarray::Axis(1));

    tensor
        .remove_axis(ndarray::Axis(0))
        .into_shape(1000)
        .unwrap()
}

pub fn postprocessing_top_k(tensor: ndarray::Array1<f32>, k: usize) -> Vec<(String, f32)> {
    // load the labels
    let labels = std::fs::read_to_string("tests/labels.txt").unwrap();
    let labels: Vec<_> = labels.lines().collect();

    // get the top k predictions
    let mut top_k_classes = tensor
        .into_iter()
        .enumerate()
        .map(|(i, x)| (i, x))
        .collect::<Vec<_>>();
    top_k_classes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    top_k_classes.truncate(k);

    top_k_classes
        .into_iter()
        .map(|(i, x)| (String::from(labels[i]), x))
        .collect()
}
