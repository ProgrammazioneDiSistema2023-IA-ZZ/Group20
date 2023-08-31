use image::GenericImageView;
use ndarray::{Array4, ShapeError};

type Prediction = (String, f32);

const IMAGENET_LABELS: [&str; 1000] = include!("labels/imagenet_labels.in");

fn single_preprocessing(image: &image::DynamicImage) -> ndarray::Array3<f32> {
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

    tensor
        / &std
            .insert_axis(ndarray::Axis(1))
            .insert_axis(ndarray::Axis(1))
}

/// Preprocessing for the ImageNet dataset.
/// It should be used only on a single image.
/// It will always add a batch dimension equal to 1 to the result.
pub fn preprocessing(image: &image::DynamicImage) -> ndarray::Array4<f32> {
    // call single_preprocessing on the image and add a batch dimension
    single_preprocessing(image).insert_axis(ndarray::Axis(0))
}

/// Preprocessing for the ImageNet dataset.
/// It should be used on a batch of images.
/// It will add a batch dimension equal to the number of images to the result.
pub fn batch_preprocessing(
    images: &[image::DynamicImage],
) -> Result<ndarray::Array4<f32>, ShapeError> {
    // call preprocessing on each image and create an array4 from the results
    Array4::from_shape_vec(
        (images.len(), 3, 224, 224),
        images.iter().flat_map(single_preprocessing).collect(),
    )
}

pub fn postprocessing(tensor: ndarray::Array2<f32>) -> ndarray::Array2<f32> {
    // softmax on the last axis of the tensor
    let tensor = tensor.mapv(|x| x.exp());

    tensor.clone()
        / tensor
            .sum_axis(ndarray::Axis(1))
            .insert_axis(ndarray::Axis(1))
}

/// Get the top k predictions for each batch element
pub fn postprocessing_top_k(tensor: ndarray::Array2<f32>, k: usize) -> Vec<Vec<Prediction>> {
    // for each row in the tensor, get the top k predictions
    tensor
        .outer_iter()
        .map(|row| postprocessing_top_k_single(row.to_owned(), k))
        .collect()
}

fn postprocessing_top_k_single(tensor: ndarray::Array1<f32>, k: usize) -> Vec<Prediction> {
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
        .map(|(i, x)| (String::from(IMAGENET_LABELS[i]), x))
        .collect()
}
