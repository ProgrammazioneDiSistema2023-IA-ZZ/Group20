mod naive;
mod parallel;

pub use naive::*;
pub use parallel::*;

use ndarray::{Array1, ArrayD};
use rayon::ThreadPool;

use crate::{
    operators::{
        BatchNormAttributes, ClipAttributes, ConcatAttributes, ConvAttributes, GatherAttributes,
        GemmAttributes, MaxPoolAttributes, OperationError, UnsqueezeAttributes,
    },
    tensor::TypeToTensorDataType,
};

pub type DefaultProvider = ParNaiveProvider;

/// A trait that has to be implemented by all the execution providers.
/// It contains the methods that are used to execute the ONNX operators.
/// Currently only a subset of the ONNX operators is supported,
/// and only a naive execution provider exists.
///
/// This interface can be limiting for executing providers that may don't want to use ndarray.
/// In the future, it could be extended to support other data structures.
pub trait Provider {
    /// Returns the name of the execution provider.
    fn name(&self) -> &str;

    /// Returns the targeted ONNX ir_version of the execution provider.
    /// This is used to check if the execution provider is compatible with the model.
    /// For example, the naive execution provider only supports ir_version 7.
    fn version(&self) -> u64;

    fn conv(
        thread_pool: &ThreadPool,
        x: ArrayD<f32>,
        weights: ArrayD<f32>,
        bias: Option<Array1<f32>>,
        attrs: ConvAttributes,
    ) -> Result<ArrayD<f32>, OperationError>;

    fn gemm(
        thread_pool: &ThreadPool,
        a: ArrayD<f32>,
        b: ArrayD<f32>,
        c: ArrayD<f32>,
        attrs: GemmAttributes,
    ) -> Result<ArrayD<f32>, OperationError>;

    fn batch_norm(
        thread_pool: &ThreadPool,
        x: ArrayD<f32>,
        scale: ArrayD<f32>,
        b: ArrayD<f32>,
        mean: ArrayD<f32>,
        var: ArrayD<f32>,
        attrs: BatchNormAttributes,
    ) -> Result<ArrayD<f32>, OperationError>;

    fn gather(
        thread_pool: &ThreadPool,
        x: ArrayD<usize>,
        index: usize,
        attrs: GatherAttributes,
    ) -> Result<ArrayD<usize>, OperationError>;

    fn unsqueeze(
        thread_pool: &ThreadPool,
        x: ArrayD<usize>,
        attrs: UnsqueezeAttributes,
    ) -> Result<ArrayD<usize>, OperationError>;

    fn relu(thread_pool: &ThreadPool, x: ArrayD<f32>) -> ArrayD<f32>;

    fn clip(thread_pool: &ThreadPool, x: ArrayD<f32>, attrs: ClipAttributes) -> ArrayD<f32>;

    fn add(
        thread_pool: &ThreadPool,
        x: ArrayD<f32>,
        y: ArrayD<f32>,
    ) -> Result<ArrayD<f32>, OperationError>;

    fn shape(thread_pool: &ThreadPool, x: ArrayD<f32>) -> ArrayD<i64>;

    fn global_average_pool(
        thread_pool: &ThreadPool,
        x: ArrayD<f32>,
    ) -> Result<ArrayD<f32>, OperationError>;

    fn reshape(
        thread_pool: &ThreadPool,
        x: ArrayD<f32>,
        shape: ArrayD<i64>,
    ) -> Result<ArrayD<f32>, OperationError>;

    fn max_pool(
        thread_pool: &ThreadPool,
        x: ArrayD<f32>,
        attrs: MaxPoolAttributes,
    ) -> Result<ArrayD<f32>, OperationError>;

    fn concat<T>(
        thread_pool: &ThreadPool,
        x: Vec<ArrayD<T>>,
        attrs: ConcatAttributes,
    ) -> Result<ArrayD<T>, OperationError>
    where
        T: TypeToTensorDataType + Copy;
}
