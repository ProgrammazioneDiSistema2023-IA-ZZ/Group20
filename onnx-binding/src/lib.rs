use onnx_runtime::service::{Config, Prediction, ServiceBuilder};
use pyo3::{exceptions::PyValueError, prelude::*};
use std::{fmt::Display, path::PathBuf};

#[pyclass(name = "Service")]
struct PyService {
    service: onnx_runtime::service::Service,
}
#[pyclass(name = "InferenceOutput")]
struct PyInferenceOutput {
    inference_output: onnx_runtime::service::InferenceOutput,
}

#[pyclass(name = "Prediction")]
pub struct PyPrediction {
    pub class: String,
    pub probability: f32,
}

impl From<Prediction> for PyPrediction {
    fn from(value: Prediction) -> Self {
        Self {
            class: value.class,
            probability: value.probability,
        }
    }
}

impl Display for PyPrediction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Class: {}, Probability: {:.2}",
            self.class, self.probability
        )
    }
}

#[pymethods]
impl PyPrediction {
    fn __repr__(&self) -> String {
        format!("({}, {:.2}%)", self.class, self.probability * 100f32)
    }
    pub fn __str__(&self) -> String {
        self.__repr__()
    }
}

#[pymethods]
impl PyService {
    pub fn prepare_and_run(
        &self,
        inputs: Vec<PathBuf>,
        input_parameters: Option<Vec<(String, usize)>>,
    ) -> PyResult<PyInferenceOutput> {
        let n_inputs = inputs.len();
        match self.service.prepare_and_run(
            inputs,
            input_parameters.unwrap_or(vec![(String::from("N"), n_inputs)]),
        ) {
            Err(err) => Err(PyErr::new::<PyValueError, _>(err.to_string())),
            Ok(inference_output) => Ok(PyInferenceOutput { inference_output }),
        }
    }
}

#[pymethods]
impl PyInferenceOutput {
    pub fn get_top_k_predictions(&self, k: usize) -> Vec<Vec<PyPrediction>> {
        // for each row in the tensor, get the top k predictions
        self.inference_output
            .get_top_k_predictions(k)
            .into_iter()
            .map(|v| v.into_iter().map(|x| x.into()).collect::<Vec<_>>())
            .collect::<Vec<_>>()
    }

    pub fn get_top_k_class_names(&self, k: usize) -> Vec<Vec<String>> {
        self.inference_output.get_top_k_class_names(k)
    }
}

/// Formats the sum of two numbers as string.
// #[pyfunction]
// fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
//     Ok((a + b).to_string())
// }
#[pyfunction]
fn build_service(model_path: PathBuf, num_threads: usize) -> PyResult<PyService> {
    match ServiceBuilder::new(model_path)
        .config(Config { num_threads })
        .build()
    {
        Err(err) => Err(PyErr::new::<PyValueError, _>(err.to_string())),
        Ok(service) => Ok(PyService { service }),
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn onnx_binding(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyService>()?;
    m.add_class::<PyInferenceOutput>()?;
    m.add_class::<PyPrediction>()?;
    m.add_function(wrap_pyfunction!(build_service, m)?)?;
    Ok(())
}
