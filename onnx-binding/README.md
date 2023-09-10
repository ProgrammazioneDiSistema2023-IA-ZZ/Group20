# Python Binding
We used the Pyo3 crate.
In order to use it, you can follow its documentation [here](https://pyo3.rs/main/getting_started).

A simplified way to use it is to install directly **maturin** with the following commands:



pip install -r onnx-runtime-pyapp/requirements.txt

pip install maturin
cd onnx-binding && maturin build --release
cd target/wheels && pip uninstall onnx_binding-0.1.0.whl && pip install onnx_binding-0.1.0.whl
