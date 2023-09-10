In order to do the binding, we used the Pyo3 crate.      

pip install -r onnx-runtime-pyapp/requirements.txt

pip install maturin
cd onnx-binding && maturin build --release
cd target/wheels && pip uninstall onnx_binding-0.1.0.whl && pip install onnx_binding-0.1.0.whl
