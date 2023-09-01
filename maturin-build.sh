#! /bin/bash
cd onnx-binding
maturin build --release
cd ../

cd target/wheels 
pip uninstall onnx_binding-0.1.0*.whl
pip install onnx_binding-0.1.0*.whl
cd ../../

python3 test.py
