# Python Binding
We used the Pyo3 crate to build a python wheel to be able to easily install our onnx-runtime library in python.
You can refer to [this documentation](https://pyo3.rs/main/getting_started) to know more about Pyo3.

To build our library we decided to use **maturin**, which is a tool in the Python ecosystem related to building and distributing Rust-based extensions for Python. It can be installed with the command
```
pip install maturin
```
Then, we initalized a new project with maturin
```
maturin new onnx-binding
```
Here, it was necessary to properly setup a **Cargo.toml** and a **pyproject.toml** file, in order to properly build the library. Then, we exported some data types and some functions as explained in the Pyo3 documentation. Finally, we built the library.
```
maturin build --release
```
The `--release` is necessary in order to avoid debugging informations in the code which make some operations (e.g. convolution) really slow.

The library can be installed with the simple
```
pip install target/wheels/wheel_file.whl
```
and it's now ready to be used.

?? We already provide a built library with this source code.