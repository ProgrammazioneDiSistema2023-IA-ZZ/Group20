FROM ubuntu:22.04
COPY . /app
WORKDIR /app
RUN apt-get update -y && apt-get install -y python3-pip python3-dev
RUN apt-get install -y curl

# Install Cargo and Rust
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
# Add .cargo/bin to PATH
ENV PATH="/root/.cargo/bin:${PATH}"

# Install python dependencies
RUN pip3 install -r onnx-runtime-pyapp/requirements.txt

RUN pip3 install maturin
RUN cd onnx-binding && maturin build --release
RUN cd target/wheels && pip uninstall onnx_binding-0.1.0*.whl && pip install onnx_binding-0.1.0*.whl

CMD ["tail", "-f", "/dev/null"]
