# Use the official Ubuntu 24.04 image as the base image
FROM ubuntu:24.04

# Set environment variables to avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update the package list and install necessary packages
RUN apt-get update && apt-get install -y \
    clang \
    cython3 \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    python3-matplotlib \
    python3-numpy \
    python3-importlib-metadata \ 
    build-essential \
    git nano vim cmake sudo \
    libboost-all-dev \
    && apt-get clean

# Create a working directory
WORKDIR /workspace

# Clone required repositories
#RUN git clone https://github.com/pybind/pybind11.git /workspace/pybind11
#RUN git clone -b $BRANCH https://github.com/pybind/pybind11.git /workspace/pybind11_sh
#COPY . /workspace/pybind11-branch
#RUN mkdir /workspace/branch
#RUN git clone https://github.com/wjakob/nanobind.git /workspace/nanobind

COPY ./microbench* ./run-bench.sh /workspace/

# Set the paths in the notebook
#RUN sed -i 's/python310/python312/g' /workspace/microbenchmark.ipynb
#RUN sed -i 's|/home/wjakob/pybind11/include|/workspace/pybind11/include|' /workspace/microbenchmark.ipynb
#RUN sed -i 's|/home/wjakob/pybind11_sh/include|/workspace/pybind11-branch/include|' /workspace/microbenchmark.ipynb

# Set the working directory to nanobind
WORKDIR /workspace/

# Run the Jupyter notebook to perform the benchmark
#RUN jupyter nbconvert --to notebook --execute docs/microbenchmark.ipynb

# Default command to run when the container starts
CMD ["bash"]

