FROM mcr.microsoft.com/devcontainers/base:ubuntu-22.04

# Install prerequisites
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    software-properties-common \
    lsb-release \
    build-essential \
    python3-pip \
    git

# Install Clang-18
RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 18 all && \
    rm llvm.sh

# Install Bazelisk (to provide 'bazel' command)
RUN wget https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 -O /usr/local/bin/bazel && \
    chmod +x /usr/local/bin/bazel

# Set Clang-18 as the default compiler for Bazel/Builds
ENV CC=/usr/bin/clang-18
ENV CXX=/usr/bin/clang++-18

# Ensure Python is available (required for JAX/Enzyme-JAX scripts)
RUN ln -s /usr/bin/python3 /usr/bin/python
