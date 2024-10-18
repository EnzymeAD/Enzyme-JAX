BAZEL_BUILD_FLAGS=()
BAZEL_BUILD_FLAGS+=(--define=no_aws_support=true)
BAZEL_BUILD_FLAGS+=(--define=no_gcp_support=true)
BAZEL_BUILD_FLAGS+=(--define=no_hdfs_support=true)
BAZEL_BUILD_FLAGS+=(--define=no_kafka_support=true)
BAZEL_BUILD_FLAGS+=(--define=no_ignite_support=true)
BAZEL_BUILD_FLAGS+=(--define=grpc_no_ares=true)

BAZEL_BUILD_FLAGS+=(--define=llvm_enable_zlib=false)

BAZEL_BUILD_FLAGS+=(--verbose_failures)
BAZEL_BUILD_FLAGS+=(--cxxopt=-std=c++17 --host_cxxopt=-std=c++17)
BAZEL_BUILD_FLAGS+=(--cxxopt=-DTCP_USER_TIMEOUT=0)
BAZEL_BUILD_FLAGS+=(--check_visibility=false)
BAZEL_BUILD_FLAGS+=(--experimental_cc_shared_library)

export TMPDIR=$HOME/.eqsat-tmp
export TMP=$TMPDIR
export TEMP=$TMPDIR
BAZEL_BUILD_FLAGS+=(--action_env=TMP=$TMPDIR --action_env=TEMP=$TMPDIR --action_env=TMPDIR=$TMPDIR --sandbox_tmpfs_path=$TMPDIR)

export CUDA_HOME=$HOME/miniconda3/
export PATH=$PATH:$CUDA_HOME/bin
export CUDACXX=$CUDA_HOME/bin/nvcc
BAZEL_BUILD_FLAGS+=(--config=cuda)
HERMETIC_PYTHON_VERSION=3.11 bazel build ${BAZEL_BUILD_FLAGS[@]} :wheel
pip install bazel-bin/enzyme_ad-0.0.8-py311-none-manylinux2014_x86_64.whl --no-deps --force-reinstall

