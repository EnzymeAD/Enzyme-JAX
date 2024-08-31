BAZEL_BUILD_FLAGS=()
BAZEL_BUILD_FLAGS+=(--define=no_aws_support=true)
BAZEL_BUILD_FLAGS+=(--define=no_gcp_support=true)
BAZEL_BUILD_FLAGS+=(--define=no_hdfs_support=true)
BAZEL_BUILD_FLAGS+=(--define=no_kafka_support=true)
BAZEL_BUILD_FLAGS+=(--define=no_ignite_support=true)
BAZEL_BUILD_FLAGS+=(--define=grpc_no_ares=true)

BAZEL_BUILD_FLAGS+=(--define=llvm_enable_zlib=false)
BAZEL_BUILD_FLAGS+=(--verbose_failures)

export TMPDIR=$HOME/.eqsat-tmp
export TMP=$TMPDIR
export TEMP=$TMPDIR
BAZEL_BUILD_FLAGS+=(--action_env=TMP=$TMPDIR --action_env=TEMP=$TMPDIR --action_env=TMPDIR=$TMPDIR --sandbox_tmpfs_path=$TMPDIR)

export CUDA_HOME=/usr/local/cuda
export PATH=$PATH:$CUDA_HOME/bin
export CUDACXX=$CUDA_HOME/bin/nvcc
BAZEL_BUILD_FLAGS+=(--repo_env TF_NEED_CUDA=1)
BAZEL_BUILD_FLAGS+=(--repo_env TF_CUDA_VERSION=12.3)
BAZEL_BUILD_FLAGS+=(--repo_env TF_CUDA_PATHS="$CUDA_HOME,/usr/lib/x86_64-linux-gnu,/usr/include")
BAZEL_BUILD_FLAGS+=(--repo_env TF_NCCL_USE_STUB=1)
BAZEL_BUILD_FLAGS+=(--action_env TF_CUDA_COMPUTE_CAPABILITIES="sm_50,sm_60,sm_70,sm_80,compute_90")
BAZEL_BUILD_FLAGS+=(--crosstool_top=@local_config_cuda//crosstool:toolchain)
BAZEL_BUILD_FLAGS+=(--@local_config_cuda//:enable_cuda)
BAZEL_BUILD_FLAGS+=(--@xla//xla/python:enable_gpu=true)
BAZEL_BUILD_FLAGS+=(--@xla//xla/python:jax_cuda_pip_rpaths=true)
BAZEL_BUILD_FLAGS+=(--define=xla_python_enable_gpu=true)
bazel build ${BAZEL_BUILD_FLAGS[@]} :enzyme_ad
pip install bazel-bin/enzyme_ad-0.0.6-py3-none-manylinux2014_x86_64.whl --force-reinstall --no-deps

