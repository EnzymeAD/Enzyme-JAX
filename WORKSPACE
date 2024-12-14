load("//:workspace.bzl", "JAX_COMMIT", "JAX_SHA256", "ENZYME_COMMIT", "ENZYME_SHA256", "XLA_PATCHES")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# http_archive(
#     name = "rules_cc",
#     sha256 = "a281b09e003a119bda2cea98c590129b783cc241fdd1a1de3baa656117e6cc3f"
#     strip_prefix = "rules_cc-34f0e1f038cff9bf28d65d101c806b48f35bf92e",
#     urls = [
#         "https://github.com/bazelbuild/rules_cc/archive/34f0e1f038cff9bf28d65d101c806b48f35bf92e.tar.gz",
#     ],
# )
# 
# load("@rules_cc//cc:repositories.bzl", "rules_cc_dependencies")
# 
# rules_cc_dependencies()

LLVM_TARGETS = ["X86", "AArch64", "AMDGPU", "NVPTX"]

http_archive(
    name = "jax",
    sha256 = JAX_SHA256,
    strip_prefix = "jax-" + JAX_COMMIT,
    urls = ["https://github.com/google/jax/archive/{commit}.tar.gz".format(commit = JAX_COMMIT)],
    patch_args = ["-p1"],
    patches = ["//:patches/jax.patch"],
)

load("@jax//third_party/xla:workspace.bzl", "XLA_COMMIT", "XLA_SHA256")
http_archive(
    name = "xla",
    sha256 = XLA_SHA256,
    strip_prefix = "xla-" + XLA_COMMIT,
    urls = ["https://github.com/wsmoses/xla/archive/{commit}.tar.gz".format(commit = XLA_COMMIT)],
    patch_cmds = XLA_PATCHES,
)

load("@xla//third_party/py:python_init_rules.bzl", "python_init_rules")
python_init_rules()

load("@xla//third_party/py:python_init_repositories.bzl", "python_init_repositories")
python_init_repositories(
    requirements = {
        "3.10": "//builddeps:requirements_lock_3_10.txt",
        "3.11": "//builddeps:requirements_lock_3_11.txt",
        "3.12": "//builddeps:requirements_lock_3_12.txt",
    },
    local_wheel_inclusion_list = [
        "enzyme_ad*",
    ]
)

load("@xla//third_party/py:python_init_toolchains.bzl", "python_init_toolchains")
python_init_toolchains()

load("@xla//third_party/py:python_init_pip.bzl", "python_init_pip")
python_init_pip()

load("@pypi//:requirements.bzl", "install_deps")
install_deps()

http_archive(
    name = "enzyme",
    sha256 = ENZYME_SHA256,
    strip_prefix = "Enzyme-" + ENZYME_COMMIT + "/enzyme",
    urls = ["https://github.com/EnzymeAD/Enzyme/archive/{commit}.tar.gz".format(commit = ENZYME_COMMIT)],
)

load("@xla//third_party/llvm:workspace.bzl", llvm = "repo")
llvm("llvm-raw")
load("@llvm-raw//utils/bazel:configure.bzl", "llvm_configure")
llvm_configure(name = "llvm-project", targets = LLVM_TARGETS)

load("@xla//:workspace4.bzl", "xla_workspace4")
xla_workspace4()

load("@xla//:workspace3.bzl", "xla_workspace3")
xla_workspace3()

load("@xla//:workspace2.bzl", "xla_workspace2")
xla_workspace2()

load("@xla//:workspace1.bzl", "xla_workspace1")
xla_workspace1()

load("@xla//:workspace0.bzl", "xla_workspace0")
xla_workspace0()

load("@jax//third_party/flatbuffers:workspace.bzl", flatbuffers = "repo")
flatbuffers()

load(
    "@tsl//third_party/gpus/cuda/hermetic:cuda_json_init_repository.bzl",
    "cuda_json_init_repository",
)

cuda_json_init_repository()

load(
    "@cuda_redist_json//:distributions.bzl",
    "CUDA_REDISTRIBUTIONS",
    "CUDNN_REDISTRIBUTIONS",
)
load(
    "@tsl//third_party/gpus/cuda/hermetic:cuda_redist_init_repositories.bzl",
    "cuda_redist_init_repositories",
    "cudnn_redist_init_repository",
)

cuda_redist_init_repositories(
    cuda_redistributions = CUDA_REDISTRIBUTIONS,
)

cudnn_redist_init_repository(
    cudnn_redistributions = CUDNN_REDISTRIBUTIONS,
)

load(
    "@tsl//third_party/gpus/cuda/hermetic:cuda_configure.bzl",
    "cuda_configure",
)

cuda_configure(name = "local_config_cuda")

load(
    "@tsl//third_party/nccl/hermetic:nccl_redist_init_repository.bzl",
    "nccl_redist_init_repository",
)

nccl_redist_init_repository()

load(
    "@tsl//third_party/nccl/hermetic:nccl_configure.bzl",
    "nccl_configure",
)

nccl_configure(name = "local_config_nccl")
