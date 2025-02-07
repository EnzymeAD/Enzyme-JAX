# add support for generating compile_commands
load("//third_party/hedron_compile_commands:workspace.bzl", hedron_compile_commands_workspace = "repo")
hedron_compile_commands_workspace()

load("@hedron_compile_commands//:workspace_setup.bzl", "hedron_compile_commands_setup")
load("@hedron_compile_commands//:workspace_setup_transitive.bzl", "hedron_compile_commands_setup_transitive")
load("@hedron_compile_commands//:workspace_setup_transitive_transitive.bzl", "hedron_compile_commands_setup_transitive_transitive")
load("@hedron_compile_commands//:workspace_setup_transitive_transitive_transitive.bzl", "hedron_compile_commands_setup_transitive_transitive_transitive")

hedron_compile_commands_setup()
hedron_compile_commands_setup_transitive()
hedron_compile_commands_setup_transitive_transitive()
hedron_compile_commands_setup_transitive_transitive_transitive()
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# LLVM_COMMIT = "dfa13d320fa3e316c88971980c6793a6a4618b08"
# LLVM_SHA256 = ""
# http_archive(
#     name = "llvm-raw",
#     build_file_content = "# empty",
#     sha256 = LLVM_SHA256,
#     strip_prefix = "llvm-project-" + LLVM_COMMIT,
#     urls = ["https://github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT)],
# )
# 
# 
# load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
# maybe(
#     http_archive,
#     name = "llvm_zlib",
#     build_file = "@llvm-raw//utils/bazel/third_party_build:zlib-ng.BUILD",
#     sha256 = "e36bb346c00472a1f9ff2a0a4643e590a254be6379da7cddd9daeb9a7f296731",
#     strip_prefix = "zlib-ng-2.0.7",
#     urls = [
#         "https://github.com/zlib-ng/zlib-ng/archive/refs/tags/2.0.7.zip",
#     ],
# )
# 
# maybe(
#     http_archive,
#     name = "llvm_zstd",
#     build_file = "@llvm-raw//utils/bazel/third_party_build:zstd.BUILD",
#     sha256 = "7c42d56fac126929a6a85dbc73ff1db2411d04f104fae9bdea51305663a83fd0",
#     strip_prefix = "zstd-1.5.2",
#     urls = [
#         "https://github.com/facebook/zstd/releases/download/v1.5.2/zstd-1.5.2.tar.gz"
#     ],
# )

load("//third_party/jax:workspace.bzl", jax_workspace = "repo")
jax_workspace()

load("//third_party/xla:workspace.bzl", xla_workspace = "repo")
xla_workspace()

load("//third_party/enzyme:workspace.bzl", enzyme_workspace = "repo")
enzyme_workspace()

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

load("@xla//third_party/llvm:workspace.bzl", llvm = "repo")
load("//:workspace.bzl", "LLVM_TARGETS")
llvm("llvm-raw")
load("@llvm-raw//utils/bazel:configure.bzl", "llvm_configure")
llvm_configure(name = "llvm-project", targets = LLVM_TARGETS)

load("@jax//third_party/flatbuffers:workspace.bzl", flatbuffers = "repo")
flatbuffers()

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
