load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "platforms",
    sha256 = "218efe8ee736d26a3572663b374a253c012b716d8af0c07e842e82f238a0a7ee",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/platforms/releases/download/0.0.10/platforms-0.0.10.tar.gz",
        "https://github.com/bazelbuild/platforms/releases/download/0.0.10/platforms-0.0.10.tar.gz",
    ],
)

load("@platforms//host:extension.bzl", "host_platform_repo")

http_archive(
    name = "bazel_skylib",
    sha256 = "bc283cdfcd526a52c3201279cda4bc298652efa898b10b4db0837dc51652756f",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.7.1/bazel-skylib-1.7.1.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.7.1/bazel-skylib-1.7.1.tar.gz",
    ],
)

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")

bazel_skylib_workspace()

http_archive(
    name = "bazel_lib",
    sha256 = "bc283cdfcd526a52c3201279cda4bc298652efa898b10b4db0837dc51652756f",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.7.1/bazel-skylib-1.7.1.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.7.1/bazel-skylib-1.7.1.tar.gz",
    ],
)

http_archive(
    name = "aspect_bazel_lib",
    sha256 = "688354ee6beeba7194243d73eb0992b9a12e8edeeeec5b6544f4b531a3112237",
    strip_prefix = "bazel-lib-2.8.1",
    url = "https://github.com/aspect-build/bazel-lib/releases/download/v2.8.1/bazel-lib-v2.8.1.tar.gz",
)

load("@aspect_bazel_lib//lib:repositories.bzl", "aspect_bazel_lib_dependencies", "aspect_bazel_lib_register_toolchains")

aspect_bazel_lib_dependencies()

aspect_bazel_lib_register_toolchains()

# LLVM_COMMIT = "cdd31610fdde4848a6260864c7bd73115be6ea74"
# LLVM_SHA256 = ""
# http_archive(
#     name = "llvm-raw",
#     build_file_content = "# empty",
#     sha256 = LLVM_SHA256,
#     strip_prefix = "llvm-project-" + LLVM_COMMIT,
#     urls = ["https://github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT)],
# )
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

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

maybe(
    host_platform_repo,
    name = "host_platform",
)

http_archive(
    name = "bazel_features",
    sha256 = "07271d0f6b12633777b69020c4cb1eb67b1939c0cf84bb3944dc85cc250c0c01",
    strip_prefix = "bazel_features-1.38.0",
    url = "https://github.com/bazel-contrib/bazel_features/releases/download/v1.38.0/bazel_features-v1.38.0.tar.gz",
)

load("@bazel_features//:deps.bzl", "bazel_features_deps")

bazel_features_deps()

http_archive(
    name = "rules_multitool",
    strip_prefix = "rules_multitool-1.11.1",
    url = "https://github.com/bazel-contrib/rules_multitool/releases/download/v1.11.1/rules_multitool-1.11.1.tar.gz",
)

load("@rules_multitool//multitool:multitool.bzl", "multitool")

multitool(
    name = "multitool",
    lockfile = "//:multitool.lock.json",
)

load("@multitool//:tools.bzl", "register_tools")

register_tools()

http_archive(
    name = "rules_uv",
    strip_prefix = "rules_uv-0.89.2",
    url = "https://github.com/bazel-contrib/rules_uv/releases/download/v0.89.2/rules_uv-0.89.2.tar.gz",
)

load("//third_party/jax:workspace.bzl", jax_workspace = "repo")
load("//third_party/ml_toolchain:workspace.bzl", ml_toolchain_workspace = "repo")

jax_workspace()

load("//third_party/xla:workspace.bzl", xla_workspace = "repo")

xla_workspace()

load("@xla//:workspace4.bzl", "xla_workspace4")

xla_workspace4()

load("@xla//:workspace3.bzl", "xla_workspace3")

xla_workspace3()

ml_toolchain_workspace()

load("@rules_ml_toolchain//cc/deps:cc_toolchain_deps.bzl", "cc_toolchain_deps")

cc_toolchain_deps()

load("@xla//third_party/py:python_init_rules.bzl", "python_init_rules")

python_init_rules()

load("@xla//third_party/py:python_init_repositories.bzl", "python_init_repositories")

python_init_repositories(
    local_wheel_inclusion_list = [
        "enzyme_ad*",
    ],
    requirements = {
        "3.11": "//builddeps:requirements_lock_3_11.txt",
        "3.12": "//builddeps:requirements_lock_3_12.txt",
        "3.13": "//builddeps:requirements_lock_3_13.txt",
    },
)

load("@xla//third_party/py:python_init_toolchains.bzl", "python_init_toolchains")

python_init_toolchains()

load("@xla//third_party/py:python_init_pip.bzl", "python_init_pip")

python_init_pip()

load("@pypi//:requirements.bzl", "install_deps")

install_deps()

load("@xla//:workspace2.bzl", "xla_workspace2")
load("@xla//third_party/llvm:workspace.bzl", llvm = "repo")
load("//:workspace.bzl", "LLVM_TARGETS")

xla_workspace2()

http_archive(
        name = "llvm-raw",
        sha256 = "49049408bdf9ae162b95b3eacfb8f9af226195e66636d0c9eff9a61d99be54c5",
        patch_cmds = ["find . -type f -name config.h -exec sed -i.bak0 's/HAVE_PTHREAD_SETNAME_NP/FAKE_HAVE_PTHREAD_SETNAME_NP/g' {} +","find . -type f -name config.h -exec sed -i.bak0 's/HAVE_PTHREAD_GETNAME_NP/FAKE_HAVE_PTHREAD_GETNAME_NP/g' {} +","find . -type f -name config.h -exec sed -i.bak0 's/ENABLE_CRASH_OVERRIDES 1/ENABLE_CRASH_OVERRIDES 0/g' {} +","find . -type f -name config.bzl -exec sed -i.bak0 's/HAVE_PTHREAD_SETNAME_NP=1/FAKE_HAVE_PTHREAD_SETNAME_NP=0/g' {} +","find . -type f -name config.bzl -exec sed -i.bak0 's/HAVE_PTHREAD_GETNAME_NP=1/FAKE_HAVE_PTHREAD_GETNAME_NP=0/g' {} +","find . -type f -name config.bzl -exec sed -i.bak0 's/HAVE_MALLINFO=1/DONT_HAVE_ANY_MALLINFO=0/g' {} +","find . -type f -name config.bzl -exec sed -i.bak0 's/LLVM_ENABLE_THREADS=1/LLVM_ENABLE_THREADS=0/g' {} +","find . -type f -name config.bzl -exec sed -i.bak0 's/HAVE_LINK_H=1/HAVE_LINK_H=0/g' {} +","find . -type f -name config.bzl -exec sed -i.bak0 's/HAVE_BACKTRACE=1/NO_HAVE_BACKTRACE=0/g' {} +"],
        strip_prefix = "llvm-project-a7c97252ed18839e87796dffd3926086ab62e8fd",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/llvm/llvm-project/archive/a7c97252ed18839e87796dffd3926086ab62e8fd.tar.gz",
            "https://github.com/llvm/llvm-project/archive/a7c97252ed18839e87796dffd3926086ab62e8fd.tar.gz",
        ],
        build_file_content = "# empty",
        patch_args = ["-p1"],
        patches = [
            "@xla//third_party/llvm:generated.patch",
            "@xla//third_party/llvm:build.patch",
            "@xla//third_party/llvm:mathextras.patch",
            "@xla//third_party/llvm:toolchains.patch",
            "@xla//third_party/llvm:zstd.patch",
            "@xla//third_party/llvm:lit_test.patch",
            "@xla//third_party/llvm:run_lit.patch",
            "//:patches/llvm_inliner.patch",
        ],
    )

load("@llvm-raw//utils/bazel:configure.bzl", "llvm_configure")

llvm_configure(
    name = "llvm-project",
    targets = LLVM_TARGETS,
)

load("@xla//:workspace1.bzl", "xla_workspace1")

xla_workspace1()

load("@xla//:workspace0.bzl", "xla_workspace0")

xla_workspace0()

load("@jax//third_party/flatbuffers:workspace.bzl", flatbuffers = "repo")

flatbuffers()

load("@jax//third_party/external_deps:workspace.bzl", "external_deps_repository")

external_deps_repository(name = "rocm_external_test_deps")

load("@jax//:test_shard_count.bzl", "test_shard_count_repository")

test_shard_count_repository(
    name = "test_shard_count",
)

load("@jax//jaxlib:jax_python_wheel.bzl", "jax_python_wheel_repository")

jax_python_wheel_repository(
    name = "jax_wheel",
    version_key = "_version",
    version_source = "@jax//jax:version.py",
)

load("@jax_wheel//:wheel.bzl", "WHEEL_VERSION")
load("@python_version_repo//:py_version.bzl", "HERMETIC_PYTHON_VERSION")
load("@jax//third_party/rocm_wheels:workspace.bzl", "rocm_wheels_repository")

# Pre-built ROCm wheels from a GitHub release (ROCm/rocm-jax).
rocm_wheels_repository(
    name = "rocm_wheels",
    jaxlib_version = WHEEL_VERSION,
    python_version = HERMETIC_PYTHON_VERSION,
    # rocm_version = "7.2.0",  # Optional: pick a specific ROCm version.
)

# Used for --//jax:build_jaxlib=false (pre-built wheels from GitHub).
external_deps_repository(
    name = "rocm_prebuilt_test_deps",
    deps = [
        "@rocm_wheels//:rocm_pjrt_py_import",
        "@rocm_wheels//:rocm_plugin_py_import",
    ],
)

load("@jax//jaxlib:jax_python_wheel.bzl", "jax_python_wheel_repository")

jax_python_wheel_repository(
    name = "jax_wheel",
    version_key = "_version",
    version_source = "@jax//jax:version.py",
)

load(
    "@xla//third_party/py:python_wheel.bzl",
    "nvidia_wheel_versions_repository",
    "python_wheel_version_suffix_repository",
)

nvidia_wheel_versions_repository(
    name = "nvidia_wheel_versions",
    versions_source = "@jax//build:nvidia-requirements.txt",
)

python_wheel_version_suffix_repository(
    name = "jax_wheel_version_suffix",
)

load(
    "@rules_ml_toolchain//gpu/cuda:cuda_json_init_repository.bzl",
    "cuda_json_init_repository",
)

cuda_json_init_repository()

load(
    "@cuda_redist_json//:distributions.bzl",
    "CUDA_REDISTRIBUTIONS",
    "CUDNN_REDISTRIBUTIONS",
)
load(
    "@rules_ml_toolchain//gpu/cuda:cuda_redist_init_repositories.bzl",
    "cuda_redist_init_repositories",
    "cudnn_redist_init_repository",
)
load(
    "@rules_ml_toolchain//gpu/cuda:cuda_redist_versions.bzl",
    "REDIST_VERSIONS_TO_BUILD_TEMPLATES",
)
load("@xla//third_party/cccl:workspace.bzl", "CCCL_3_2_0_DIST_DICT", "CCCL_GITHUB_VERSIONS_TO_BUILD_TEMPLATES")

cuda_redist_init_repositories(
    cuda_redistributions = CUDA_REDISTRIBUTIONS | CCCL_3_2_0_DIST_DICT,
    redist_versions_to_build_templates = REDIST_VERSIONS_TO_BUILD_TEMPLATES | CCCL_GITHUB_VERSIONS_TO_BUILD_TEMPLATES,
)

cudnn_redist_init_repository(
    cudnn_redistributions = CUDNN_REDISTRIBUTIONS,
)

load(
    "@rules_ml_toolchain//gpu/cuda:cuda_configure.bzl",
    "cuda_configure",
)

cuda_configure(name = "local_config_cuda")

load(
    "@rules_ml_toolchain//gpu/nccl:nccl_redist_init_repository.bzl",
    "nccl_redist_init_repository",
)

nccl_redist_init_repository()

load(
    "@rules_ml_toolchain//gpu/nccl:nccl_configure.bzl",
    "nccl_configure",
)

nccl_configure(name = "local_config_nccl")

load(
    "@rules_ml_toolchain//gpu/nvshmem:nvshmem_json_init_repository.bzl",
    "nvshmem_json_init_repository",
)

nvshmem_json_init_repository()

load(
    "@nvshmem_redist_json//:distributions.bzl",
    "NVSHMEM_REDISTRIBUTIONS",
)
load(
    "@rules_ml_toolchain//gpu/nvshmem:nvshmem_redist_init_repository.bzl",
    "nvshmem_redist_init_repository",
)

nvshmem_redist_init_repository(
    nvshmem_redistributions = NVSHMEM_REDISTRIBUTIONS,
)

load("//third_party/cuda_tile:workspace.bzl", cuda_tile_workspace = "repo")
load("//third_party/enzyme:workspace.bzl", enzyme_workspace = "repo")

# add support for generating compile_commands
load("//third_party/hedron_compile_commands:workspace.bzl", hedron_compile_commands_workspace = "repo")

enzyme_workspace()

cuda_tile_workspace()

hedron_compile_commands_workspace()

load("@hedron_compile_commands//:workspace_setup.bzl", "hedron_compile_commands_setup")
load("@hedron_compile_commands//:workspace_setup_transitive.bzl", "hedron_compile_commands_setup_transitive")
load("@hedron_compile_commands//:workspace_setup_transitive_transitive.bzl", "hedron_compile_commands_setup_transitive_transitive")
load("@hedron_compile_commands//:workspace_setup_transitive_transitive_transitive.bzl", "hedron_compile_commands_setup_transitive_transitive_transitive")

hedron_compile_commands_setup()

hedron_compile_commands_setup_transitive()

hedron_compile_commands_setup_transitive_transitive()

hedron_compile_commands_setup_transitive_transitive_transitive()
