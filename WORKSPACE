load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "rules_cc",
    sha256 = "85723d827f080c5e927334f1fb18a294c0b3f94fee6d6b45945f5cdae6ea0fd4",
    strip_prefix = "rules_cc-c8c38f8c710cbbf834283e4777916b68261b359c",
    urls = [
        "https://github.com/bazelbuild/rules_cc/archive/c8c38f8c710cbbf834283e4777916b68261b359c.tar.gz",
    ],
)

load("@rules_cc//cc:repositories.bzl", "rules_cc_dependencies")

rules_cc_dependencies()

LLVM_COMMIT = "d3ef86708241a3bee902615c190dead1638c4e09"
LLVM_SHA256 = ""
LLVM_TARGETS = ["X86", "AArch64", "AMDGPU", "NVPTX"]

http_archive(
    name = "llvm-raw",
    build_file_content = "# empty",
    sha256 = LLVM_SHA256,
    strip_prefix = "llvm-project-" + LLVM_COMMIT,
    urls = ["https://github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT)],
    patch_args = ["-p1"],
    patches = ["//:patches/llvm_build.patch"]
)

load("@llvm-raw//utils/bazel:configure.bzl", "llvm_configure")
llvm_configure(name = "llvm-project", targets = LLVM_TARGETS)

XLA_COMMIT = "9f84844fa54020a6d99add1645a5d386603b6462"
XLA_COMMIT = "9e3cb9fa5fbb6ecc6e631d581cbb851b018d0728"
XLA_SHA256 = ""

http_archive(
    name = "xla",
    sha256 = XLA_SHA256,
    strip_prefix = "xla-" + XLA_COMMIT,
    urls = ["https://github.com/wsmoses/xla/archive/{commit}.tar.gz".format(commit = XLA_COMMIT)],
    patch_args = ["-p1"],
    patches = ["//:patches/xla.patch"],
)

PYRULES_COMMIT = "fe33a4582c37499f3caeb49a07a78fc7948a8949"
PYRULES_SHA256 = "cfa6957832ae0e0c7ee2ccf455a888a291e8419ed8faf45f4420dd7414d5dd96"

http_archive(
    name = "rules_python",
    sha256 = PYRULES_SHA256,
    strip_prefix = "rules_python-" + PYRULES_COMMIT,
    urls = ["https://github.com/bazelbuild/rules_python/archive/{commit}.tar.gz".format(commit = PYRULES_COMMIT)]
)

load("@rules_python//python:repositories.bzl", "py_repositories")

py_repositories()

load("@rules_python//python/pip_install:repositories.bzl", "pip_install_dependencies")

pip_install_dependencies()

ENZYME_COMMIT = "bad7df07dd7657c0c6884667d62ad1d9bcfd1d16"
ENZYME_SHA256 = ""

http_archive(
    name = "enzyme",
    sha256 = ENZYME_SHA256,
    strip_prefix = "Enzyme-" + ENZYME_COMMIT + "/enzyme",
    urls = ["https://github.com/EnzymeAD/Enzyme/archive/{commit}.tar.gz".format(commit = ENZYME_COMMIT)],
)

JAX_COMMIT = "f691fe468a8e1f8545f7d624055d58b823ee3201"
JAX_SHA256 = ""

http_archive(
    name = "jax",
    sha256 = JAX_SHA256,
    strip_prefix = "jax-" + JAX_COMMIT,
    urls = ["https://github.com/google/jax/archive/{commit}.tar.gz".format(commit = JAX_COMMIT)],
    patch_args = ["-p1"],
    patches = ["//:patches/jax.patch"],
)

load("@jax//third_party/xla:workspace.bzl", jax_xla_workspace = "repo")
jax_xla_workspace()

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

load("@jax//third_party/robin_map:workspace.bzl", robin_map = "repo")
robin_map()

load("@jax//third_party/nanobind:workspace.bzl", nanobind = "repo")
nanobind()
