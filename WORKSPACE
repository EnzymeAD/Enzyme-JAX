load("//:workspace.bzl", "JAX_COMMIT", "JAX_SHA256", "ENZYME_COMMIT", "ENZYME_SHA256", "PYRULES_COMMIT", "PYRULES_SHA256", "XLA_PATCHES")
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
    patch_cmds = XLA_PATCHES
)

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
