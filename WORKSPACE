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

LLVM_COMMIT = "668865789620f390fbad4d7093ed8ca6eb932c31"
LLVM_SHA256 = "8d7cbbe492a17656c09af1e79b802303f11cb47d64768760b70d52f11ed4d9da"
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

XLA_COMMIT = "a6e6c1f6a53d4a23451c649110519c7ba8581bf9"
XLA_SHA256 = "5fe6dfa30621bd50b022a6cab026d6f4cde9883a3e150ce1b6fd52822a57c59a"

http_archive(
    name = "xla",
    sha256 = XLA_SHA256,
    strip_prefix = "xla-" + XLA_COMMIT,
    urls = ["https://github.com/openxla/xla/archive/{commit}.tar.gz".format(commit = XLA_COMMIT)],
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

ENZYME_COMMIT = "77b4fff47701a240b537a93a2e722626f7421342"
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
