load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Replace with the LLVM commit you want to use.
LLVM_COMMIT = "c5f6a287499a816cba5585708999e2c8b134290f"
LLVM_SHA256 = "03a8eb4b243846ee037d700b048ec48a87eeef480cb129ab56aa7e0537172b98"

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
llvm_configure(name = "llvm-project", targets = ["HOST","NVPTX"])

XLA_COMMIT = "7423e38a383ccd25fb144db298257a6b6dd8cc4d"
XLA_SHA256 = "e7ea840e4a58a91bdb5dbcaee71669dd799cecc8d2084106389699619fe76129"

http_archive(
    name = "xla",
    sha256 = XLA_SHA256,
    strip_prefix = "xla-" + XLA_COMMIT,
    urls = ["https://github.com/openxla/xla/archive/{commit}.tar.gz".format(commit = XLA_COMMIT)]
)


PYRULES_COMMIT = "693a1587baf055979493565933f8f40225c00c6d"
PYRULES_SHA256 = "c493a9506b5e1ea99e3c22fb15e672cdd2a6fa19fd7c627ec5d485aced23e50a"

http_archive(
    name = "rules_python",
    sha256 = PYRULES_SHA256,
    strip_prefix = "rules_python-" + PYRULES_COMMIT,
    urls = ["https://github.com/bazelbuild/rules_python/archive/{commit}.tar.gz".format(commit = PYRULES_COMMIT)]
)

local_repository(
    name = "enzyme",
    path = "Enzyme/enzyme",
)

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


JAX_COMMIT = "21fc6e0229e0f5f1cb5f1f69d2c3daa2e5c2ca11"
JAX_SHA256 = ""

http_archive(
    name = "jax",
    sha256 = JAX_SHA256,
    strip_prefix = "jax-" + PYRULES_COMMIT,
    urls = ["https://github.com/google/jax/archive/{commit}.tar.gz".format(commit = JAX_COMMIT)],
    patch_args = ["-p1"],
    patches = ["//:patches/jax_workspace.patch"]
)
