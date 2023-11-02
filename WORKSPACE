load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

LLVM_COMMIT = "aa495214b39d475bab24b468de7a7c676ce9e366"
LLVM_SHA256 = "73cb1e91901d975bf4c97f1ea7000dd1554ad77f704a2d899498866a67471444"

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
llvm_configure(name = "llvm-project", targets = ["X86", "AArch64", "AMDGPU", "ARM", "NVPTX"])

XLA_COMMIT = "7423e38a383ccd25fb144db298257a6b6dd8cc4d"
XLA_SHA256 = "e7ea840e4a58a91bdb5dbcaee71669dd799cecc8d2084106389699619fe76129"

http_archive(
    name = "xla",
    sha256 = XLA_SHA256,
    strip_prefix = "xla-" + XLA_COMMIT,
    urls = ["https://github.com/openxla/xla/archive/{commit}.tar.gz".format(commit = XLA_COMMIT)],
    patch_args = ["-p1"],
    patches = ["//:patches/xla.patch"],
)


PYRULES_COMMIT = "693a1587baf055979493565933f8f40225c00c6d"
PYRULES_SHA256 = "c493a9506b5e1ea99e3c22fb15e672cdd2a6fa19fd7c627ec5d485aced23e50a"

http_archive(
    name = "rules_python",
    sha256 = PYRULES_SHA256,
    strip_prefix = "rules_python-" + PYRULES_COMMIT,
    urls = ["https://github.com/bazelbuild/rules_python/archive/{commit}.tar.gz".format(commit = PYRULES_COMMIT)]
)

ENZYME_COMMIT = "bcd061afc6260d2266ca9a8489830c36a4ceefe6"
ENZYME_SHA256 = "f215f6654b000a7eed387d89fe51561382dc7a86e1ed83941399335f819c1f66"

http_archive(
    name = "enzyme",
    sha256 = ENZYME_SHA256,
    strip_prefix = "Enzyme-" + ENZYME_COMMIT + "/enzyme",
    urls = ["https://github.com/EnzymeAD/Enzyme/archive/{commit}.tar.gz".format(commit = ENZYME_COMMIT)],
)

JAX_COMMIT = "21fc6e0229e0f5f1cb5f1f69d2c3daa2e5c2ca11"
JAX_SHA256 = "fe6d76285eef8cfd4b3ec7ec61240f92acabf554576111ee0c31d96fb6a746ce"

http_archive(
    name = "jax",
    sha256 = JAX_SHA256,
    strip_prefix = "jax-" + JAX_COMMIT,
    urls = ["https://github.com/google/jax/archive/{commit}.tar.gz".format(commit = JAX_COMMIT)],
    patch_args = ["-p1"],
    patches = ["//:patches/jax.patch"],
)

load("@jax//third_party/ducc:workspace.bzl", ducc = "repo")
ducc()

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
