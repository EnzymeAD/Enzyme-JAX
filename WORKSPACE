load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

git_repository(
    name = "llvm-raw",
    build_file_content = "# empty",
    remote = "https://github.com/llvm/llvm-project",
    commit = "c5f6a287499a816cba5585708999e2c8b134290f",
    patch_args = ["-p1"],
    patches = ["//:patches/llvm_build.patch"]
)

load("@llvm-raw//utils/bazel:configure.bzl", "llvm_configure", "llvm_disable_optional_support_deps")

llvm_disable_optional_support_deps()
llvm_configure(name = "llvm-project", targets = ["X86"])

git_repository(
    name = "xla",
    commit = "c1e4a16e77a7ba2000003ccade3ffba3749ada35",
    remote = "https://github.com/openxla/xla"
)

git_repository(
    name = "rules_python",
    commit = "693a1587baf055979493565933f8f40225c00c6d",
    remote = "https://github.com/bazelbuild/rules_python"
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

git_repository(
    name = "jax",
    commit = "21fc6e0229e0f5f1cb5f1f69d2c3daa2e5c2ca11",
    remote = "https://github.com/google/jax",
    patch_args = ["-p1"],
    patches = ["//:patches/jax_workspace.patch"]
)
