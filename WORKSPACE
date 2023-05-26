new_local_repository(
    name = "llvm-raw",
    build_file_content = "# empty",
    path = "llvm-project",
)

load("@llvm-raw//utils/bazel:configure.bzl", "llvm_configure", "llvm_disable_optional_support_deps")

llvm_disable_optional_support_deps()
llvm_configure(name = "llvm-project", targets = ["X86"])

local_repository(
    name = "xla",
    path = "xla",
)
local_repository(
    name = "enzyme",
    path = "Enzyme/enzyme",
)

local_repository(
    name = "rules_python",
    path = "rules_python",
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

local_repository(
    name = "jax",
    path = "jax",
)
