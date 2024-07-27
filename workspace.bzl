JAX_COMMIT = "084b0f9624db53f2760ee6c39c44ce72443e4ae3"
JAX_SHA256 = "be1a13156a48e27520b75a1f456771dd6500ee67b2711cf20ac5698e2848a108"

ENZYME_COMMIT = "6e8dd3e3faff7e766a9e957bf4068d6fcda54539"
ENZYME_SHA256 = ""

XLA_PATCHES = [
    """
    sed -i.bak0 "s/\\/\\/third_party:repo.bzl/@bazel_tools\\/\\/tools\\/build_defs\\/repo:http.bzl/g" third_party/llvm/workspace.bzl
    """,
    """
    sed -i.bak0 "s/patch_file/patch_args = [\\\"-p1\\\"],patches/g" third_party/llvm/workspace.bzl
    """,
    """
    sed -i.bak0 "/link_file/d" third_party/llvm/workspace.bzl
    """,
    """
    sed -i.bak0 "s/build_file.*/build_file_content = \\\"# empty\\\",/g" third_party/llvm/workspace.bzl
    """,
    """
    sed -i.bak0 "s/\\/\\/third_party/@xla\\/\\/third_party/g" third_party/llvm/workspace.bzl
    """,
    """
    sed -i.bak0 "s/tf_http_archive/http_archive/g" third_party/llvm/workspace.bzl
    """,
    """
    sed -i.bak0 "s/strip_prefix/patch_cmds = [\\\"find . -type f -name config.bzl -exec sed -i.bak0 's\\/HAVE_BACKTRACE=1\\/NO_HAVE_BACKTRACE=0\\/g' {} +\\\"], strip_prefix/g" third_party/llvm/workspace.bzl
    """,
    "find . -type f -name BUILD -exec sed -i.bak1 's/\\/\\/third_party\\/py\\/enzyme_ad\\/\\.\\.\\./public/g' {} +", 
    "find . -type f -name BUILD -exec sed -i.bak2 's/\\/\\/xla\\/mlir\\/memref:friends/\\/\\/visibility:public/g' {} +",
    "find xla/mlir -type f -name BUILD -exec sed -i.bak3 's/\\/\\/xla:internal/\\/\\/\\/\\/visibility:public/g' {} +"
]
