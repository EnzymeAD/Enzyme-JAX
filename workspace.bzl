JAX_COMMIT = "545bb182a003584c5bd7d77e270381656b803ce0"
JAX_SHA256 = ""

ENZYME_COMMIT = "4ec5801f3c56ce1b26f9ce68ea5ca20bbb8bbd00"
ENZYME_SHA256 = ""

# If the empty string this will automatically use the commit above
# otherwise this should be a path to the folder containing the BUILD file for enzyme
OVERRIDE_ENZYME_PATH = ""

HEDRON_COMPILE_COMMANDS_COMMIT = "4f28899228fb3ad0126897876f147ca15026151e"
HEDRON_COMPILE_COMMANDS_SHA256 = ""

XLA_PATCHES = [
    """
    sed -i.bak0 "s/e07debd5e257ec1e118f18c54068977b89f03b2f/9018c682b99eb20d5874a4e38271ce63d7393879/g" third_party/stablehlo/workspace.bzl
    """,
    """
    sed -i.bak0 "s/\\/\\/third_party:repo.bzl/@bazel_tools\\/\\/tools\\/build_defs\\/repo:http.bzl/g" third_party/llvm/workspace.bzl
    """,
    """
    sed -i.bak0 "s/patch_file/patch_args = [\\\"-p1\\\"],patches/g" third_party/llvm/workspace.bzl
    """,
    """
    sed -i.bak0 "s/patches = \\[/ patches = \\[\\\"\\/\\/:patches\\/llvm.patch\\\",/g" third_party/llvm/workspace.bzl
    """,
    """
    sed -i.bak0 "s/patches = \\[/ patches = \\[\\\"\\/\\/:patches\\/llvm2.patch\\\",/g" third_party/llvm/workspace.bzl
    """,
    # TODO remove
    """
    sed -i.bak0 "s/DCHECK_NE(runtime, nullptr/DCHECK_NE(runtime.get(), nullptr/g" xla/backends/cpu/runtime/xnnpack/xnn_fusion_thunk.cc
    """,
    # TODO remove
    """
    sed -i.bak0 "s/^bool IsSupportedType/static inline bool IsSupportedType/g" xla/backends/cpu/runtime/convolution_lib.cc
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
    "find xla/mlir -type f -name BUILD -exec sed -i.bak3 's/\\/\\/xla:internal/\\/\\/\\/\\/visibility:public/g' {} +",
]

LLVM_TARGETS = ["X86", "AArch64", "AMDGPU", "NVPTX"]
