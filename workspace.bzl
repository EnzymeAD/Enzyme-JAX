JAX_COMMIT = "3a22eea644237001df0f3dd42253225cc059b43c"
JAX_SHA256 = ""

ENZYME_COMMIT = "f10a216d9a40e2d85547572776a41cd9054fe49d"
ENZYME_SHA256 = ""

ML_TOOLCHAIN_COMMIT = "d8d8f49297a1e74fcceffc9ef6c7f8da9b0a0c53"
ML_TOOLCHAIN_SHA256 = "4133c6c2045de5d7a133f6fc008ee6bd613af778f12143d09003e908dd541d8c"

# If the empty string this will automatically use the commit above
# otherwise this should be a path to the folder containing the BUILD file for enzyme
OVERRIDE_ENZYME_PATH = ""

HEDRON_COMPILE_COMMANDS_COMMIT = "4f28899228fb3ad0126897876f147ca15026151e"
HEDRON_COMPILE_COMMANDS_SHA256 = ""

XLA_PATCHES = [
    """
    sed -i.bak0 "s/\\\"\\/\\/xla\\/service\\/gpu\\/llvm_gpu_backend:amdgpu_backend\\\"/] + if_rocm_is_configured([\\0]) + [/g" xla/backends/gpu/codegen/triton/BUILD
    """,
    """
    sed -i.bak0 "s/\\\"if_cuda_is_configured\\\",/\\0\\\"if_rocm_is_configured\\\",/g" xla/backends/gpu/codegen/triton/BUILD
    """,
    """
    sed -i.bak0 "s/e07debd5e257ec1e118f18c54068977b89f03b2f/9018c682b99eb20d5874a4e38271ce63d7393879/g" third_party/stablehlo/workspace.bzl
    """,
    """
    sed -i.bak0 "s/\\/\\/third_party:repo.bzl/@bazel_tools\\/\\/tools\\/build_defs\\/repo:http.bzl/g" third_party/llvm/workspace.bzl
    """,
    """
    sed -i.bak0 "s/patch_file/patch_args = [\\\"-p1\\\"],patches/g" third_party/llvm/workspace.bzl
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
    sed -i.bak0 "s/Node::Leaf(std::forward<decltype(value)>/Node::Leaf(std::forward<T>/g" xla/tuple_tree.h
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
    """
echo "--- a/src/google/protobuf/stubs/port.h" >> third_party/proto.patch
echo "+++ b/src/google/protobuf/stubs/port.h" >> third_party/proto.patch
echo "@@ -27,7 +27,7 @@" >> third_party/proto.patch
echo " #include <intrin.h>" >> third_party/proto.patch
echo " #elif defined(__APPLE__)" >> third_party/proto.patch
echo " #include <libkern/OSByteOrder.h>" >> third_party/proto.patch
echo "-#elif defined(__linux__) || defined(__ANDROID__) || defined(__CYGWIN__)" >> third_party/proto.patch
echo "+#elif !defined(__NVCC__) && (defined(__linux__) || defined(__ANDROID__) || defined(__CYGWIN__))" >> third_party/proto.patch
echo " #include <byteswap.h>  // IWYU pragma: export" >> third_party/proto.patch
echo " #endif" >> third_party/proto.patch
echo "" >> third_party/proto.patch
echo "@@ -143,7 +143,7 @@" >> third_party/proto.patch
echo " #define bswap_32(x) OSSwapInt32(x)" >> third_party/proto.patch
echo " #define bswap_64(x) OSSwapInt64(x)" >> third_party/proto.patch
echo "" >> third_party/proto.patch
echo "-#elif !defined(__linux__) && !defined(__ANDROID__) && !defined(__CYGWIN__)" >> third_party/proto.patch
echo "+#elif defined(__NVCC__) || (!defined(__linux__) && !defined(__ANDROID__) && !defined(__CYGWIN__))" >> third_party/proto.patch
echo "" >> third_party/proto.patch
echo " #ifndef bswap_16" >> third_party/proto.patch
echo " static inline uint16_t bswap_16(uint16_t x) {" >> third_party/proto.patch
sed -i.bak0 "s/protobuf.patch\\"/protobuf.patch\\", \\":proto.patch\\"/g" workspace2.bzl
sed -i.bak0 "s/patch_file = \\[\\"@xla\\/\\/third_party\\/protobuf:protobuf.patch\\"/patches = \\[Label(\\"@xla\\/\\/third_party\\/protobuf:protobuf.patch\\"), Label(\\"\\/\\/third_party:proto.patch\\"\\)], patch_args = \\[\\"-p1\\"/g" third_party/py/python_init_rules.bzl
sed -i.bak0 "s/tf_http_archive(/http_archive(/g" third_party/py/python_init_rules.bzl

""",
    """
sed -i.bak0 "s/def main():/def main():\\n  if TMPDIR: os.environ['TMPDIR'] = TMPDIR/g" third_party/gpus/crosstool/clang/bin/crosstool_wrapper_driver_is_not_gcc.tpl
""",
    """
sed -i.bak0 "s/__chkstk/__chkstk_ms/g" xla/service/cpu/runtime_symbol_generator.cc
""",
    """
sed -i.bak0 "1s/^/#include \\"llvm\\/Support\\/DynamicLibrary.h\\"\\n/g" xla/service/cpu/runtime_symbol_generator.cc
""",
    """
sed -i.bak0 "s/(__chkstk_ms)/(llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(\\"__chkstk_ms\\"))/g" xla/service/cpu/runtime_symbol_generator.cc
""",
    """
sed -i.bak0 "s/Shlwapi/shlwapi/g" xla/tsl/platform/windows/load_library.cc xla/tsl/platform/windows/windows_file_system.cc xla/tsl/platform/windows/env.cc
""",
    """
sed -i.bak0 "1s/^/#ifdef PLATFORM_WINDOWS\\n#include <immintrin.h>\\n#include <intrin.h>\\n#endif/g" third_party/tsl/tsl/platform/cpu_info.cc
""",
    """
sed -i.bak0 "1s/^/#define _USE_MATH_DEFINES\\n/g" xla/fp_util.h xla/hlo/builder/lib/prng.cc xla/literal_comparison.cc xla/hlo/builder/lib/math.cc xla/service/spmd/fft_handler.cc xla/service/cpu/onednn_contraction_rewriter.cc xla/hlo/evaluator/hlo_evaluator.cc
""",
    """
sed -i.bak0 "s/Windows\\.h/windows\\.h/g" xla/tsl/platform/windows/port.cc xla/tsl/platform/windows/wide_char.cc xla/tsl/platform/windows/env.cc xla/tsl/platform/windows/windows_file_system.cc
""",
    """
sed -i.bak0 "/D_FORTIFY_SOURCE/d" third_party/gpus/crosstool/cc_toolchain_config.bzl.tpl tools/toolchains/cross_compile/cc/BUILD tools/toolchains/clang6/CROSSTOOL.tpl third_party/gpus/crosstool/BUILD.rocm.tpl
""",
    """
sed -i.bak0 "s/i64/LL/g" xla/tsl/platform/windows/env_time.cc
""",
    """
sed -i.bak0 "s/\\/D/-D/g" third_party/farmhash/farmhash.BUILD
""",
    """
sed -i.bak0 "s/Node::Leaf(std::forward<decltype(pair.second)>/Node::Leaf(std::forward<T>/g" xla/tuple_tree.h
""",
    """
sed -i.bak0 "s/kDeprecatedFlags({/kDeprecatedFlags(absl::flat_hash_set<std::string>{/g" xla/debug_options_flags.cc
""",
    """
sed -i.bak0 "s/kStableFlags({/kStableFlags(absl::flat_hash_set<std::string>{/g" xla/debug_options_flags.cc
""",
    """
sed -i.bak0 "s/cupti_driver_cbid/cupti/g" xla/backends/profiler/gpu/cupti_tracer.cc
""",
    """
sed -i.bak0 "s/patch_cmds = \\[/patch_cmds = \\[\\\"find . -type f -name config.bzl -exec sed -i.bak0 's\\/HAVE_LINK_H=1\\/HAVE_LINK_H=0\\/g' {} +\\\",/g" third_party/llvm/workspace.bzl
""",
    """
sed -i.bak0 "s/patch_cmds = \\[/patch_cmds = \\[\\\"find . -type f -name config.bzl -exec sed -i.bak0 's\\/LLVM_ENABLE_THREADS=1\\/LLVM_ENABLE_THREADS=0\\/g' {} +\\\",/g" third_party/llvm/workspace.bzl
""",
    """
sed -i.bak0 "s/patch_cmds = \\[/patch_cmds = \\[\\\"find . -type f -name config.bzl -exec sed -i.bak0 's\\/HAVE_MALLINFO=1\\/DONT_HAVE_ANY_MALLINFO=0\\/g' {} +\\\",/g" third_party/llvm/workspace.bzl
""",
    """
sed -i.bak0 "s/patch_cmds = \\[/patch_cmds = \\[\\\"find . -type f -name config.bzl -exec sed -i.bak0 's\\/HAVE_PTHREAD_GETNAME_NP=1\\/FAKE_HAVE_PTHREAD_GETNAME_NP=0\\/g' {} +\\\",/g" third_party/llvm/workspace.bzl
""",
    """
sed -i.bak0 "s/patch_cmds = \\[/patch_cmds = \\[\\\"find . -type f -name config.bzl -exec sed -i.bak0 's\\/HAVE_PTHREAD_SETNAME_NP=1\\/FAKE_HAVE_PTHREAD_SETNAME_NP=0\\/g' {} +\\\",/g" third_party/llvm/workspace.bzl
""",
    """
sed -i.bak0 "s/patch_cmds = \\[/patch_cmds = \\[\\\"find . -type f -name config.h -exec sed -i.bak0 's\\/ENABLE_CRASH_OVERRIDES 1\\/ENABLE_CRASH_OVERRIDES 0\\/g' {} +\\\",/g" third_party/llvm/workspace.bzl
""",
    """
sed -i.bak0 "s/patch_cmds = \\[/patch_cmds = \\[\\\"find . -type f -name config.h -exec sed -i.bak0 's\\/HAVE_PTHREAD_GETNAME_NP\\/FAKE_HAVE_PTHREAD_GETNAME_NP\\/g' {} +\\\",/g" third_party/llvm/workspace.bzl
""",
    """
sed -i.bak0 "s/patch_cmds = \\[/patch_cmds = \\[\\\"find . -type f -name config.h -exec sed -i.bak0 's\\/HAVE_PTHREAD_SETNAME_NP\\/FAKE_HAVE_PTHREAD_SETNAME_NP\\/g' {} +\\\",/g" third_party/llvm/workspace.bzl
""",
]

LLVM_TARGETS = ["X86", "AArch64", "AMDGPU", "NVPTX"]
