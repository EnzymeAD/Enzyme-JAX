JAX_COMMIT = "1014bf3d7c551921d501da2c857fbef468441cea"
JAX_SHA256 = ""

ENZYME_COMMIT = "c11fe5d30c914a4fca88ccc1a064169aa71afb2a"
ENZYME_SHA256 = ""

ML_TOOLCHAIN_COMMIT = "78ef5eda03c54a912c000f1f872242d4ca6063a4"
ML_TOOLCHAIN_SHA256 = ""

# If the empty string this will automatically use the commit above
# otherwise this should be a path to the folder containing the BUILD file for enzyme
OVERRIDE_ENZYME_PATH = ""

HEDRON_COMPILE_COMMANDS_COMMIT = "d107d9c9025915902fd52346f1c6e18d87f7013a"
HEDRON_COMPILE_COMMANDS_SHA256 = ""

XLA_PATCHES = [
    """
    sed -i.bak0 "s/\\\"-lamd_comgr\\\",//g" third_party/gpus/rocm/BUILD.tpl 
    """,
    """
    sed -i.bak0 "s/return TryDlopenCUDALibraries()/LOG(INFO) << \\"GPU libraries are statically linked, skip dlopen check.\\";\\nreturn absl::OkStatus();/g" xla/tsl/platform/default/dlopen_checker.cc
""",
    """
    sed -i.bak0 "s/return TryDlopenCUDALibraries()/LOG(INFO) << \\"GPU libraries are statically linked, skip dlopen check.\\";\\nreturn absl::OkStatus();/g" n
    sed -i.bak0 "s/namespace/THIS_SHOULD_NEVER_BE_COMPILED/g" xla/tsl/cuda/{cublas,cublasLt,cufft,cusolver,cusparse,cudnn,cudart}_stub.cc
""",
    """
	sed -i.bak0 "/amdgpu_backend/d" xla/backends/gpu/codegen/triton/BUILD
    """,
    """
    	sed -i.bak0 "s/\\\"\\/\\/xla\\/service\\/gpu\\/llvm_gpu_backend:nvptx_backend\\\"/\\0]) + if_rocm_is_configured([\\\"\\/\\/xla\\/service\\/gpu\\/llvm_gpu_backend:amdgpu_backend\\\"/g" xla/backends/gpu/codegen/triton/BUILD
    """,
    """
    sed -i.bak0 "s/load(\\\"\\/\\/xla\\/tsl:tsl.bzl\\\", \\\"if_google\\\")/load(\\\"\\/\\/xla\\/tsl:tsl.bzl\\\", \\\"if_google\\\")\\nload(\\\"@local_config_rocm\\/\\/rocm:build_defs.bzl\\\", \\\"if_rocm_is_configured\\\")/g" xla/backends/gpu/codegen/triton/BUILD
    """,
    """
    sed -i.bak0 "s,third_party/llvm/llvm-project/llvm/include/,,g" third_party/stablehlo/temporary.patch
    sed -i.bak0 "s,third_party/llvm/llvm-project/mlir/include/,,g" third_party/stablehlo/temporary.patch
    """,
    """
    sed -i.bak0 "s/\\/\\/third_party:repo.bzl/@bazel_tools\\/\\/tools\\/build_defs\\/repo:http.bzl/g" third_party/llvm/workspace.bzl
    """,
    """
    sed -i.bak0 "s/patch_file/patch_args = [\\\"-p1\\\"],patches/g" third_party/llvm/workspace.bzl
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
    sed -i.bak0 "s/def repo/load(\\\"@bazel_tools\\/\\/tools\\/build_defs\\/repo:http.bzl\\\", \\\"http_archive\\\")\\ndef repo/g" third_party/pthreadpool/workspace.bzl
    """,
    """
    sed -i.bak0 "s/tf_http_archive(/http_archive(/g" third_party/pthreadpool/workspace.bzl
    """,
    """
    sed -i.bak0 "s/strip_prefix/patch_cmds = [\\\"sed -i.bak0 's\\/_MSC_VER\\/_WIN32\\/g' src\\/pthreads.c\\\"], strip_prefix/g" third_party/pthreadpool/workspace.bzl
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
sed -i.bak0 "s/= \\[\\"@xla\\/\\/third_party\\/protobuf:protobuf.patch\\"/= \\[\\"@xla\\/\\/third_party\\/protobuf:protobuf.patch\\", \\"\\/\\/third_party:proto.patch\\"/g" third_party/py/python_init_rules.bzl

""",
    """
sed -i.bak0 's/registry\\["__chkstk"\\] = SymbolDef(__chkstk)/registry["__chkstk"] = SymbolDef(__chkstk_ms);\\nregistry["__chkstk_ms"] = SymbolDef(__chkstk_ms)/g' xla/backends/cpu/codegen/builtin_definition_generator.cc
""",
    """
sed -i.bak0 's/void __chkstk(size_t)/void __chkstk_ms(size_t)/g' xla/backends/cpu/codegen/builtin_definition_generator.cc
""",
    """
sed -i.bak0 "1s/^/#include \\"llvm\\/Support\\/DynamicLibrary.h\\"\\n/g" xla/backends/cpu/codegen/builtin_definition_generator.cc
""",
    """
sed -i.bak0 "s/SymbolDef(__chkstk_ms)/SymbolDef(reinterpret_cast<void* (*)(size_t)>(llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(\\"__chkstk_ms\\")))/g" xla/backends/cpu/codegen/builtin_definition_generator.cc
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
    """
    sed -i.bak0 "s/def repo/load(\\\"@bazel_tools\\/\\/tools\\/build_defs\\/repo:http.bzl\\\", \\\"http_archive\\\")\\ndef repo/g" third_party/eigen3/workspace.bzl
    sed -i.bak0 "s/tf_http_archive(/http_archive(/g" third_party/eigen3/workspace.bzl
    sed -i.bak0 "s/build_file = \\\"/build_file = \\\"@xla/g" third_party/eigen3/workspace.bzl

    sed -i.bak0 "s/urls = /patch_cmds = \\[\\\"sed -i.bak -e 's\\/return PACKET_TYPE(0) == PACKET_TYPE(0);\\/return (PACKET_TYPE)(PACKET_TYPE(0) == PACKET_TYPE(0));\\/g' -e 's\\/return CAST_FROM_INT(CAST_TO_INT(a) == CAST_TO_INT(a));\\/return CAST_FROM_INT((decltype(CAST_TO_INT(a)))(CAST_TO_INT(a) == CAST_TO_INT(a)));\\/' Eigen\\/src\\/Core\\/arch\\/clang\\/PacketMath.h\\\"\\],urls = /g" third_party/eigen3/workspace.bzl
    """,
]

LLVM_TARGETS = ["X86", "AArch64", "AMDGPU", "NVPTX"]
