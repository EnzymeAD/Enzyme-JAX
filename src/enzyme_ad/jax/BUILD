load("@jax//jaxlib:symlink_files.bzl", "symlink_inputs")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension", "pybind_library")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@llvm-project//llvm:tblgen.bzl", "gentbl")

licenses(["notice"])

package(
    default_visibility = ["//:__subpackages__"],
    features = ["layering_check"],
)

pybind_library(
    name = "clang_compile",
    srcs = ["clang_compile.cc"],
    hdrs = ["clang_compile.h"],
    deps = [
        "@enzyme//:EnzymeStatic",
        "@llvm-project//clang:ast",
        "@llvm-project//clang:basic",
        "@llvm-project//clang:codegen",
        "@llvm-project//clang:driver",
        "@llvm-project//clang:frontend",
        "@llvm-project//clang:frontend_tool",
        "@llvm-project//clang:lex",
        "@llvm-project//clang:parse",
        "@llvm-project//clang:sema",
        "@llvm-project//clang:serialization",
        "@llvm-project//llvm:AsmParser",
        "@llvm-project//llvm:CodeGen",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:Linker",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:Support",
    ],
)

py_library(
    name = "enzyme_jax_internal",
    srcs = ["primitives.py", "__init__.py"],
    visibility = ["//visibility:public"]
)

symlink_inputs(
    name = "EnzymeImplementationsCommonTdFiles",
    rule = td_library,
    symlinked_inputs = {"srcs": {
        "Implementations": ["@enzyme//:ImplementationsCommonTdFiles"],
    }},
)

td_library(
    name = "ImplementationsCommonTdFiles",
    srcs = [
         ":EnzymeImplementationsCommonTdFiles",
    ],
    deps = [
        ":EnzymeImplementationsCommonTdFiles",
    ],
)

gentbl_cc_library(
    name = "mhlo-derivatives",
    tbl_outs = [(
        ["-gen-mlir-derivatives"],
        "Implementations/MHLODerivatives.inc",
    )],
    tblgen = "@enzyme//:enzyme-tblgen",
    td_file = "Implementations/MHLODerivatives.td",
    td_srcs = [
        "Implementations/MHLODerivatives.td",
        "Implementations/HLODerivatives.td",
    ],
    deps = [
        "@enzyme//:enzyme-tblgen",
        ":ImplementationsCommonTdFiles",
    ],
)

gentbl_cc_library(
    name = "stablehlo-derivatives",
    tbl_outs = [(
        ["-gen-mlir-derivatives"],
        "Implementations/StableHLODerivatives.inc",
    )],
    tblgen = "@enzyme//:enzyme-tblgen",
    td_file = "Implementations/StableHLODerivatives.td",
    td_srcs = [
        "Implementations/StableHLODerivatives.td",
        "Implementations/HLODerivatives.td",
    ],
    deps = [
        "@enzyme//:enzyme-tblgen",
        ":EnzymeImplementationsCommonTdFiles",
    ],
)

td_library(
    name = "EnzymeXLAPassesTdFiles",
    srcs = [
    ],
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "EnzymeXLAPassesIncGen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=enzymexla",
            ],
            "Passes/Passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Passes/Passes.td",
    deps = [":EnzymeXLAPassesTdFiles"],
)

cc_library(
    name = "XLADerivatives",
    srcs = glob(
        [
            "Implementations/*.cpp",
            "Passes/*.cpp",
        ],
    ),
    hdrs = glob([
        "Implementations/*.h",
        "Passes/*.h",
    ]),
    deps = [
        ":EnzymeXLAPassesIncGen",
        ":mhlo-derivatives",
        ":stablehlo-derivatives",
        "@stablehlo//:stablehlo_ops",
        "@xla//xla/mlir_hlo",
        "@enzyme//:EnzymeMLIR",
    ]
)

pybind_library(
    name = "compile_with_xla",
    srcs = ["compile_with_xla.cc"],
    hdrs = glob(["compile_with_xla.h", "Implementations/*.h", "Passes/*.h"]),
    deps = [
        ":XLADerivatives",
        # This is similar to xla_binary rule and is needed to make XLA client compile.
        "@tsl//tsl/framework:allocator",
        "@tsl//tsl/framework:allocator_registry_impl",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:env_impl",
        "@tsl//tsl/platform:tensor_float_32_utils",
        "@tsl//tsl/profiler/backends/cpu:annotation_stack_impl",
        "@tsl//tsl/profiler/backends/cpu:traceme_recorder_impl",
        "@tsl//tsl/profiler/utils:time_utils_impl",
        "@tsl//tsl/protobuf:dnn_proto_cc_impl",
        "@tsl//tsl/protobuf:histogram_proto_cc",
        "@tsl//tsl/protobuf:histogram_proto_cc_impl",
        "@tsl//tsl/protobuf:protos_all_cc_impl",
        "@tsl//tsl/util:determinism",

        # This is similar to xla_binary rule and is needed to make XLA client compile.
        "@xla//xla:autotune_results_proto_cc",
        "@xla//xla:autotune_results_proto_cc_impl",
        "@xla//xla:autotuning_proto_cc",
        "@xla//xla:autotuning_proto_cc_impl",
        "@xla//xla/client",
        "@xla//xla/client:client_library",
        "@xla//xla/client:executable_build_options",
        "@xla//xla/client:xla_computation",
        "@xla//xla/service:buffer_assignment_proto_cc",
        "@xla//xla/service:buffer_assignment_proto_cc_impl",
        "@xla//xla/service/cpu:cpu_executable",
        "@xla//xla/service/cpu:backend_config_proto_cc",
        "@xla//xla/service/gpu:backend_configs_cc",
        "@xla//xla/service/gpu:backend_configs_cc_impl",
        "@xla//xla/service:hlo_proto_cc",
        "@xla//xla/service:hlo_proto_cc_impl",
        "@xla//xla/service/memory_space_assignment:memory_space_assignment_proto_cc_impl",
        "@xla//xla/stream_executor:device_description_proto_cc",
        "@xla//xla/stream_executor:device_description_proto_cc_impl",
        "@xla//xla/translate/mhlo_to_hlo:type_to_shape",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla:xla_data_proto_cc_impl",
        "@xla//xla:xla_proto_cc",
        "@xla//xla:xla_proto_cc_impl",

        # Make CPU target available to XLA.
        "@xla//xla/service:cpu_plugin",

        # MHLO stuff.
        "@xla//xla/mlir_hlo",
        "@xla//xla/translate/mhlo_to_hlo:mlir_hlo_to_hlo",

        # This is necessary for XLA protobufs to link
        "@com_google_protobuf//:protobuf",

        # MLIR dialects and parser.
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Parser",
	
    # EnzymeMLIR
        "@enzyme//:EnzymeMLIR",
       
        # Mosaic
	"@jax//jaxlib/mosaic:tpu_dialect",
    ],
)

pybind_extension(
    name = "enzyme_call",
    srcs = ["enzyme_call.cc"],
    deps = [
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:ExecutionEngine",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:OrcTargetProcess",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        ":clang_compile",
        ":compile_with_xla",
        "@com_google_absl//absl/status:statusor",
        "@stablehlo//:stablehlo_passes",
        "@xla//xla/stream_executor:stream_executor_impl",
    ],
    visibility = ["//visibility:public"],
)
