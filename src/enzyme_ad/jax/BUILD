load("@jax//jaxlib:symlink_files.bzl", "symlink_inputs")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension", "pybind_library")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

exports_files(["enzymexlamlir-opt.cpp"])

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
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
    ],
)

py_library(
    name = "enzyme_jax_internal",
    srcs = [
        "__init__.py",
        "primitives.py",
        "utils.py",
    ],
    visibility = ["//visibility:public"],
)

symlink_inputs(
    name = "EnzymeImplementationsCommonTdFiles",
    rule = td_library,
    symlinked_inputs = {"srcs": {
        "Implementations": ["@enzyme//:ImplementationsCommonTdFiles"],
    }},
)

td_library(
    name = "TransformOpsTdFiles",
    srcs = [
        "TransformOps/TransformOps.td",
    ],
    deps = [
        "@llvm-project//mlir:TransformDialectTdFiles",
    ]
)

gentbl_cc_library(
    name = "TransformOpsIncGen",
    tbl_outs = [(
            ["-gen-op-decls"],
            "TransformOps/TransformOps.h.inc",
        ), (
            ["-gen-op-defs"],
            "TransformOps/TransformOps.cpp.inc",
        ), (
            ["-gen-op-interface-decls"],
            "TransformOps/OpInterfaces.h.inc",
        ), (
            ["-gen-op-interface-defs"],
            "TransformOps/OpInterfaces.cpp.inc",
        ),
    ],
    td_file = "TransformOps/TransformOps.td",
    deps = [
        ":TransformOpsTdFiles",
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
)

gentbl_cc_library(
    name = "TransformOpsImplIncGen",
    tbl_outs = [(
        ["-gen-populate-patterns-interface-impl"],
        "TransformOps/TransformOpsImpl.cpp.inc"
    )],
    td_file = "TransformOps/TransformOps.td",
    deps = [
        ":TransformOpsTdFiles",
    ],
    tblgen = "//:enzymexlamlir-tblgen",
)

cc_library(
    name = "TransformOps",
    srcs = glob(["TransformOps/*.cpp"]),
    hdrs = glob(["TransformOps/*.h"]),
    deps = [
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgTransformOps",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TransformDialect",
        "@llvm-project//mlir:TransformDialectInterfaces",
        ":TransformOpsIncGen",
        ":TransformOpsImplIncGen",
        ":XLADerivatives",
    ],
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
        ":EnzymeImplementationsCommonTdFiles",
    ],
)

gentbl_cc_library(
    name = "chlo-derivatives",
    tbl_outs = [(
        ["-gen-mlir-derivatives"],
        "Implementations/CHLODerivatives.inc",
    )],
    tblgen = "@enzyme//:enzyme-tblgen",
    td_file = "Implementations/CHLODerivatives.td",
    td_srcs = [
        "Implementations/CHLODerivatives.td",
    ],
    deps = [
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

gentbl_cc_library(
    name = "EnzyeHLOPatternsIncGen",
    tbl_outs = [
        (
            ["-gen-populate-patterns-func-decls"],
            "Passes/EnzymeHLOPatterns.h.inc",
        ), (
            ["-gen-populate-patterns-func-defs"],
            "Passes/EnzymeHLOPatterns.cpp.inc",
        )],
    td_file = "TransformOps/TransformOps.td",
    deps = [
        ":TransformOpsTdFiles",
    ],
    tblgen = "//:enzymexlamlir-tblgen",
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
    copts = [
        "-Werror=unused-variable",
        "-Werror=unused-but-set-variable",
        "-Werror=return-type",
        "-Werror=unused-result",
    ],
    deps = [
        ":EnzymeXLAPassesIncGen",
        ":EnzyeHLOPatternsIncGen",
        ":mhlo-derivatives",
        ":stablehlo-derivatives",
        ":chlo-derivatives",
        "@enzyme//:EnzymeMLIR",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:CommonFolders",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@stablehlo//:reference_ops",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:stablehlo_passes",
        "@stablehlo//:stablehlo_type_inference",
        "@xla//xla/mlir_hlo",
    ],
)

pybind_library(
    name = "compile_with_xla",
    srcs = ["compile_with_xla.cc"],
    hdrs = glob([
        "compile_with_xla.h",
        "Implementations/*.h",
        "Passes/*.h",
    ]),
    deps = [
        ":XLADerivatives",
        ":TransformOps",
        # This is similar to xla_binary rule and is needed to make XLA client compile.
        # "@tsl//tsl/framework:allocator",
        # "@tsl//tsl/framework:allocator_registry_impl",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:env_impl",
        "@tsl//tsl/platform:tensor_float_32_utils",
	"@xla//xla/tsl/profiler/backends/cpu:annotation_stack_impl",
        "@xla//xla/tsl/profiler/backends/cpu:traceme_recorder_impl",
        "@xla//xla/tsl/profiler/utils:time_utils_impl",
        "@xla//xla/tsl/protobuf:dnn_proto_cc_impl",
        "@xla//xla/tsl/protobuf:histogram_proto_cc",
        "@xla//xla/tsl/protobuf:histogram_proto_cc_impl",
        "@xla//xla/tsl/protobuf:protos_all_cc_impl",

        # This is similar to xla_binary rule and is needed to make XLA client compile.
        "@xla//xla:autotune_results_proto_cc",
        "@xla//xla:autotune_results_proto_cc_impl",
        "@xla//xla:autotuning_proto_cc",
        "@xla//xla:autotuning_proto_cc_impl",
        "@xla//xla/client",
        "@xla//xla/client:client_library",
        "@xla//xla/client:executable_build_options",
        "@xla//xla/client:local_client",
        "@xla//xla/client:xla_computation",
        "@xla//xla/service",
        "@xla//xla/service:metrics_proto_cc",
        "@xla//xla/service:metrics_proto_cc_impl",
        "@xla//xla/service:local_service",
        "@xla//xla/service:local_service_utils",
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
        "@xla//xla/service/cpu:cpu_compiler_pure",
        "@xla//xla/service:compiler",
        "@xla//xla/service:hlo_module_util",
        "@xla//xla/service:hlo_proto_util",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:chlo_ops",

        # Make CPU target available to XLA.
        "@xla//xla/service:cpu_plugin",

        # MHLO stuff.
        "@xla//xla/mlir_hlo",
        "@xla//xla/translate/mhlo_to_hlo:mlir_hlo_to_hlo",
        "@xla//xla/hlo/ir:hlo",

        # This is necessary for XLA protobufs to link
        "@com_google_protobuf//:protobuf",

        # MLIR dialects and parser.
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:LinalgTransformOps",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@xla//xla/mlir_hlo:all_passes",
        "@xla//xla:printer",

        # EnzymeMLIR
        "@enzyme//:EnzymeMLIR",
        "@com_google_absl//absl/status:statusor",

        # Mosaic
        "@jax//jaxlib/mosaic:tpu_dialect",
    ],
)

pybind_extension(
    name = "enzyme_call",
    srcs = ["enzyme_call.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":clang_compile",
        ":compile_with_xla",
        ":TransformOps",
        "@com_google_absl//absl/status:statusor",
        "@enzyme//:EnzymeMLIR",
        "@enzyme//:EnzymeStatic",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:ExecutionEngine",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:OrcTargetProcess",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:CAPIIR",
        "@llvm-project//mlir:MLIRBindingsPythonHeaders",
        "@stablehlo//:stablehlo_passes",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/mlir_hlo:all_passes",
        "@xla//xla/mlir_hlo:deallocation_passes",
        "@xla//xla/service/cpu:cpu_executable",
        "@xla//xla/stream_executor:stream_executor_impl",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:env_impl",
    ],
)
