load("@jax//jaxlib:symlink_files.bzl", "symlink_inputs")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension", "pybind_library")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@rules_rust//rust:defs.bzl", "rust_binary", "rust_library")
load("@tensat_crate_index//:defs.bzl", "all_crate_deps")
load("deps/rust_cxx_bridge.bzl", "rust_cxx_bridge")

exports_files(["enzymexlamlir-opt.cpp", "deps/tensat/converted.txt", "deps/tensat/extractor/extract.py"])
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
    alwayslink = True
)

py_library(
    name = "enzyme_jax_internal",
    srcs = [
        "__init__.py",
        "primitives.py",
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
            "deps/*.cpp",
        ]
    ),
    hdrs = glob([
        "Implementations/*.h",
        "Passes/*.h",
        "deps/include/*.h",
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
        ":tensat_lib",
        "@enzyme//:EnzymeMLIR",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:AsyncDialect",
        "@llvm-project//mlir:CommonFolders",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:ConversionPasses",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:DLTIDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:OpenMPDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformDialect",
        "@llvm-project//mlir:TransformDialectInterfaces",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:TransformUtils",

        "@stablehlo//:reference_ops",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:stablehlo_passes",
        "@stablehlo//:stablehlo_type_inference",
        "@xla//xla/mlir_hlo",

        "@xla//xla/pjrt:pjrt_api",
        "@xla//xla/pjrt/cpu:cpu_client",
        "@xla//xla/pjrt:pjrt_c_api_client",
        "@xla//xla/pjrt/gpu:se_gpu_pjrt_client",

        "@xla//xla/service/cpu:cpu_compiler",
        "@xla//xla/stream_executor/tpu:tpu_on_demand_compiler",
        "@xla//xla/stream_executor/tpu:tpu_executor",
        "@xla//xla/stream_executor/tpu:tpu_transfer_manager",
        
        "@xla//xla/service/cpu:cpu_transfer_manager",
        "@xla//xla/tsl/framework:allocator_registry_impl",

        "@xla//xla/pjrt:status_casters",
        "@xla//xla/python/ifrt:ifrt",
        "@xla//xla/python/pjrt_ifrt:xla_ifrt",
        "@xla//xla/python/ifrt/hlo:hlo_program",
        "@xla//xla/ffi:call_frame",
        "@com_google_protobuf//:protobuf",
        "@tsl//tsl/profiler/backends/cpu:annotation_stack_impl",
        "@tsl//tsl/profiler/backends/cpu:traceme_recorder_impl",
        "@tsl//tsl/profiler/utils:time_utils_impl",
        "@tsl//tsl/platform:env_impl",
        "@xla//xla/stream_executor:stream_executor_impl",
        "@xla//xla/mlir/utils:type_util",
        "@stablehlo//:stablehlo_capi_objects",
        "@stablehlo//:chlo_capi_objects",
        "@com_google_absl//absl/hash:hash",
        "@com_google_absl//absl/log:initialize",
        "@llvm-project//mlir:CAPIIRObjects",  

        "@xla//xla/translate/stablehlo_to_hlo:translate",
        "@xla//xla/translate/mhlo_to_hlo:mlir_hlo_to_hlo",

        "@xla//xla/service/gpu/model:gpu_performance_model",
        "@xla//xla/service/gpu/model:gpu_hlo_cost_analysis",
        "@xla//xla/service/gpu:gpu_latency_hiding_scheduler",
        "@xla//xla/stream_executor:device_description",
    ] + select({
    "@bazel_tools//src/conditions:darwin": [],
    "//conditions:default": [
        "@xla//xla/stream_executor/cuda:all_runtime", 
        "@xla//xla/stream_executor/rocm:all_runtime",
        "@xla//xla/service/gpu/model:hlo_op_profiles",
        "@xla//xla/service/gpu/model:hlo_op_profile_proto_cc_impl",
        "@xla//xla/service/gpu:nvptx_compiler",
        # "@xla//xla/service/gpu:amdgpu_compiler",
        "@xla//xla/service/gpu:gpu_transfer_manager",
        "@xla//xla/stream_executor:kernel", 
    ]}),
    alwayslink = 1,
    linkstatic = 1
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
        "@tsl//tsl/profiler/backends/cpu:annotation_stack_impl",
        "@tsl//tsl/profiler/backends/cpu:traceme_recorder_impl",
        "@tsl//tsl/profiler/utils:time_utils_impl",
        "@tsl//tsl/protobuf:dnn_proto_cc_impl",
        "@tsl//tsl/protobuf:histogram_proto_cc",
        "@tsl//tsl/protobuf:histogram_proto_cc_impl",
        "@tsl//tsl/protobuf:protos_all_cc_impl",
 
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
        "@xla//xla:statusor",
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
    alwayslink = True
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
    linkstatic = 1
)

load("@tensat_crate_index//:defs.bzl", "aliases", "all_crate_deps")

rust_library(
    name = "tensat_lib",
    crate_name = "tensat",
    visibility = ["//visibility:public"],
    srcs = glob(["deps/tensat/src/*.rs"]),
    deps = all_crate_deps(
        normal = True,
    ) + [":tensat_bridge"],
    proc_macro_deps = all_crate_deps(
        proc_macro = True,
    ),
    edition = "2021",
    alwayslink = True,
)

rust_binary(
    name = "tensat",
    srcs = glob(["deps/tensat/src/main.rs"]),
    visibility = ["//visibility:public"],
    deps = all_crate_deps(
        normal = True,
    ) + [":tensat_lib"],
    proc_macro_deps = all_crate_deps(
        proc_macro = True,
    ),
    edition = "2021",
)

cc_library(
    name = "equality_saturation",
    hdrs = ["Passes/EqualitySaturation.h"],
    deps = [
        ":cxx_core",
        "@llvm-project//mlir:IR",
    ],
    strip_include_prefix = "Passes",
)

rust_cxx_bridge(
    name = "tensat_bridge",
    src = "deps/tensat/src/input.rs",
    deps = [
        ":cxx_core",
        ":equality_saturation",
    ],
)

genrule(
    name = "cxx_header",
    outs = ["cxx.h"], 
    cmd = "cxxbridge --header -o $(location cxx.h)",
)

cc_library(
    name = "cxx_core",
    hdrs = ["cxx.h"],
    include_prefix = "rust",
)

cc_binary(
    name = "graphtest",
    visibility = ["//visibility:public"],
    srcs = ["deps/tensat/src/graph.cc"],
    deps = [
        ":tensat_lib",
    ]
)   
