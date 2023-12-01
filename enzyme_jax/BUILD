load("@pybind11_bazel//:build_defs.bzl", "pybind_extension", "pybind_library")

licenses(["notice"])

package(
    default_visibility = ["//:__subpackages__"],
)

cc_library(
    name = "clang_compile",
    srcs = ["clang_compile.cc"],
    hdrs = ["clang_compile.h"],
    deps = [
        "@pybind11",
        "@llvm-project//clang:ast",
        "@llvm-project//clang:basic",
        "@llvm-project//clang:driver",
        "@llvm-project//clang:frontend",
        "@llvm-project//clang:frontend_tool",
        "@llvm-project//clang:lex",
        "@llvm-project//clang:serialization",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:OrcJIT",
        "@enzyme//:EnzymeStatic"
    ],
)

py_library(
    name = "enzyme_jax_internal",
    srcs = ["primitives.py", "__init__.py"],
    visibility = ["//visibility:public"]
)

pybind_library(
    name = "compile_with_xla",
    srcs = ["compile_with_xla.cc"],
    deps = [
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
        "@xla//xla:autotuning_proto_cc",
        "@xla//xla:autotuning_proto_cc_impl",
        "@xla//xla:autotune_results_proto_cc",
        "@xla//xla:autotune_results_proto_cc_impl",
        "@xla//xla/client",
        "@xla//xla/client:client_library",
        "@xla//xla/service/cpu:cpu_executable",
        "@xla//xla/service/gpu:backend_configs_cc",
        "@xla//xla/service/gpu:backend_configs_cc_impl",
        "@xla//xla/service:buffer_assignment_proto_cc",
        "@xla//xla/service:buffer_assignment_proto_cc_impl",
        "@xla//xla/service:hlo_proto_cc",
        "@xla//xla/service:hlo_proto_cc_impl",
        "@xla//xla/service/memory_space_assignment:memory_space_assignment_proto_cc_impl",
        "@xla//xla/stream_executor:stream_executor_impl",
        "@xla//xla/stream_executor:device_description_proto_cc",
        "@xla//xla/stream_executor:device_description_proto_cc_impl",
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
    ],
)

pybind_extension(
    name = "enzyme_call",
    srcs = ["enzyme_call.cc"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:OrcJIT",
        ":clang_compile",
        ":compile_with_xla",
        "@com_google_absl//absl/status"
    ],
    visibility = ["//visibility:public"],
)
