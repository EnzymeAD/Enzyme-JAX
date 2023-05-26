
load("@tsl//tsl:tsl.bzl", _if_windows = "if_windows", pybind_extension = "tsl_pybind_extension_opensource")

py_library(
    name = "enzyme_jax_internal",
    srcs = ["primitives.py", "__init__.py"],
    visibility = ["//visibility:public"]
)

pybind_extension(
    name = "enzyme_call",
    srcs = ["enzyme_call.cc"],
    deps = [
        "@pybind11",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:OrcJIT",
        "//:clang_compile",
    ],
    visibility = ["//visibility:public"]
)