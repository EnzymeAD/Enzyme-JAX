
licenses(["notice"])

load("@tsl//tsl:tsl.bzl", _if_windows = "if_windows", pybind_extension = "tsl_pybind_extension_opensource")

package(
    default_applicable_licenses = [],
    default_visibility = ["//:__subpackages__"],
)

cc_library(
    name = "clang_compile",
    srcs = ["clang_compile.cc"],
    hdrs = ["clang_compile.h"],
    data = ["@llvm-project//clang:builtin_headers_gen"],
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
        "@enzyme//:EnzymeStatic"
    ],
)

pybind_extension(
    name = "enzyme_call",
    srcs = ["enzyme_call.cc"],
    deps = [
        "@pybind11",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:OrcJIT",
        ":clang_compile",
    ],
)

load("@rules_python//python:packaging.bzl", "py_wheel")

py_library(
    name = "enzyme_jax_internal",
    srcs = ["enzyme_jax/primitives.py", "enzyme_jax/__init__.py"],
    data = [':enzyme_call'],
    deps = [
        ":enzyme_call",
    ],
    visibility = ["//visibility:public"]
)

py_wheel(
    name = "enzyme_jax",
    # Package data. We're building "example_minimal_package-0.0.1-py3-none-any.whl"
    distribution = "enzyme_jax",
    python_tag = "py3",
    version = "0.0.1",
    deps = [":enzyme_jax_internal"],
)