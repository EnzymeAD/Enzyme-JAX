
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

load("@rules_python//python:packaging.bzl", "py_wheel")

load(":package.bzl", "py_package")

py_package(
    name = "enzyme_jax_data",
    # Only include these Python packages.
    # packages = ["examples.wheel"],
    deps = ["//enzyme_jax:enzyme_call", "@llvm-project//clang:builtin_headers_gen"],
    prefix = "enzyme_jax/",
)

py_wheel(
    name = "enzyme_jax",
    # Package data. We're building "example_minimal_package-0.0.1-py3-none-any.whl"
    distribution = "enzyme_jax",
    python_tag = "py3",
    version = "0.0.1",
    deps = ["//enzyme_jax:enzyme_jax_internal", ":enzyme_jax_data"],
)