
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
    packages = ["@//enzyme_jax:enzyme_call.so", "@llvm-project//clang:builtin_headers_gen"],
    deps = ["//enzyme_jax:enzyme_call", "@llvm-project//clang:builtin_headers_gen"],
    prefix = "enzyme_jax/",
)

py_wheel(
    name = "enzyme_jax",
    # Package data. We're building "example_minimal_package-0.0.1-py3-none-any.whl"
    distribution = "enzyme_jax",
    author="Enzyme Authors",
    license='LLVM',
    author_email="wmoses@mit.edu, zinenko@google.com",
    python_tag = "py3",
    version = "0.0.3",
    platform = select({
        "@bazel_tools//src/conditions:windows_x64": "win_amd64",
        "@bazel_tools//src/conditions:darwin_arm64": "macosx_11_0_arm64",
        "@bazel_tools//src/conditions:darwin_x86_64": "macosx_10_14_x86_64",
        "@bazel_tools//src/conditions:linux_aarch64": "manylinux2014_aarch64",
        "@bazel_tools//src/conditions:linux_x86_64": "manylinux2014_x86_64",
        "@bazel_tools//src/conditions:linux_ppc64le": "manylinux2014_ppc64le",
    }),
    deps = ["//enzyme_jax:enzyme_jax_internal", ":enzyme_jax_data"]
)
