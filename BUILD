load("@rules_python//python:packaging.bzl", "py_wheel")
load(":package.bzl", "py_package")

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//:__subpackages__"],
)

py_package(
    name = "enzyme_jax_data",
    # Only include these Python packages.
    packages = [
        "@//src/enzyme_ad/jax:enzyme_call.so",
        "@llvm-project//clang:builtin_headers_gen",
    ],
    deps = [
        "//src/enzyme_ad/jax:enzyme_call.so",
        "@llvm-project//clang:builtin_headers_gen",
    ],
)

cc_binary(
    name = "enzymexlamlir-opt",
    srcs = ["//src/enzyme_ad/jax:enzymexlamlir-opt.cpp"],
    visibility = ["//visibility:public"],
    deps = [
        "@enzyme//:EnzymeMLIR",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:AsyncDialect",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:ConversionPasses",
        "@llvm-project//mlir:DLTIDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:OpenMPDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:TransformDialect",
        "@llvm-project//mlir:Transforms",
        "//src/enzyme_ad/jax:TransformOps",
        "//src/enzyme_ad/jax:XLADerivatives",
        "@stablehlo//:chlo_ops",
    ],
)

cc_binary(
    name = "enzymexlamlir-tblgen",
    srcs = ["//src/enzyme_ad/tools:enzymexlamlir-tblgen.cpp"],
    visibility = ["//visibility:public"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TableGen",
        "@llvm-project//llvm:config",
    ],
)

py_wheel(
    name = "enzyme_ad",
    author = "Enzyme Authors",
    author_email = "wmoses@mit.edu, zinenko@google.com",
    distribution = "enzyme_ad",
    homepage = "https://enzyme.mit.edu/",
    license = "LLVM",
    platform = select({
        "@bazel_tools//src/conditions:windows_x64": "win_amd64",
        "@bazel_tools//src/conditions:darwin_arm64": "macosx_11_0_arm64",
        "@bazel_tools//src/conditions:darwin_x86_64": "macosx_10_14_x86_64",
        "@bazel_tools//src/conditions:linux_aarch64": "manylinux2014_aarch64",
        "@bazel_tools//src/conditions:linux_x86_64": "manylinux2014_x86_64",
        "@bazel_tools//src/conditions:linux_ppc64le": "manylinux2014_ppc64le",
    }),
    project_urls = {
        "GitHub": "https://github.com/EnzymeAD/Enzyme-JAX/",
    },
    python_tag = "py3",
    requires = [
        "absl_py >= 2.0.0",
        "jax >= 0.4.21",
        "jaxlib >= 0.4.21",
    ],
    strip_path_prefixes = ["src/"],
    summary = "Enzyme automatic differentiation tool.",
    version = "0.0.6",
    deps = [
        ":enzyme_jax_data",
        "//src/enzyme_ad/jax:enzyme_jax_internal",
    ],
)
