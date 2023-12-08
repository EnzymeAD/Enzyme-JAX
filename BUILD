load("@rules_python//python:packaging.bzl", "py_wheel")
load(":package.bzl", "py_package")

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//:__subpackages__"],
)

py_package(
    name = "enzyme_jax_data",
    deps = [
        "//src/enzyme_ad/jax:enzyme_call.so",
        "@llvm-project//clang:builtin_headers_gen",
    ],
    # Only include these Python packages.
    packages = ["@//src/enzyme_ad/jax:enzyme_call.so", "@llvm-project//clang:builtin_headers_gen"],
)

py_wheel(
    name = "enzyme_ad",
    distribution = "enzyme_ad",
    summary = "Enzyme automatic differentiation tool.",
    homepage = "https://enzyme.mit.edu/",
    project_urls = {
        "GitHub": "https://github.com/EnzymeAD/Enzyme-JAX/",
    },
    author="Enzyme Authors",
    license="LLVM",
    author_email="wmoses@mit.edu, zinenko@google.com",
    python_tag = "py3",
    version = "0.0.6",
    platform = select({
        "@bazel_tools//src/conditions:windows_x64": "win_amd64",
        "@bazel_tools//src/conditions:darwin_arm64": "macosx_11_0_arm64",
        "@bazel_tools//src/conditions:darwin_x86_64": "macosx_10_14_x86_64",
        "@bazel_tools//src/conditions:linux_aarch64": "manylinux2014_aarch64",
        "@bazel_tools//src/conditions:linux_x86_64": "manylinux2014_x86_64",
        "@bazel_tools//src/conditions:linux_ppc64le": "manylinux2014_ppc64le",
    }),
    deps = ["//src/enzyme_ad/jax:enzyme_jax_internal", ":enzyme_jax_data"],
    strip_path_prefixes = ["src/"],
    requires = [
        "jax >= 0.4.21",
        "jaxlib >= 0.4.21",
    ],
)
