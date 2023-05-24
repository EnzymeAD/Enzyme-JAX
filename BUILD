
licenses(["notice"])

load("@tsl//tsl:tsl.bzl", _if_windows = "if_windows", pybind_extension = "tsl_pybind_extension_opensource")

package(
    default_applicable_licenses = [],
    default_visibility = ["//:__subpackages__"],
)

pybind_extension(
    name = "enzyme_call",
    srcs = ["enzyme_call.cc"],
    deps = [
        "@pybind11",
        "@llvm-project//llvm:Support",
    ],
)

