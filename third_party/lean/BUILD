load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

cmake(
    name = "lean4",
    build_args = ["-j128"],
    generate_crosstool_file = False,
    lib_source = "@lean//:lean_src",
    out_binaries = select({
        "//conditions:default": ["lean"],
    }),
    tags = ["requires-network"],
    visibility = ["//visibility:public"],
)
