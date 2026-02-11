load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")


cmake(
    name = "lean4",
    lib_source = "@lean//:lean_src",
    out_binaries = select({
        "//conditions:default": ["lean"],
    }),
    build_args = ["-j128"],
    tags = ["requires-network"],
    visibility = ["//visibility:public"],
    generate_crosstool_file = False,
)
