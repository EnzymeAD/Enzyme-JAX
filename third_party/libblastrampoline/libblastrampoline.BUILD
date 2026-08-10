load("@rules_foreign_cc//foreign_cc:defs.bzl", "make")

exports_files(["LICENSE.md"])

filegroup(
    name = "srcs",
    srcs = glob(["src/**"]) + glob(["include/**"]),
    visibility = ["//:__subpackages__"],
)

make(
    name = "libblastrampoline",
    env = {},  # TODO forward env vars from BB for cross-compiling
    lib_source = "//:srcs",
    out_shared_libs = ["libblastrampoline.so.5"],
    targets = [
        "all",
        "install",
    ],
)
