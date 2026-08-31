load("@rules_foreign_cc//foreign_cc:defs.bzl", "make")

package(
    default_visibility = ["//visibility:public"],
)

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
    out_include_dir = "include",
    out_shared_libs = [
        "libblastrampoline.so",
        "libblastrampoline.so.5",
        "libblastrampoline.so.5.15.0",
    ],
)
