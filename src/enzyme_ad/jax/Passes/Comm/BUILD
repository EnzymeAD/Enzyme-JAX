load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

td_library(
    name = "PassesTdFiles",
    srcs = [
        "Passes.td",
    ],
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "PassesIncGen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=comm",
            ],
            "Passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Passes.td",
    deps = [
        ":PassesTdFiles",
    ],
)

cc_library(
    name = "Passes",
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    visibility = ["//visibility:public"],
    deps = [
        ":PassesIncGen",
        "//src/enzyme_ad/jax/Dialect/Comm:CommDialect",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@stablehlo//:stablehlo_ops",
    ],
)
