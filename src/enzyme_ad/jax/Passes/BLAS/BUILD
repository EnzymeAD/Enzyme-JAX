load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

td_library(
    name = "BlasPassesTdFiles",
    srcs = [
        "Passes.td",
    ],
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "BlasPassesIncGen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=blass",
            ],
            "Passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Passes.td",
    deps = [":BlasPassesTdFiles"],
)

cc_library(
    name = "BlasPasses",
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    visibility = ["//visibility:public"],
    deps = [
        ":BlasPassesIncGen",
        "//src/enzyme_ad/jax:EnzymeXLADialect",
        "//src/enzyme_ad/jax/Dialect/BLAS:BlasDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:Pass",
        "@stablehlo//:stablehlo_ops",
        # "@stablehlo//:reference_ops",
        # "@stablehlo//:stablehlo_passes",
        # "@xla//xla/mlir_hlo:mlir_hlo",
    ],
)
