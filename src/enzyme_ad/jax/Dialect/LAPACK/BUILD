load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
)

td_library(
    name = "LapackDialectTdFiles",
    srcs = [
        "Dialect.td",
        "Ops.td",
    ],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "LapackDialectIncGen",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
                "-dialect=lapack",
            ],
            "LapackDialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect=lapack",
            ],
            "LapackDialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Dialect.td",
    deps = [
        ":LapackDialectTdFiles",
    ],
)

gentbl_cc_library(
    name = "LapackOpsIncGen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "LapackOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "LapackOps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Ops.td",
    deps = [
        ":LapackDialectTdFiles",
        "//src/enzyme_ad/jax/Dialect/BLAS:BlasDialectTdFiles",
        "@stablehlo//:stablehlo_ops_td_files",
    ],
)

cc_library(
    name = "LapackDialect",
    srcs = [
        "Dialect.cpp",
        "Ops.cpp",
    ],
    hdrs = [
        "Dialect.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":LapackDialectIncGen",
        ":LapackOpsIncGen",
        "//src/enzyme_ad/jax/Dialect/BLAS:BlasDialect",
        "//src/enzyme_ad/jax/Dialect/BLAS:BlasDialectUtils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BytecodeOpInterface",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:QuantOps",
        "@stablehlo//:base",
    ],
)
