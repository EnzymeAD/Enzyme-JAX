load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
)

td_library(
    name = "BlasDialectTdFiles",
    srcs = [
        "Attrs.td",
        "Dialect.td",
        "Ops.td",
    ],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "BlasDialectIncGen",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
                "-dialect=blas",
            ],
            "BlasDialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect=blas",
            ],
            "BlasDialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Dialect.td",
    deps = [
        ":BlasDialectTdFiles",
    ],
)

gentbl_cc_library(
    name = "BlasAttrsIncGen",
    tbl_outs = [
        (
            [
                "--gen-enum-decls",
                "--attrdefs-dialect=blas",
            ],
            "BlasEnums.h.inc",
        ),
        (
            [
                "--gen-enum-defs",
                "--attrdefs-dialect=blas",
            ],
            "BlasEnums.cpp.inc",
        ),
        (
            ["--gen-attrdef-decls"],
            "BlasAttrs.h.inc",
        ),
        (
            ["--gen-attrdef-defs"],
            "BlasAttrs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Attrs.td",
    deps = [
        ":BlasDialectTdFiles",
    ],
)

gentbl_cc_library(
    name = "BlasOpsIncGen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "BlasOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "BlasOps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Ops.td",
    deps = [
        ":BlasDialectTdFiles",
        "@stablehlo//:stablehlo_ops_td_files",
    ],
)

cc_library(
    name = "BlasDialectUtils",
    srcs = ["Utils.cpp"],
    hdrs = ["Utils.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "BlasDialect",
    srcs = [
        "Dialect.cpp",
        "Ops.cpp",
    ],
    hdrs = [
        "Dialect.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":BlasAttrsIncGen",
        ":BlasDialectIncGen",
        ":BlasDialectUtils",
        ":BlasOpsIncGen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BytecodeOpInterface",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@stablehlo//:base",
    ],
)
