load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

td_library(
    name = "MPSDialectTdFiles",
    srcs = [
        "Dialect.td",
        "Enums.td",
        "Ops.td",
    ],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "MPSDialectIncGen",
    tbl_outs = {
        "MPSDialect.h.inc": [
            "-gen-dialect-decls",
            "-dialect=mps",
        ],
        "MPSDialect.cpp.inc": [
            "-gen-dialect-defs",
            "-dialect=mps",
        ],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Dialect.td",
    deps = [":MPSDialectTdFiles"],
)

gentbl_cc_library(
    name = "MPSEnumsIncGen",
    tbl_outs = {
        "MPSEnums.h.inc": ["-gen-enum-decls"],
        "MPSEnums.cpp.inc": ["-gen-enum-defs"],
        "MPSAttrs.h.inc": [
            "-gen-attrdef-decls",
            "-attrdefs-dialect=mps",
        ],
        "MPSAttrs.cpp.inc": [
            "-gen-attrdef-defs",
            "-attrdefs-dialect=mps",
        ],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Enums.td",
    deps = [":MPSDialectTdFiles"],
)

gentbl_cc_library(
    name = "MPSOpsIncGen",
    tbl_outs = {
        "MPSOps.h.inc": ["-gen-op-decls"],
        "MPSOps.cpp.inc": ["-gen-op-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Ops.td",
    deps = [":MPSDialectTdFiles"],
)

cc_library(
    name = "MPSDialect",
    srcs = [
        "Dialect.cpp",
        "Ops.cpp",
    ],
    hdrs = [
        "Dialect.h",
        "Ops.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":MPSDialectIncGen",
        ":MPSEnumsIncGen",
        ":MPSOpsIncGen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BytecodeOpInterface",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:SideEffectInterfaces",
    ],
)
