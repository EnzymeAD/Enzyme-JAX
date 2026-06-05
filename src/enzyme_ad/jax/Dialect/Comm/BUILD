load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

td_library(
    name = "CommDialectTdFiles",
    srcs = [
        "Dialect.td",
        "Types.td",

        # MPI
        "MPIAttrs.td",
        "MPIOps.td",
    ],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "CommDialectIncGen",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
                "-dialect=comm",
            ],
            "CommDialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect=comm",
            ],
            "CommDialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Dialect.td",
    deps = [
        ":CommDialectTdFiles",
    ],
)

gentbl_cc_library(
    name = "CommTypesIncGen",
    tbl_outs = {
        "Types.h.inc": [
            "-gen-typedef-decls",
        ],
        "Types.cpp.inc": [
            "-gen-typedef-defs",
        ],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Types.td",
    deps = [
        ":CommDialectTdFiles",
    ],
)

gentbl_cc_library(
    name = "CommMPIAttrsIncGen",
    tbl_outs = {
        "MPIAttrEnums.h.inc": [
            "-gen-enum-decls",
            "-attrdefs-dialect=comm",
        ],
        "MPIAttrEnums.cpp.inc": [
            "-gen-enum-defs",
            "-attrdefs-dialect=comm",
        ],
        "MPIAttrDefs.h.inc": [
            "-gen-attrdef-decls",
            "-attrdefs-dialect=comm",
        ],
        "MPIAttrDefs.cpp.inc": [
            "-gen-attrdef-defs",
            "-attrdefs-dialect=comm",
        ],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "MPIAttrs.td",
    deps = [
        ":CommDialectTdFiles",
    ],
)

gentbl_cc_library(
    name = "CommMPIOpsIncGen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "MPIOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "MPIOps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "MPIOps.td",
    deps = [
        ":CommDialectTdFiles",
    ],
)

cc_library(
    name = "CommDialect",
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
        ":CommDialectIncGen",
        ":CommTypesIncGen",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SideEffectInterfaces",

        # MPI
        ":CommMPIAttrsIncGen",
        ":CommMPIOpsIncGen",
    ],
)
