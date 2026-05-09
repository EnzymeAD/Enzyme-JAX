load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

td_library(
    name = "CommDialectTdFiles",
    srcs = [
        "Dialect.td",

        # MPI
        "MPITypes.td",
    ],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
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
    name = "CommMPITypesIncGen",
    tbl_outs = {
        "MPITypes.h.inc": [
            "-gen-typedef-decls",
        ],
        "MPITypes.cpp.inc": [
            "-gen-typedef-defs",
        ],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "MPITypes.td",
    deps = [
        ":CommDialectTdFiles",
    ],
)

cc_library(
    name = "CommDialect",
    srcs = [
        "Dialect.cpp",
    ],
    hdrs = [
        "Dialect.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":CommDialectIncGen",
        ":CommMPITypesIncGen",
        "@llvm-project//mlir:IR",
    ],
)
