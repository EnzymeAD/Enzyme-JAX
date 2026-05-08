load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

td_library(
    name = "CommDialectTdFiles",
    srcs = [
        "Dialect.td",
    ],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
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
        "@llvm-project//mlir:IR",
    ],
)
