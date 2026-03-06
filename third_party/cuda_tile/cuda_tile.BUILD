# BUILD file for NVIDIA CUDA Tile IR
# This provides Bazel targets for the CUDA Tile MLIR dialect

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0 with LLVM exception

# TableGen files for the CudaTile dialect
td_library(
    name = "CudaTileDialectTdFiles",
    srcs = [
        "include/cuda_tile/Dialect/CudaTile/IR/AttrDefs.td",
        "include/cuda_tile/Dialect/CudaTile/IR/Dialect.td",
        "include/cuda_tile/Dialect/CudaTile/IR/Interfaces.td",
        "include/cuda_tile/Dialect/CudaTile/IR/Ops.td",
        "include/cuda_tile/Dialect/CudaTile/IR/Types.td",
    ],
    includes = ["include"],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:CallInterfacesTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:FunctionInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "CudaTileDialectIncGen",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
                "-dialect=cuda_tile",
            ],
            "include/cuda_tile/Dialect/CudaTile/IR/Dialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect=cuda_tile",
            ],
            "include/cuda_tile/Dialect/CudaTile/IR/Dialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/cuda_tile/Dialect/CudaTile/IR/Dialect.td",
    deps = [":CudaTileDialectTdFiles"],
)

gentbl_cc_library(
    name = "CudaTileOpsIncGen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/cuda_tile/Dialect/CudaTile/IR/Ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/cuda_tile/Dialect/CudaTile/IR/Ops.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/cuda_tile/Dialect/CudaTile/IR/Ops.td",
    deps = [":CudaTileDialectTdFiles"],
)

gentbl_cc_library(
    name = "CudaTileTypesIncGen",
    tbl_outs = [
        (
            [
                "-gen-typedef-decls",
                "-typedefs-dialect=cuda_tile",
            ],
            "include/cuda_tile/Dialect/CudaTile/IR/Types.h.inc",
        ),
        (
            [
                "-gen-typedef-defs",
                "-typedefs-dialect=cuda_tile",
            ],
            "include/cuda_tile/Dialect/CudaTile/IR/Types.cpp.inc",
        ),
        (
            [
                "-gen-type-constraint-decls",
                "-dialect=cuda_tile",
            ],
            "include/cuda_tile/Dialect/CudaTile/IR/TypeConstraints.h.inc",
        ),
        (
            [
                "-gen-type-constraint-defs",
                "-dialect=cuda_tile",
            ],
            "include/cuda_tile/Dialect/CudaTile/IR/TypeConstraints.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/cuda_tile/Dialect/CudaTile/IR/Types.td",
    deps = [":CudaTileDialectTdFiles"],
)

gentbl_cc_library(
    name = "CudaTileAttrsIncGen",
    tbl_outs = [
        (
            [
                "-gen-attrdef-decls",
                "-attrdefs-dialect=cuda_tile",
            ],
            "include/cuda_tile/Dialect/CudaTile/IR/AttrDefs.h.inc",
        ),
        (
            [
                "-gen-attrdef-defs",
                "-attrdefs-dialect=cuda_tile",
            ],
            "include/cuda_tile/Dialect/CudaTile/IR/AttrDefs.cpp.inc",
        ),
        (
            [
                "-gen-enum-decls",
                "-dialect=cuda_tile",
            ],
            "include/cuda_tile/Dialect/CudaTile/IR/Enums.h.inc",
        ),
        (
            [
                "-gen-enum-defs",
                "-dialect=cuda_tile",
            ],
            "include/cuda_tile/Dialect/CudaTile/IR/Enums.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/cuda_tile/Dialect/CudaTile/IR/AttrDefs.td",
    deps = [":CudaTileDialectTdFiles"],
)

gentbl_cc_library(
    name = "CudaTileInterfacesIncGen",
    tbl_outs = [
        (
            ["-gen-attr-interface-decls"],
            "include/cuda_tile/Dialect/CudaTile/IR/AttrInterfaces.h.inc",
        ),
        (
            ["-gen-attr-interface-defs"],
            "include/cuda_tile/Dialect/CudaTile/IR/AttrInterfaces.cpp.inc",
        ),
        (
            ["-gen-type-interface-decls"],
            "include/cuda_tile/Dialect/CudaTile/IR/TypeInterfaces.h.inc",
        ),
        (
            ["-gen-type-interface-defs"],
            "include/cuda_tile/Dialect/CudaTile/IR/TypeInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/cuda_tile/Dialect/CudaTile/IR/Interfaces.td",
    deps = [":CudaTileDialectTdFiles"],
)

# CudaTile Dialect library
cc_library(
    name = "CudaTileDialect",
    srcs = [
        "lib/Dialect/CudaTile/IR/Attributes.cpp",
        "lib/Dialect/CudaTile/IR/CudaTile.cpp",
        "lib/Dialect/CudaTile/IR/Interfaces.cpp",
        "lib/Dialect/CudaTile/IR/Traits.cpp",
        "lib/Dialect/CudaTile/IR/Types.cpp",
    ],
    hdrs = [
        "include/cuda_tile/Dialect/CudaTile/IR/Attributes.h",
        "include/cuda_tile/Dialect/CudaTile/IR/Dialect.h",
        "include/cuda_tile/Dialect/CudaTile/IR/Interfaces.h",
        "include/cuda_tile/Dialect/CudaTile/IR/Ops.h",
        "include/cuda_tile/Dialect/CudaTile/IR/SharedFuncParserAndPrinter.h",
        "include/cuda_tile/Dialect/CudaTile/IR/SharedVerifiers.h",
        "include/cuda_tile/Dialect/CudaTile/IR/Traits.h",
        "include/cuda_tile/Dialect/CudaTile/IR/Types.h",
    ],
    includes = ["include"],
    deps = [
        ":CudaTileAttrsIncGen",
        ":CudaTileDialectIncGen",
        ":CudaTileInterfacesIncGen",
        ":CudaTileOpsIncGen",
        ":CudaTileTypesIncGen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:CallOpInterfaces",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:InliningUtils",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
    ],
)

# TableGen files for CudaTile transforms/passes
td_library(
    name = "CudaTileTransformsTdFiles",
    srcs = [
        "include/cuda_tile/Dialect/CudaTile/Transforms/Passes.td",
    ],
    includes = ["include"],
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "CudaTileTransformsIncGen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=CudaTile",
            ],
            "include/cuda_tile/Dialect/CudaTile/Transforms/Passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/cuda_tile/Dialect/CudaTile/Transforms/Passes.td",
    deps = [":CudaTileTransformsTdFiles"],
)

# CudaTile Transforms library
cc_library(
    name = "CudaTileTransforms",
    srcs = [
        "lib/Dialect/CudaTile/Transforms/FuseFMA.cpp",
        "lib/Dialect/CudaTile/Transforms/LoopSplit.cpp",
        "lib/Dialect/CudaTile/Transforms/SynthesizeDebugInfoScopes.cpp",
    ],
    hdrs = [
        "include/cuda_tile/Dialect/CudaTile/Transforms/Passes.h",
    ],
    includes = ["include"],
    deps = [
        ":CudaTileDialect",
        ":CudaTileTransformsIncGen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)
