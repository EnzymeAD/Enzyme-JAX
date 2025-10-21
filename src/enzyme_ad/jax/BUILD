load("@jax//jaxlib:symlink_files.bzl", "symlink_inputs")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_python//python:py_library.bzl", "py_library")
load("@xla//xla/tsl:tsl.bzl", pybind_extension = "tsl_pybind_extension_opensource")

package(
    default_visibility = ["//:__subpackages__"],
    features = ["layering_check"],
)

exports_files([
    "enzymexlamlir-opt.cpp",
    "raise.cpp",
    "RegistryUtils.cpp",
])

licenses(["notice"])

config_setting(
    name = "is_clang_build",
    define_values = {"using_clang": "true"},
)

cc_library(
    name = "gpu",
    srcs = ["gpu.cc"],
    copts = ["-Wno-vla-cxx-extension"],
    deps = [
        "@xla//xla/ffi:ffi_api",
        "@xla//xla/ffi/api:ffi",
    ],
)

cc_library(
    name = "cpu",
    srcs = ["cpu.cc"],
    deps = [
        "@xla//xla/service:custom_call_status",
        "@xla//xla/service:custom_call_target_registry",
    ],
)

cc_library(
    name = "clang_compile",
    srcs = ["clang_compile.cc"],
    hdrs = ["clang_compile.h"],
    copts = [
        "-Wno-implicit-fallthrough",
        "-Wno-non-virtual-dtor",
        "-Wno-pessimizing-move",
        "-Wno-unused-result",
        "-Wno-unused-variable",
    ],
    deps = [
        "@enzyme//:EnzymeStatic",
        "@llvm-project//clang:ast",
        "@llvm-project//clang:basic",
        "@llvm-project//clang:codegen",
        "@llvm-project//clang:driver",
        "@llvm-project//clang:frontend",
        "@llvm-project//clang:frontend_tool",
        "@llvm-project//clang:lex",
        "@llvm-project//clang:parse",
        "@llvm-project//clang:sema",
        "@llvm-project//clang:serialization",
        "@llvm-project//llvm:AsmParser",
        "@llvm-project//llvm:CodeGen",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:Linker",
        "@llvm-project//llvm:MC",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
        "@nanobind",
    ],
)

py_library(
    name = "enzyme_jax_internal",
    srcs = [
        "__init__.py",
        "primitives.py",
        "utils.py",
    ],
    visibility = ["//visibility:public"],
)

symlink_inputs(
    name = "EnzymeImplementationsCommonTdFiles",
    rule = td_library,
    symlinked_inputs = {"srcs": {
        "Implementations": ["@enzyme//:ImplementationsCommonTdFiles"],
    }},
)

td_library(
    name = "TransformOpsTdFiles",
    srcs = [
        "TransformOps/TransformOps.td",
    ],
    deps = [
        "@llvm-project//mlir:TransformDialectTdFiles",
    ],
)

gentbl_cc_library(
    name = "TransformOpsIncGen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "TransformOps/TransformOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "TransformOps/TransformOps.cpp.inc",
        ),
        (
            ["-gen-op-interface-decls"],
            "TransformOps/OpInterfaces.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "TransformOps/OpInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "TransformOps/TransformOps.td",
    deps = [
        ":TransformOpsTdFiles",
    ],
)

gentbl_cc_library(
    name = "TransformOpsImplIncGen",
    tbl_outs = [(
        ["-gen-populate-patterns-interface-impl"],
        "TransformOps/TransformOpsImpl.cpp.inc",
    )],
    tblgen = "//:enzymexlamlir-tblgen",
    td_file = "TransformOps/TransformOps.td",
    deps = [
        ":TransformOpsTdFiles",
    ],
)

td_library(
    name = "RaisingTransformOpsTdFiles",
    srcs = [
        "TransformOps/RaisingTransformOps.td",
    ],
    deps = [
        "@llvm-project//mlir:TransformDialectTdFiles",
    ],
)

gentbl_cc_library(
    name = "RaisingTransformOpsIncGen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "TransformOps/RaisingTransformOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "TransformOps/RaisingTransformOps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "TransformOps/RaisingTransformOps.td",
    deps = [
        ":RaisingTransformOpsTdFiles",
    ],
)

gentbl_cc_library(
    name = "RaisingTransformOpsImplIncGen",
    tbl_outs = [(
        ["-gen-populate-raising-patterns-interface-impl"],
        "TransformOps/RaisingTransformOpsImpl.cpp.inc",
    )],
    tblgen = "//:enzymexlamlir-tblgen",
    td_file = "TransformOps/RaisingTransformOps.td",
    deps = [
        ":RaisingTransformOpsTdFiles",
    ],
)

gentbl_cc_library(
    name = "RaisingTransformPatternsIncGen",
    tbl_outs = [
        (
            ["-gen-populate-raising-patterns-func-decls"],
            "TransformOps/RaisingTransformPatterns.h.inc",
        ),
        (
            ["-gen-populate-raising-patterns-func-defs"],
            "TransformOps/RaisingTransformPatterns.cpp.inc",
        ),
    ],
    tblgen = "//:enzymexlamlir-tblgen",
    td_file = "TransformOps/RaisingTransformOps.td",
    deps = [
        ":RaisingTransformOpsTdFiles",
    ],
)

cc_library(
    name = "RaisingTransformOps",
    srcs = ["TransformOps/RaisingTransformOps.cpp"],
    hdrs = ["TransformOps/RaisingTransformOps.h"],
    copts = select({
        "//conditions:default": [],
        ":is_clang_build": ["-fbracket-depth=768"],
    }),
    deps = [
        ":RaisingTransformOpsImplIncGen",
        ":RaisingTransformOpsIncGen",
        ":RaisingTransformPatternsIncGen",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LinalgTransformOps",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFUtils",
        "@llvm-project//mlir:TransformDialect",
        "@llvm-project//mlir:TransformDialectInterfaces",
    ],
)

cc_library(
    name = "TransformOps",
    srcs = [
        "TransformOps/GenerateApplyPatterns.cpp",
        "TransformOps/TransformOps.cpp",
    ],
    hdrs = ["TransformOps/TransformOps.h"],
    copts = select({
        "//conditions:default": [],
        ":is_clang_build": ["-fbracket-depth=768"],
    }),
    deps = [
        ":TransformOpsImplIncGen",
        ":TransformOpsIncGen",
        ":XLADerivatives",
        "@enzyme//:EnzymeMLIR",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgTransformOps",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TransformDialect",
        "@llvm-project//mlir:TransformDialectInterfaces",
    ],
)

td_library(
    name = "ImplementationsCommonTdFiles",
    srcs = [
        ":EnzymeImplementationsCommonTdFiles",
    ],
    deps = [
        ":EnzymeImplementationsCommonTdFiles",
    ],
)

gentbl_cc_library(
    name = "mhlo-derivatives",
    tbl_outs = [(
        ["-gen-mlir-derivatives"],
        "Implementations/MHLODerivatives.inc",
    )],
    tblgen = "@enzyme//:enzyme-tblgen",
    td_file = "Implementations/MHLODerivatives.td",
    td_srcs = [
        "Implementations/MHLODerivatives.td",
        "Implementations/HLODerivatives.td",
    ],
    deps = [
        ":ImplementationsCommonTdFiles",
    ],
)

gentbl_cc_library(
    name = "stablehlo-derivatives",
    tbl_outs = [(
        ["-gen-mlir-derivatives"],
        "Implementations/StableHLODerivatives.inc",
    )],
    tblgen = "@enzyme//:enzyme-tblgen",
    td_file = "Implementations/StableHLODerivatives.td",
    td_srcs = [
        "Implementations/StableHLODerivatives.td",
        "Implementations/HLODerivatives.td",
    ],
    deps = [
        ":EnzymeImplementationsCommonTdFiles",
    ],
)

gentbl_cc_library(
    name = "chlo-derivatives",
    tbl_outs = [(
        ["-gen-mlir-derivatives"],
        "Implementations/CHLODerivatives.inc",
    )],
    tblgen = "@enzyme//:enzyme-tblgen",
    td_file = "Implementations/CHLODerivatives.td",
    td_srcs = [
        "Implementations/CHLODerivatives.td",
    ],
    deps = [
        ":EnzymeImplementationsCommonTdFiles",
    ],
)

td_library(
    name = "EnzymeXLAPassesTdFiles",
    srcs = [
    ],
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "EnzymeXLAPassesIncGen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=enzymexla",
            ],
            "Passes/Passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Passes/Passes.td",
    deps = [":EnzymeXLAPassesTdFiles"],
)

td_library(
    name = "EnzymeXLADialectTdFiles",
    srcs = [
        "Dialect/Dialect.td",
        "Dialect/EnzymeXLAAttrs.td",
        "Dialect/EnzymeXLAOps.td",
    ],
    includes = ["."],
    deps = [
        "@enzyme//:EnzymeDialectTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:FunctionInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:LoopLikeInterfaceTdFiles",
        "@llvm-project//mlir:MemorySlotInterfacesTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
        "@llvm-project//mlir:ViewLikeInterfaceTdFiles",
    ],
)

gentbl_cc_library(
    name = "EnzymeXLAAttrsIncGen",
    tbl_outs = {
        "Dialect/EnzymeXLAAttrEnums.h.inc": [
            "--gen-enum-decls",
            "--attrdefs-dialect=enzymexla",
        ],
        "Dialect/EnzymeXLAAttrEnums.cpp.inc": [
            "--gen-enum-defs",
            "--attrdefs-dialect=enzymexla",
        ],
        "Dialect/EnzymeXLAAttrDefs.h.inc": [
            "--gen-attrdef-decls",
            "--attrdefs-dialect=enzymexla",
        ],
        "Dialect/EnzymeXLAAttrDefs.cpp.inc": [
            "--gen-attrdef-defs",
            "--attrdefs-dialect=enzymexla",
        ],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Dialect/EnzymeXLAAttrs.td",
    deps = [
        ":EnzymeXLADialectTdFiles",
    ],
)

gentbl_cc_library(
    name = "EnzymeXLAOpsIncGen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "Dialect/EnzymeXLAOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "Dialect/EnzymeXLAOps.cpp.inc",
        ),
        (
            [
                "-gen-dialect-decls",
                "-dialect=enzymexla",
            ],
            "Dialect/EnzymeXLADialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect=enzymexla",
            ],
            "Dialect/EnzymeXLADialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Dialect/EnzymeXLAOps.td",
    deps = [
        ":EnzymeXLAAttrsIncGen",
        ":EnzymeXLADialectTdFiles",
        "@enzyme//:EnzymeDialectTdFiles",
        "@llvm-project//mlir:GPUOpsTdFiles",
        "@stablehlo//:stablehlo_ops_td_files",
    ],
)

gentbl_cc_library(
    name = "EnzymeHLOPatternsIncGen",
    tbl_outs = [
        (
            ["-gen-populate-patterns-func-decls"],
            "Passes/EnzymeHLOPatterns.h.inc",
        ),
        (
            ["-gen-populate-patterns-func-defs"],
            "Passes/EnzymeHLOPatterns.cpp.inc",
        ),
    ],
    tblgen = "//:enzymexlamlir-tblgen",
    td_file = "TransformOps/TransformOps.td",
    deps = [
        ":TransformOpsTdFiles",
    ],
)

gentbl_cc_library(
    name = "StablehloOptPatternsIncGen",
    tbl_outs = [
        (
            ["--gen-rewriters"],
            "Passes/StablehloOptPatterns.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Passes/StablehloOptPatterns.td",
    deps = [
        ":StablehloOptPatternsTdFiles",
    ],
)

td_library(
    name = "StablehloOptPatternsTdFiles",
    srcs = [
        "Passes/StablehloOptPatterns.td",
    ],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@stablehlo//:stablehlo_ops_td_files",
    ],
)

td_library(
    name = "DistributedDialectFiles",
    srcs = [
        "Dialect/Distributed/Dialect.td",
        "Dialect/Distributed/Interfaces.td",
        "Dialect/Distributed/Ops.td",
        "Dialect/Distributed/Types.td",
    ],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "DistributedDialectIncGen",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
                "-dialect=distributed",
            ],
            "Dialect/Distributed/DistributedDialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect=distributed",
            ],
            "Dialect/Distributed/DistributedDialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Dialect/Distributed/Dialect.td",
    deps = [
        ":DistributedDialectFiles",
    ],
)

gentbl_cc_library(
    name = "DistributedOpsIncGen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "Dialect/Distributed/DistributedOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "Dialect/Distributed/DistributedOps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Dialect/Distributed/Ops.td",
    deps = [
        ":DistributedDialectFiles",
    ],
)

gentbl_cc_library(
    name = "DistributedTypesIncGen",
    tbl_outs = [
        (
            ["--gen-typedef-decls"],
            "Dialect/Distributed/DistributedTypes.h.inc",
        ),
        (
            ["--gen-typedef-defs"],
            "Dialect/Distributed/DistributedTypes.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Dialect/Distributed/Types.td",
    deps = [
        ":DistributedDialectFiles",
    ],
)

gentbl_cc_library(
    name = "DistributedInterfacesIncGen",
    tbl_outs = [
        (
            ["--gen-interface-decls"],
            "Dialect/Distributed/DistributedInterfaces.h.inc",
        ),
        (
            ["--gen-interface-defs"],
            "Dialect/Distributed/DistributedInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Dialect/Distributed/Interfaces.td",
    deps = [
        ":DistributedDialectFiles",
    ],
)

td_library(
    name = "TesseraDialectTdFiles",
    srcs = [
        "Dialect/Tessera/Dialect.td",
        "Dialect/Tessera/Ops.td",
    ],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:CallOpInterfaces",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:FunctionInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "TesseraDialectIncGen",
    tbl_outs = [
        (
            [
                "-gen-dialect-decls",
                "-dialect=tessera",
            ],
            "Dialect/Tessera/TesseraDialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect=tessera",
            ],
            "Dialect/Tessera/TesseraDialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Dialect/Tessera/Dialect.td",
    deps = [
        ":TesseraDialectTdFiles",
    ],
)

gentbl_cc_library(
    name = "TesseraOpsIncGen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "Dialect/Tessera/TesseraOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "Dialect/Tessera/TesseraOps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Dialect/Tessera/Ops.td",
    deps = [
        ":TesseraDialectTdFiles",
    ],
)

td_library(
    name = "TesseraPassesTdFiles",
    srcs = [
    ],
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "TesseraPassesIncGen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=tessera",
            ],
            "Passes/Tessera/Passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Passes/Tessera/Passes.td",
    deps = [":TesseraPassesTdFiles"],
)

cc_library(
    name = "CheckedRewrite",
    hdrs = ["CheckedRewrite.h"],
    deps = [
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "XLADerivatives",
    srcs = glob([
        "Implementations/*.cpp",
        "Passes/*.cpp",
        "Passes/Tessera/*.cpp",
        "Dialect/*.cpp",
        "Dialect/Distributed/*.cpp",
        "Dialect/Tessera/*.cpp",
    ]) + [
        "Utils.cpp",
    ],
    hdrs = glob([
        "Implementations/*.h",
        "Passes/*.h",
        "Passes/Tessera/*.h",
        "Dialect/*.h",
        "Dialect/Distributed/*.h",
        "Dialect/Tessera/*.h",
    ]) + [
        "Utils.h",
    ],
    copts = [
        "-Werror=unused-variable",
        "-Werror=unused-result",
        "-Werror=missing-braces",
        "-Werror=unused-but-set-variable",
        "-Werror=return-type",
        "-Werror=unused-result",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":CheckedRewrite",
        ":DistributedDialectIncGen",
        ":DistributedOpsIncGen",
        ":DistributedTypesIncGen",
        ":EnzymeHLOPatternsIncGen",
        ":EnzymeXLAAttrsIncGen",
        ":EnzymeXLAOpsIncGen",
        ":EnzymeXLAPassesIncGen",
        ":RaisingTransformOps",
        ":RaisingTransformOpsImplIncGen",
        ":RaisingTransformOpsIncGen",
        ":RaisingTransformPatternsIncGen",
        ":StablehloOptPatternsIncGen",
        ":TesseraDialectIncGen",
        ":TesseraOpsIncGen",
        ":TesseraPassesIncGen",
        ":chlo-derivatives",
        ":mhlo-derivatives",
        ":stablehlo-derivatives",
        "//src/external/isl:Isl",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@enzyme//:EnzymeMLIR",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:ExecutionEngine",
        "@llvm-project//llvm:IPO",
        "@llvm-project//llvm:InstCombine",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:OrcTargetProcess",
        "@llvm-project//llvm:Passes",
        "@llvm-project//llvm:Scalar",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineAnalysis",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineToStandard",
        "@llvm-project//mlir:AffineTransforms",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithToLLVM",
        "@llvm-project//mlir:ArithUtils",
        "@llvm-project//mlir:AsyncDialect",
        "@llvm-project//mlir:BytecodeOpInterface",
        "@llvm-project//mlir:CallOpInterfaces",
        "@llvm-project//mlir:CommonFolders",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:ControlFlowToLLVM",
        "@llvm-project//mlir:ControlFlowToSCF",
        "@llvm-project//mlir:DLTIDialect",
        "@llvm-project//mlir:FromLLVMIRTranslation",
        "@llvm-project//mlir:FromLLVMIRTranslationRegistration",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncToLLVM",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:GPUCommonTransforms",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:GPUPipelines",
        "@llvm-project//mlir:GPUToGPURuntimeTransforms",
        "@llvm-project//mlir:GPUToNVVMTransforms",
        "@llvm-project//mlir:GPUTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:IndexToLLVM",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:InliningUtils",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MathToLLVM",
        "@llvm-project//mlir:MathToLibm",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MemRefToLLVM",
        "@llvm-project//mlir:MemRefTransforms",
        "@llvm-project//mlir:MemRefUtils",
        "@llvm-project//mlir:MemorySlotInterfaces",
        "@llvm-project//mlir:NVGPUDialect",
        "@llvm-project//mlir:NVGPUToNVVM",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:NVVMToLLVM",
        "@llvm-project//mlir:OpenMPDialect",
        "@llvm-project//mlir:OpenMPToLLVM",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:ROCDLDialect",
        "@llvm-project//mlir:ReconcileUnrealizedCasts",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFToControlFlow",
        "@llvm-project//mlir:SCFToOpenMP",
        "@llvm-project//mlir:SCFTransforms",
        "@llvm-project//mlir:SCFUtils",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:ToLLVMIRTranslation",
        "@llvm-project//mlir:ToLLVMIRTranslationRegistration",
        "@llvm-project//mlir:TransformDialect",
        "@llvm-project//mlir:TransformDialectInterfaces",
        "@llvm-project//mlir:TransformDialectTransforms",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:TranslateLib",
        "@llvm-project//mlir:UBDialect",
        "@llvm-project//mlir:UBToLLVM",
        "@llvm-project//mlir:VectorDialect",
        "@llvm-project//mlir:VectorToLLVM",
        "@llvm-project//mlir:VectorToSCF",
        "@llvm-project//mlir:ViewLikeInterface",
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@shardy//shardy/dialect/sdy/transforms/propagation:op_sharding_rule_builder",
        "@stablehlo//:base",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:reference_ops",
        "@stablehlo//:reference_types",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:stablehlo_passes",
        "@stablehlo//:stablehlo_type_inference",
        "@xla//xla/mlir/utils:type_util",
        "@xla//xla/mlir_hlo",
    ],
)

cc_library(
    name = "RegistryUtils",
    srcs = ["RegistryUtils.cpp"],
    hdrs = glob([
        "Implementations/*.h",
        "Passes/*.h",
    ]) + [
        "RegistryUtils.h",
    ],
    deps = [
        ":TransformOps",
        ":XLADerivatives",
        ":EnzymeXLAPassesIncGen",
        "@com_google_protobuf//:protobuf",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:env_impl",
        "@xla//xla/tsl/protobuf:dnn_proto_cc_impl",
        "@xla//xla/tsl/protobuf:histogram_proto_cc",
        "@xla//xla/tsl/protobuf:histogram_proto_cc_impl",
        "@xla//xla/tsl/protobuf:protos_all_cc_impl",
        "@xla//xla:xla_data_proto_cc_impl",
        "@xla//xla:xla_proto_cc_impl",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/service:hlo_proto_cc_impl",
        "@xla//xla/service:metrics_proto_cc_impl",
        "@com_google_absl//absl/status:statusor",

        # MHLO stuff.
        "@xla//xla/mlir_hlo",
        "@xla//xla/mlir_hlo:stablehlo_extension_passes",
        "@xla//xla/mlir_hlo:mhlo_passes",
        "@xla//xla/hlo/translate/mhlo_to_hlo:mlir_hlo_to_hlo",
        "@xla//xla/hlo/ir:hlo",

        # MLIR dialects and parser.
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:UBDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithTransforms",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:ConversionPasses",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:SparseTensorDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:LLVMIRTransforms",
        "@llvm-project//mlir:NVVMTarget",
        "@llvm-project//mlir:LinalgTransformOps",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:NVVMToLLVMIRTranslation",
        "@llvm-project//mlir:ArithToLLVM",
        "@llvm-project//mlir:ComplexToLLVM",
        "@llvm-project//mlir:IndexToLLVM",
        "@llvm-project//mlir:ControlFlowToLLVM",
        "@llvm-project//mlir:ConvertToLLVM",
        "@llvm-project//mlir:FuncToLLVM",
        "@llvm-project//mlir:MathToLLVM",
        "@llvm-project//mlir:MemRefToLLVM",
        "@llvm-project//mlir:NVVMToLLVM",
        "@llvm-project//mlir:OpenMPToLLVM",
        "@llvm-project//mlir:UBToLLVM",
        "@llvm-project//mlir:VectorToLLVM",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineTransforms",
        "@llvm-project//mlir:AsyncDialect",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:DLTIDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:GPUTransforms",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:OpenMPDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:TransformDialect",
        "@llvm-project//mlir:TransformDialectTransforms",
        "@llvm-project//mlir:VectorDialect",
        "@llvm-project//mlir:NVGPUDialect",
        "@llvm-project//mlir:BuiltinToLLVMIRTranslation",
        "@llvm-project//mlir:GPUToLLVMIRTranslation",
        "@llvm-project//mlir:OpenMPToLLVMIRTranslation",
        "@llvm-project//mlir:LLVMIRToLLVMTranslation",
        "@llvm-project//mlir:LLVMIRToNVVMTranslation",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:Transforms",

        # Triton
        "@triton//:TritonDialects",
        "@triton//:TritonGPUToLLVM",
        "@triton//:TritonGPUTransforms",
        "@triton//:TritonLLVMIR",
        "@triton//:TritonNvidiaGPUTransforms",
        "@triton//:TritonToTritonGPU",
        "@triton//:TritonToTritonGPUPasses",
        "@triton//:TritonTransforms",
        "@triton//:WarpSpecialization",
        "@triton//:triton_conversion_triton_to_triton_gpu_passes_inc_gen",
        "@triton//:triton_nvidia_gpu_transforms_inc_gen",
        "@triton//third_party/amd:TritonAMDGPU",
        "@triton//third_party/amd:TritonAMDGPUToLLVM",
        "@triton//third_party/amd:TritonAMDGPUTransforms",
        "@triton//third_party/nvidia:NVGPUDialect",
        "@triton//third_party/nvidia:NVGPUToLLVM",
        "@triton//third_party/nvidia:NVHopperTransforms",
        "@triton//third_party/nvidia:NVWSDialect",
        "@triton//third_party/nvidia:NVWSTransforms",
        "@triton//third_party/nvidia:TritonNVIDIAGPUToLLVM",
        "@triton//third_party/proton:ProtonIRDialect",

        # Shardy stuff
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@shardy//shardy/dialect/sdy/transforms/propagation:op_sharding_rule_builder",

        # XLA stuff
        "@xla//xla/mlir_hlo:all_passes",
        "@xla//xla:printer",

        # EnzymeMLIR
        "@enzyme//:EnzymeMLIR",

        # Mosaic
        "@jax//jaxlib/mosaic:tpu_dialect",

        # SHLO
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:linalg_passes",
        "@stablehlo//:tosa_passes",
        "@stablehlo//:stablehlo_passes",
        "@stablehlo//:stablehlo_passes_optimization",
        "@stablehlo//stablehlo/tests:check_ops",

        # Shardy
        "@shardy//shardy/dialect/sdy/transforms:passes",
        "@shardy//shardy/dialect/sdy/transforms/propagation:passes",
        "@shardy//shardy/dialect/sdy/transforms/export:passes",
        "@shardy//shardy/dialect/sdy/transforms/import:passes",
        "@xla//xla/service/spmd/shardy/stablehlo_round_trip:export_shardings",
        "@xla//xla/service/spmd/shardy/stablehlo_round_trip:stablehlo_import",
        "@xla//xla/service/spmd/shardy/stablehlo_round_trip:stablehlo_export",
        "@xla//xla/service/spmd/shardy/sdy_round_trip:pipelines",
    ],
)

cc_library(
    name = "compile_with_xla",
    srcs = ["compile_with_xla.cc"],
    hdrs = ["compile_with_xla.h"],
    copts = [
        "-Wno-implicit-fallthrough",
        "-Wno-non-virtual-dtor",
        "-Wno-pessimizing-move",
        "-Wno-unused-result",
        "-Wno-unused-variable",
        "-Wno-vla-cxx-extension",
    ],
    deps = [
        "@nanobind",
        ":XLADerivatives",
        ":TransformOps",
        ":RegistryUtils",
        "@com_google_protobuf//:protobuf",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:env_impl",
        "@tsl//tsl/platform:tensor_float_32_utils",
        "@xla//xla/tsl/profiler/backends/cpu:traceme_recorder_impl",
        "@xla//xla/tsl/profiler/utils:time_utils_impl",
        "@xla//xla/tsl/protobuf:dnn_proto_cc_impl",
        "@xla//xla/tsl/protobuf:histogram_proto_cc",
        "@xla//xla/tsl/protobuf:histogram_proto_cc_impl",
        "@xla//xla/tsl/protobuf:protos_all_cc_impl",
        "@xla//xla/client",
        "@xla//xla/client:client_library",
        "@xla//xla/client:executable_build_options",
        "@xla//xla/client:local_client",
        "@xla//xla/hlo/builder:xla_computation",
        "@xla//xla/service",
        "@xla//xla/service:metrics_proto_cc",
        "@xla//xla/service:metrics_proto_cc_impl",
        "@xla//xla/service:local_service",
        "@xla//xla/service:local_service_utils",
        "@xla//xla/service/cpu:cpu_executable",
        "@xla//xla/service/cpu:backend_config_proto_cc",
        "@xla//xla/service/gpu:backend_configs_cc",
        "@xla//xla/service/gpu:backend_configs_cc_impl",
        "@xla//xla/service:hlo_proto_cc",
        "@xla//xla/service:hlo_proto_cc_impl",
        "@xla//xla/service/memory_space_assignment:memory_space_assignment_proto_cc_impl",
        "@xla//xla/hlo/translate/mhlo_to_hlo:type_to_shape",
        "@xla//xla:autotune_results_proto_cc",
        "@xla//xla:autotune_results_proto_cc_impl",
        "@xla//xla:autotuning_proto_cc",
        "@xla//xla:autotuning_proto_cc_impl",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla:xla_data_proto_cc_impl",
        "@xla//xla:xla_proto_cc",
        "@xla//xla:xla_proto_cc_impl",
        "@xla//xla/service/cpu:cpu_compiler_pure",
        "@xla//xla/service:compiler",
        "@xla//xla/service:hlo_module_util",
        "@xla//xla/service:hlo_proto_util",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:chlo_ops",

        # Make CPU target available to XLA.
        "@xla//xla/service:cpu_plugin",

        # MHLO stuff.
        "@xla//xla/mlir_hlo",
        "@xla//xla/hlo/translate/mhlo_to_hlo:mlir_hlo_to_hlo",
        "@xla//xla/hlo/ir:hlo",

        # MLIR dialects and parser.
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:UBDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:SparseTensorDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:LLVMIRTransforms",
        "@llvm-project//mlir:NVVMTarget",
        "@llvm-project//mlir:LinalgTransformOps",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:NVVMToLLVMIRTranslation",
        "@llvm-project//mlir:ArithToLLVM",
        "@llvm-project//mlir:ComplexToLLVM",
        "@llvm-project//mlir:IndexToLLVM",
        "@llvm-project//mlir:ControlFlowToLLVM",
        "@llvm-project//mlir:ConvertToLLVM",
        "@llvm-project//mlir:FuncToLLVM",
        "@llvm-project//mlir:MathToLLVM",
        "@llvm-project//mlir:MemRefToLLVM",
        "@llvm-project//mlir:NVVMToLLVM",
        "@llvm-project//mlir:OpenMPToLLVM",
        "@llvm-project//mlir:UBToLLVM",
        "@llvm-project//mlir:VectorToLLVM",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AsyncDialect",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:DLTIDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:OpenMPDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:TransformDialect",
        "@llvm-project//mlir:VectorDialect",
        "@llvm-project//mlir:NVGPUDialect",
        "@llvm-project//mlir:BuiltinToLLVMIRTranslation",
        "@llvm-project//mlir:GPUToLLVMIRTranslation",
        "@llvm-project//mlir:OpenMPToLLVMIRTranslation",
        "@llvm-project//mlir:LLVMIRToLLVMTranslation",
        "@llvm-project//mlir:LLVMIRToNVVMTranslation",

        # Shardy stuff
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@shardy//shardy/dialect/sdy/transforms/propagation:op_sharding_rule_builder",

        # XLA stuff
        "@xla//xla/mlir_hlo:all_passes",
        "@xla//xla:printer",

        # EnzymeMLIR
        "@enzyme//:EnzymeMLIR",
        "@com_google_absl//absl/status:statusor",

        # Mosaic
        "@jax//jaxlib/mosaic:tpu_dialect",
    ],
)

pybind_extension(
    name = "enzyme_call",
    srcs = [
        "enzyme_call.cc",
    ],
    copts = [
        "-Wno-implicit-fallthrough",
        "-Wno-non-virtual-dtor",
        "-Wno-pessimizing-move",
        "-Wno-unused-result",
        "-Wno-unused-variable",
        "-Wno-vla-cxx-extension",
    ],
    linkopts = select({
        "@platforms//os:macos": [
            "-undefined",
            "dynamic_lookup",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":RegistryUtils",
        ":TransformOps",
        ":XLADerivatives",
        ":clang_compile",
        ":compile_with_xla",
        ":cpu",
        ":gpu",
        "@com_google_absl//absl/status:statusor",
        "@enzyme//:EnzymeMLIR",
        "@enzyme//:EnzymeStatic",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:ExecutionEngine",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:OrcTargetProcess",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:ArithToLLVM",
        "@llvm-project//mlir:ArithTransforms",
        "@llvm-project//mlir:AsyncDialect",
        "@llvm-project//mlir:AsyncTransforms",
        "@llvm-project//mlir:BufferizationPipelines",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:BuiltinToLLVMIRTranslation",
        "@llvm-project//mlir:CAPIIR",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:ComplexToLLVM",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:ControlFlowToLLVM",
        "@llvm-project//mlir:ConversionPasses",
        "@llvm-project//mlir:ConvertToLLVM",
        "@llvm-project//mlir:DLTIDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:FuncToLLVM",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:GPUToLLVMIRTranslation",
        "@llvm-project//mlir:IndexToLLVM",
        "@llvm-project//mlir:LLVMIRToLLVMTranslation",
        "@llvm-project//mlir:LLVMIRToNVVMTranslation",
        "@llvm-project//mlir:LLVMIRTransforms",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:LinalgTransformOps",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MLIRBindingsPythonHeaders",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MathToLLVM",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MemRefToLLVM",
        "@llvm-project//mlir:MemRefTransforms",
        "@llvm-project//mlir:NVGPUDialect",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:NVVMTarget",
        "@llvm-project//mlir:NVVMToLLVM",
        "@llvm-project//mlir:NVVMToLLVMIRTranslation",
        "@llvm-project//mlir:OpenMPDialect",
        "@llvm-project//mlir:OpenMPToLLVM",
        "@llvm-project//mlir:OpenMPToLLVMIRTranslation",
        "@llvm-project//mlir:RegisterAllPasses",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:ShapeTransforms",
        "@llvm-project//mlir:SparseTensorDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformDialect",
        "@llvm-project//mlir:UBDialect",
        "@llvm-project//mlir:UBToLLVM",
        "@llvm-project//mlir:VectorDialect",
        "@llvm-project//mlir:VectorToLLVM",
        "@nanobind",
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@shardy//shardy/dialect/sdy/transforms/propagation:op_sharding_rule_builder",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:stablehlo_passes",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:env_impl",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/mlir_hlo",
        "@xla//xla/mlir_hlo:all_passes",
        "@xla//xla/mlir_hlo:deallocation_passes",
        "@xla//xla/service:buffer_assignment_proto_cc_impl",
        "@xla//xla/service/cpu:cpu_executable",
        "@xla//xla/stream_executor:device_description_proto_cc_impl",
        "@xla//xla/stream_executor:stream_executor_impl",
        "@xla//xla/stream_executor/cuda:cuda_compute_capability_proto_cc_impl",
        "@xla//xla/tsl/profiler/backends/cpu:annotation_stack_impl",
    ],
)
