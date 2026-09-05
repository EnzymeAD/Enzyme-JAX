exports_files([
    "EnzymeXLALspServerMain.cpp",
    "enzymexlamlir-tblgen.cpp",
])

licenses(["notice"])

package(
    default_visibility = ["//:__subpackages__"],
    features = ["layering_check"],
)

cc_binary(
    name = "differential_fuzzer",
    srcs = ["differential_fuzzer.cc"],
    deps = [
        "//src/enzyme_ad/jax:TransformOps",
        "//src/enzyme_ad/jax:XLADerivatives",
        
        # MLIR parsing and execution
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:TransformDialect",
        "@llvm-project//mlir:TransformDialectTransforms",

        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:TensorDialect",

        "@llvm-project//mlir:LLVMDialect",

        "@triton//:TritonToTritonGPU",
        
        # StableHLO Interpreter
        "@stablehlo//:reference_api",
        "@stablehlo//:reference_ops",
        "@stablehlo//:reference_scope",
        "@stablehlo//:reference_types",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:stablehlo_passes",
    ],
)
