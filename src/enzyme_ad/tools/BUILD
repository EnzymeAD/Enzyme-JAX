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
    name = "enzymexlamlir-fuzz",
    srcs = ["enzymexlamlir-fuzz.cpp"],
    deps = [
        "//src/enzyme_ad/jax:TransformOps",
        "//src/enzyme_ad/jax:XLADerivatives",
        "//src/enzyme_ad/jax:RegistryUtils",

        # MLIR parsing and execution
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:ParseUtilities",

        # StableHLO Interpreter
        "@stablehlo//:reference_api",
        "@stablehlo//:reference_ops",
        "@stablehlo//:reference_scope",
        "@stablehlo//:reference_types",
        "@stablehlo//:stablehlo_passes",
        "@stablehlo//:interpreter_ops",
        "@stablehlo//stablehlo/tests:check_ops",
    ],
)
