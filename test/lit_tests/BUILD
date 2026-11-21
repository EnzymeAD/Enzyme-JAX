load("@llvm-project//llvm:lit_test.bzl", "lit_test")

cc_binary(
    name = "reactant-pipeline-driver",
    testonly = True,
    srcs = ["ReactantPipelineDriver.cpp"],
    deps = [
        "//:RaiseLib",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:Support",
    ],
)

[
    lit_test(
        name = "%s.test" % src,
        timeout = "short",
        srcs = [src],
        data = [
            "//:enzymexlamlir-opt",
            "//test:lit.cfg.py",
            "//test:lit_site_cfg_py",
            "@llvm-project//llvm:FileCheck",
            "@llvm-project//llvm:count",
            "@llvm-project//llvm:not",
            "@stablehlo//:stablehlo-translate",
        ] + glob(["**/*.h"]) + ([
            ":reactant-pipeline-driver",
        ] + glob(["sql/Inputs/*.ll"]) if src.startswith("sql/pipeline") else []),
    )
    for src in glob(
        [
            "**/*.mlir",
        ],
    )
]
