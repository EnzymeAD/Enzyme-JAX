load("@llvm-project//llvm:lit_test.bzl", "lit_test")

[
    lit_test(
        name = "%s.test" % src,
        timeout = "short",
        srcs = [src],
        data = [
            "@llvm-project//llvm:FileCheck",
            "@llvm-project//llvm:count",
            "@llvm-project//llvm:not",
            "//:enzymexlamlir-opt",
            "//test:lit.cfg.py",
            "//test:lit_site_cfg_py",
            "@stablehlo//:stablehlo-translate",
        ] + glob(["**/*.h"]),
            )
    for src in glob(
        [
            "**/*.mlir",
        ],
    )
]
