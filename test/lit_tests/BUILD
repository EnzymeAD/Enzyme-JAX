load("@llvm-project//llvm:lit_test.bzl", "lit_test")

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
        ] + glob(["**/*.h"]),
    )
    for src in glob(
        [
            "**/*.mlir",
        ],
    )
]
