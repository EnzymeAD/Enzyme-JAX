load("@llvm-project//llvm:lit_test.bzl", "lit_test", "package_path")
load("@bazel_skylib//rules:expand_template.bzl", "expand_template")

[
    lit_test(
        name = "%s.test" % src,
        srcs = [src],
        data = [
            "//test:lit.cfg.py",
            "//test:lit_site_cfg_py",
            "//:enzymexlamlir-opt",
            "@llvm-project//llvm:FileCheck",
            "@llvm-project//llvm:count",
            "@llvm-project//llvm:not",
            "@stablehlo//:stablehlo-translate"
        ] + glob(["**/*.h"]),
    )
    for src in glob(
        [
            "**/*.mlir",
        ],
    )
]