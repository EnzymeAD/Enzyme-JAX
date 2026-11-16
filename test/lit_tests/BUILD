load("@llvm-project//llvm:lit_test.bzl", "lit_test")

package(default_visibility = ["//visibility:public"])

[
    lit_test(
        name = "%s.test" % src,
        timeout = "short",
        srcs = [src],
        data = [
            "//test:lit.cfg.py",
            "//test:lit_site_cfg_py",
            "@llvm-project//llvm:FileCheck",
            "@llvm-project//llvm:count",
            "@llvm-project//llvm:not",
            "//:enzymexlamlir-opt",
            "@stablehlo//:stablehlo-translate",
        ] + glob(["**/*.h"]),
    )
    for src in glob(
        [
            "**/*.mlir",
        ],
        exclude = [
            "perfify/**",
            "cutile/**",
            "lowering/gpu-binary-llvm-opts.mlir",
        ],
    )
]
