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
            # TODO(leonardoroman): Re-enable once NVPTX options reach the target
            # machine again after LLVM r644962 (llvm/llvm-project#238141).
            "lowering/gpu-binary-llvm-opts.mlir",
        ],
    )
]
