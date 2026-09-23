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
            # `.mlir.txt` is the convention for a real (or realistic) auxiliary
            # MLIR fixture that a RUN line reads via %S but that isn't itself a
            # standalone test -- the extension keeps it out of the `srcs` glob
            # below, which would otherwise treat it as its own FileCheck test.
        ] + glob([
            "**/*.h",
            "**/*.mlir.txt",
        ]),
    )
    for src in glob(
        [
            "**/*.mlir",
        ],
    )
]
