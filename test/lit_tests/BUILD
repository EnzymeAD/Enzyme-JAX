load("@llvm-project//llvm:lit_test.bzl", "lit_test")

# Tests whose RUN lines each drive a real search over enough candidates that
# the default "short" (60s) timeout isn't enough headroom.
_MODERATE_TIMEOUT_TESTS = [
    "distributed/search_twostage_surrogate.mlir",
]

[
    lit_test(
        name = "%s.test" % src,
        timeout = "moderate" if src in _MODERATE_TIMEOUT_TESTS else "short",
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
