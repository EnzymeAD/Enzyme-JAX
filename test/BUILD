load("@rules_python//python:py_test.bzl", "py_test")
load("@llvm-project//llvm:lit_test.bzl", "package_path", "lit_test")


exports_files(
    ["lit_tests/lit.cfg.py"],
    visibility = [":__subpackages__"],
)

[
    lit_test(
        name = "%s.test" % src,
        srcs = [src],
        data = [
            "//test:lit_tests/lit.cfg.py",
            "//src/enzyme_ad/jax:enzyme_jax_internal",
            "@llvm-project//clang:builtin_headers_gen",
            "@llvm-project//llvm:FileCheck",
            "@llvm-project//llvm:count",
            "@llvm-project//llvm:not",
        ] + glob(["**/*.h"]),
    )
    for src in glob(
        [
            "**/*.pyt",
        ],
    )
]

py_test(
    name = "test",
    srcs = [
        "test.py",
    ],
    deps = [
        "//src/enzyme_ad/jax:enzyme_jax_internal",
    ],
)

py_test(
    name = "bench_vs_xla",
    srcs = [
        "bench_vs_xla.py",
    ],
    deps = [
        "//src/enzyme_ad/jax:enzyme_jax_internal",
    ],
)

