load("@rules_python//python:py_test.bzl", "py_test")

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

