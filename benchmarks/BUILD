load("@bazel_skylib//rules:common_settings.bzl", "string_flag")
load("@rules_python//python:py_binary.bzl", "py_binary")

string_flag(
    name = "test_tpu",
    build_setting_default = "False",
)

config_setting(
    name = "use_tpu",
    flag_values = {
        ":test_tpu": "True",
    },
)

BENCH_DEPS = [
    "//:enzyme_ad",
    "@pypi_jax//:pkg",
] + select({
    ":use_tpu": [
        "@pypi_libtpu_nightly//:pkg",
        "@pypi_requests//:pkg",
    ],
    "@bazel_tools//src/conditions:linux_x86_64": ["@pypi_jax_cuda12_plugin//:pkg"],
    "//conditions:default": [],
})

py_binary(
    name = "jaxmd",
    srcs = [
        "jaxmd.py",
        "bench_utils.py",
    ],
    imports = ["."],
    deps = BENCH_DEPS + [
        "@pypi_jax_md//:pkg",
    ]
)