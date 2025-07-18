load("@bazel_skylib//rules:common_settings.bzl", "string_flag")
load("@bazel_skylib//rules:expand_template.bzl", "expand_template")
load("@llvm-project//llvm:lit_test.bzl", "lit_test", "package_path")
load("@rules_python//python:py_test.bzl", "py_test")

expand_template(
    name = "lit_site_cfg_py",
    testonly = True,
    out = "lit.site.cfg.py",
    substitutions = {
        "@LIT_SITE_CFG_IN_HEADER@": "# Autogenerated, do not edit.",
        "@LLVM_TOOLS_BINARY_DIR@": package_path("@llvm-project//llvm:BUILD"),
        "@ENZYMEXLA_BINARY_DIR@": "",
        "@LLVM_LIBS_DIR@": package_path("@llvm-project//llvm:BUILD"),
        "@ENZYME_SOURCE_DIR@": "",
        "@ENZYME_BINARY_DIR@": "",
    },
    template = "lit.site.cfg.py.in",
    visibility = [":__subpackages__"],
)

exports_files(
    ["lit.cfg.py"],
    visibility = [":__subpackages__"],
)

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

TEST_DEPS = [
    "//:enzyme_ad",
    "@pypi_jax//:pkg",
    "@pypi_absl_py//:pkg",
] + select({
    ":use_tpu": [
        "@pypi_libtpu_nightly//:pkg",
        "@pypi_requests//:pkg",
    ],
    "@bazel_tools//src/conditions:linux_x86_64": ["@pypi_jax_cuda12_plugin//:pkg"],
    "//conditions:default": [],
})

[
    lit_test(
        name = "%s.test" % src,
        srcs = [src],
        data = [
            ":lit.cfg.py",
            ":lit_site_cfg_py",
            "//:enzyme_ad",
            "@llvm-project//clang:builtin_headers_gen",
            "@llvm-project//llvm:FileCheck",
            "@llvm-project//llvm:count",
            "@llvm-project//llvm:not",
        ] + glob(["**/*.h"]),
        deps = TEST_DEPS,
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
    imports = ["site-packages"],
    tags = ["pypi_name=enzyme-ad"],
    deps = TEST_DEPS,
)

py_test(
    name = "bench_vs_xla",
    srcs = [
        "bench_vs_xla.py",
        "test_utils.py",
    ],
    imports = ["."],
    tags = ["exclusive"],
    deps = TEST_DEPS,
)

py_test(
    name = "testffi",
    srcs = [
        "test_utils.py",
        "testffi.py",
    ],
    imports = ["."],
    tags = ["exclusive"],
    deps = TEST_DEPS,
)

py_test(
    name = "llama",
    srcs = [
        "llama.py",
        "test_utils.py",
    ],
    imports = ["."],
    tags = ["exclusive"],
    deps = TEST_DEPS,
)

py_test(
    name = "jaxmd",
    timeout = "long",
    srcs = [
        "jaxmd.py",
        "test_utils.py",
    ],
    imports = ["."],
    tags = ["exclusive"],
    deps = TEST_DEPS + select({
        "@bazel_tools//src/conditions:linux_x86_64": [
            "@pypi_jax_md//:pkg",
            "@pypi_jraph//:pkg",
        ],
        "//conditions:default": [],
    }),
)

py_test(
    name = "neuralgcm_test",
    timeout = "eternal",
    srcs = [
        "neuralgcm_test.py",
        "test_utils.py",
    ],
    imports = ["."],
    tags = ["exclusive"],
    deps = TEST_DEPS + ["@pypi_neuralgcm//:pkg"] + select({
        "@bazel_tools//src/conditions:linux_x86_64": ["@pypi_gcsfs//:pkg"],
        "//conditions:default": [],
    }),
)

py_test(
    name = "keras_test",
    timeout = "eternal",
    srcs = [
        "keras_test.py",
        "test_utils.py",
    ],
    imports = ["."],
    tags = ["exclusive"],
    deps = TEST_DEPS + select({
        "@bazel_tools//src/conditions:linux_x86_64": [
            "@pypi_benchmark//:pkg",
            "@pypi_keras//:pkg",
            "@pypi_keras_cv//:pkg",
            "@pypi_keras_nlp//:pkg",
            "@pypi_tf_keras//:pkg",
        ],
        "//conditions:default": [],
    }),
)

# py_test(
#     name = "maxtext",
#     srcs = [
#         "maxtext.py",
#     ],
#     imports = ["."],
#     deps = TEST_DEPS + ["@pypi_maxtext//:pkg"],
#     timeout='long'
#     tags = ["exclusive"]
# )
