licenses(["notice"])

load("@python//:defs.bzl", "compile_pip_requirements")
load("@python_version_repo//:py_version.bzl", "REQUIREMENTS")

compile_pip_requirements(
    name = "requirements",
    extra_args = [
        "--allow-unsafe",
        "--build-isolation",
        "--rebuild",
    ],
    requirements_in = "requirements.in",
    requirements_txt = REQUIREMENTS,
    generate_hashes = True,
    data = ["test-requirements.txt"]
)


py_library(
    name = "all_py_deps",
    deps = [
        "@pypi_absl_py//:pkg",
        "@jax//:pkg",
    ]
)

