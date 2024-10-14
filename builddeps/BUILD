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
    extra_deps = [
        # "@pypi_wheel//:pkg"
    ],
    requirements_in = "requirements.in",
    requirements_txt = REQUIREMENTS,
    generate_hashes = True,
    data = ["test-requirements.txt"]
)
