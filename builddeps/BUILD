load("@python_version_repo//:py_version.bzl", "REQUIREMENTS")
load("@rules_python//python:pip.bzl", "compile_pip_requirements")

licenses(["notice"])

compile_pip_requirements(
    name = "requirements",
    data = ["test-requirements.txt"],
    extra_args = [
        "--allow-unsafe",
        "--build-isolation",
        "--rebuild",
    ],
    extra_deps = [
        # "@pypi_wheel//:pkg"
    ],
    generate_hashes = True,
    requirements_in = "requirements.in",
    requirements_txt = REQUIREMENTS,
)
