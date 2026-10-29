load("@python_version_repo//:py_version.bzl", "REQUIREMENTS")
load("@rules_python//python:py_runtime.bzl", "py_runtime")
load("@rules_python//python:py_runtime_pair.bzl", "py_runtime_pair")
load("@rules_uv//uv:pip.bzl", "pip_compile")
load(":trimmed_python.bzl", "filtered_files")

# The hermetic toolchain's runfiles, minus what no test needs at runtime:
# pip/setuptools (site-packages), the C headers, and stdlib packages for
# IDEs, GUIs and building extensions. Selected with
# --extra_toolchains=//builddeps:trimmed_py_toolchain.
filtered_files(
    name = "trimmed_python_files",
    srcs = ["@python_3_11_x86_64-unknown-linux-gnu//:files"],
    exclude_substrings = [
        "/site-packages/",
        "/include/",
        "/idlelib/",
        "/lib2to3/",
        "/distutils/",
        "/ensurepip/",
        "/turtledemo/",
        "/tkinter/",
        "/pydoc_data/",
        "/share/",
    ],
)

py_runtime(
    name = "trimmed_py3_runtime",
    files = [":trimmed_python_files"],
    implementation_name = "cpython",
    interpreter = "@python_3_11_x86_64-unknown-linux-gnu//:bin/python3",
    interpreter_version_info = {
        "major": "3",
        "micro": "15",
        "minor": "11",
        "releaselevel": "final",
        "serial": "0",
    },
    pyc_tag = "cpython-311",
    python_version = "PY3",
)

py_runtime_pair(
    name = "trimmed_py_runtimes",
    py3_runtime = ":trimmed_py3_runtime",
)

toolchain(
    name = "trimmed_py_toolchain",
    target_compatible_with = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
    toolchain = ":trimmed_py_runtimes",
    toolchain_type = "@rules_python//python:toolchain_type",
)

pip_compile(
    name = "requirements",
    data = [
        "overrides.txt",
        "test-requirements.txt",
    ],
    extra_args = [
        "--build-isolation",
        "--prerelease=allow",
        "--override=builddeps/overrides.txt",
        "--resolution=highest",
    ],
    requirements_in = "requirements.in",
    requirements_txt = REQUIREMENTS,
)
