load("@python_version_repo//:py_version.bzl", "REQUIREMENTS")
load("@rules_python//python:py_runtime.bzl", "py_runtime")
load("@rules_python//python:py_runtime_pair.bzl", "py_runtime_pair")
load("@rules_uv//uv:pip.bzl", "pip_compile")
load(":python_tree.bzl", "python_tree")

# The hermetic Python runtime with lib/ as one directory artifact, so that a
# py_test's runfiles tree holds two entries for the runtime instead of one
# symlink per file (~2,300). The Linux CI runners select it with
# --extra_toolchains; see .github/workflows/build.yml.
[
    (
        python_tree(
            name = "python_dir_" + cpu,
            srcs = ["@python_3_11_%s-unknown-linux-gnu//:files" % cpu],
            exclude_substrings = [
                "/site-packages/",
                "/idlelib/",
                "/lib2to3/",
                "/distutils/",
                "/ensurepip/",
                "/turtledemo/",
                "/tkinter/",
                "/pydoc_data/",
            ],
            interpreter = "@python_3_11_%s-unknown-linux-gnu//:bin/python3" % cpu,
            interpreter_out = "python_dir_%s/bin/python3" % cpu,
        ),
        py_runtime(
            name = "dir_py3_runtime_" + cpu,
            files = [":python_dir_" + cpu],
            implementation_name = "cpython",
            interpreter = ":python_dir_%s/bin/python3" % cpu,
            interpreter_version_info = {
                "major": "3",
                "micro": "15",
                "minor": "11",
                "releaselevel": "final",
                "serial": "0",
            },
            pyc_tag = "cpython-311",
            python_version = "PY3",
        ),
        py_runtime_pair(
            name = "dir_py_runtimes_" + cpu,
            py3_runtime = ":dir_py3_runtime_" + cpu,
        ),
        toolchain(
            name = "dir_py_toolchain_" + cpu,
            target_compatible_with = [
                "@platforms//os:linux",
                "@platforms//cpu:" + cpu,
            ],
            toolchain = ":dir_py_runtimes_" + cpu,
            toolchain_type = "@rules_python//python:toolchain_type",
        ),
    )
    for cpu in [
        "x86_64",
        "aarch64",
    ]
]

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
