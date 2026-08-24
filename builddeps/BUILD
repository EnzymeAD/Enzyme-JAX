load("@python_version_repo//:py_version.bzl", "HERMETIC_PYTHON_VERSION")
load("@rules_uv//uv:pip.bzl", "pip_compile")

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
    requirements_txt = "requirements_lock_{}.txt".format(HERMETIC_PYTHON_VERSION.replace(".", "_")),
)
