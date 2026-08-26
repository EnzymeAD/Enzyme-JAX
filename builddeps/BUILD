load("@python_version_repo//:py_version.bzl", "REQUIREMENTS")
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
    requirements_txt = REQUIREMENTS,
)
