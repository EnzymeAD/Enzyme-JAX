"""Loads the Flatbuffers library JAX generates with.

JAX dropped its own definition of this repository when it moved to Bzlmod;
the version it asks for is now the `bazel_dep` in its MODULE.bazel, and
jaxlib's `flatbuffer_cc_library` still expects the repository under the name
that module is bound to.
"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@jax_deps//:deps.bzl", "FLATBUFFERS_VERSION")

# The two build fixes JAX carried in the patch that came with its definition.
FLATBUFFERS_PATCHES = [
    """
    # Bazel links C++ with `clang` rather than `clang++`, so the math symbols
    # flatc uses have to be asked for explicitly.
    sed -i.bak0 's|"//:platform_openbsd": \\["-lm"\\],|"//:platform_openbsd": ["-lm"], "//:platform_linux": ["-lm"],|g' src/BUILD.bazel
    sed -i.bak1 's|name = "platform_openbsd",|name = "platform_linux", constraint_values = ["@platforms//os:linux"],)\\n\\nconfig_setting(\\n    name = "platform_openbsd",|' BUILD.bazel
    """,
    """
    # flatc's code generators register themselves from static initializers.
    perl -0777 -pi -e 's|(cc_library\\(\\n    name = "flatc",)|$1\\n    alwayslink = 1,|' src/BUILD.bazel
    """,
]

def repo():
    http_archive(
        name = "com_github_google_flatbuffers",
        patch_cmds = FLATBUFFERS_PATCHES,
        strip_prefix = "flatbuffers-" + FLATBUFFERS_VERSION,
        urls = ["https://github.com/google/flatbuffers/archive/v{version}.tar.gz".format(
            version = FLATBUFFERS_VERSION,
        )],
    )
