"""Loads libblastrampoline."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

LIBBLASTRAMPOLINE_COMMIT = "072b5f67895bec0b92f8c83194567c1c48e9833d"  # v5.15.0
LIBBLASTRAMPOLINE_SHA256 = ""

PATCHES = [
    """
    sed -i.bak0 "s/\\$(LBT_ROOT)\\/src\\//\\$(LBT_ROOT)\\/libblastrampoline.build_tmpdir\\//g" src/Makefile
    """,
    """
    sed -i.bak0 "s/prefix/PREFIX/g" src/Make.inc
    """,
]

def repo(repo_name = ""):
    # When used as an external dependency, repo_name should be "@enzyme_ad"
    # When used standalone, repo_name should be "" (empty string)
    build_file_label = repo_name + "//third_party/libblastrampoline:libblastrampoline.BUILD"
    http_archive(
        name = "libblastrampoline",
        sha256 = LIBBLASTRAMPOLINE_SHA256,
        strip_prefix = "libblastrampoline-" + LIBBLASTRAMPOLINE_COMMIT,
        urls = ["https://github.com/JuliaLinearAlgebra/libblastrampoline/archive/{commit}.tar.gz".format(commit = LIBBLASTRAMPOLINE_COMMIT)],
        build_file = build_file_label,
        patch_cmds = PATCHES,
    )
