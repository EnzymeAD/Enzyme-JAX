"""Loads libblastrampoline."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

LIBBLASTRAMPOLINE_COMMIT = "072b5f67895bec0b92f8c83194567c1c48e9833d"  # v5.15.0
LIBBLASTRAMPOLINE_SHA256 = ""

PATCHES = [
    # fix LBT_ROOT path
    """
    sed -i "s/LBT_ROOT := .*/LBT_ROOT := \\$(EXT_BUILD_ROOT)\\/external\\/libblastrampoline/g" src/Makefile
    """,
    # remove prefix assignment and replace DESTDIR with INSTALLDIR
    """
    sed -i "/prefix ?= prefix/d" src/Make.inc
    """,
    """
    sed -i "s/DESTDIR/INSTALLDIR/g" src/Makefile
    """,
    # dereference symbolic links
    """
    sed -i "s/cp -a/cp -l/g" src/Makefile
    """,
    """
    sed -i "s/cp -Ra/cp -RL/g" src/Makefile
    """,
    # enable echoing commands to ease debugging
    """
    sed -i -E "s/^(\\s*)@/\\1/g" src/Makefile
    """,
    """
    sed -i "s/-@//g" src/Makefile
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
