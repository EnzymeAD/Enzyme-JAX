"""Loads libblastrampoline."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

LIBBLASTRAMPOLINE_COMMIT = "072b5f67895bec0b92f8c83194567c1c48e9833d"  # v5.15.0
LIBBLASTRAMPOLINE_SHA256 = ""

PATCHES = [
    # fix PREFIX var name
    """
    sed -i "s/prefix/PREFIX/g" src/Make.inc
    """,
    # fix Make.inc path
    """
    sed -i "s/include \\$(LBT_ROOT)\\/src/include \\$(BUILD_TMPDIR)/g" src/Makefile
    """,
    # remove erroring install command (also, no header to copy)
    """
    sed -i "/-@cp -Ra \\$(LBT_ROOT)\\/include\\/*/d" src/Makefile
    """,
    # # fix source path of header install command
    """
    sed -i "s/@cp -a \\$(LBT_ROOT)\\/src\\/libblastrampoline.h/@cp -a \\$(BUILD_TMPDIR)\\/libblastrampoline.h/g" src/Makefile
    """,
    # follow symbolic links
    """
    sed -i "s/@cp/@cp -L/g" src/Makefile
    """,
    # enable echoing commands to ease debugging
    """
    sed -i "s/-@//g" src/Makefile
    """,
    """
    sed -i "s/^@//g" src/Makefile
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
