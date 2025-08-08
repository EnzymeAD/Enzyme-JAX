"""Loads JAX."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//:workspace.bzl", "JAX_COMMIT", "JAX_SHA256")

JAX_PATCHES = [
    """
    sed -i.bak0 "s/\\/\\/jaxlib\\/.../public/g" jaxlib/symlink_files.bzl
    """,
    """
    sed -i.bak0 "s/jaxlib\\/experimental:mosaic_users/visibility:public/g" jaxlib/mosaic/BUILD
    """,
]

def repo():
    http_archive(
        name = "jax",
        sha256 = JAX_SHA256,
        strip_prefix = "jax-" + JAX_COMMIT,
        urls = ["https://github.com/google/jax/archive/{commit}.tar.gz".format(commit = JAX_COMMIT)],
        patch_cmds = JAX_PATCHES,
    )
