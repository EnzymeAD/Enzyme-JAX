"""Loads JAX."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//:workspace.bzl", "JAX_COMMIT", "JAX_SHA256")

JAX_PATCHES = [
    """
    sed -i.bak0 "s/\\/\\/jaxlib\\/.../public/g" jaxlib/symlink_files.bzl
    """,
    """
    sed -i.bak0 "s/jax\\/experimental:mosaic_users/visibility:public/g" jaxlib/mosaic/BUILD
    """,
]

def repo(extra_patches = [], override_commit = ""):
    commit = JAX_COMMIT
    sha = JAX_SHA256
    if len(override_commit):
        commit = override_commit
        sha = ""
    http_archive(
        name = "jax",
        sha256 = sha,
        strip_prefix = "jax-" + commit,
        urls = ["https://github.com/google/jax/archive/{commit}.tar.gz".format(commit = commit)],
        patch_cmds = JAX_PATCHES + extra_patches,
    )
