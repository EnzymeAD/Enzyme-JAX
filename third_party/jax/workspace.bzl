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
    """
    sed -i.bak0 "s/#include \\"jaxlib\\/mosaic\\/dialect\\/tpu\\/tpu_enums.h.inc/#undef ARG_MAX\\n#include \\"jaxlib\\/mosaic\\/dialect\\/tpu\\/tpu_enums.h.inc/g" jaxlib/mosaic/dialect/tpu/tpu_dialect.h
    """,
    """
    sed -i.bak0 "s/if (is_batched_syev_supported) {/if constexpr (false) {/g" jaxlib/gpu/solver_kernels_ffi.cc
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
