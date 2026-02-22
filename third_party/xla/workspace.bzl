"""Loads XLA."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@jax//third_party/xla:revision.bzl", "XLA_COMMIT", "XLA_SHA256")
load("//:workspace.bzl", "XLA_PATCHES")

def repo(extra_patches = [], override_commit = ""):
    commit = XLA_COMMIT
    sha = XLA_SHA256
    override_commit = "a6a6c1e163e1012693a2acaae973319f29e81be4"
    if len(override_commit):
        commit = override_commit
        sha = ""
    http_archive(
        name = "xla",
        sha256 = sha,
        type = "tar.gz",
        strip_prefix = "xla-{commit}".format(commit = commit),
        urls = ["https://github.com/openxla/xla/archive/{commit}.tar.gz".format(commit = commit)],
        patch_cmds = XLA_PATCHES + extra_patches,
        patches = ["//:patches/xla.patch", "//:patches/xla_spmd.patch", "//:patches/xla_tile_assignment.patch"],
        patch_args = ["-p1"],
    )
