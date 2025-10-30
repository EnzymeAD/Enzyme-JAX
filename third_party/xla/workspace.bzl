"""Loads XLA."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@jax//third_party/xla:revision.bzl", "XLA_COMMIT", "XLA_SHA256")
load("//:workspace.bzl", "XLA_PATCHES")

def repo(extra_patches = [], override_commit = ""):
    override_commit = "b13ea16e1dad3b1175d24d5694b02af27dce17f5"
    commit = XLA_COMMIT
    sha = XLA_SHA256
    if len(override_commit):
        commit = override_commit
        sha = ""
    http_archive(
        name = "xla",
        sha256 = sha,
        type = "tar.gz",
        strip_prefix = "openxla-xla-{commit}".format(commit = commit[:7]),
        urls = ["https://api.github.com/repos/openxla/xla/tarball/{commit}".format(commit = commit)],
        patch_cmds = XLA_PATCHES + extra_patches,
    )
