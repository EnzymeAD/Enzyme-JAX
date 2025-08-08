"""Loads XLA."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@jax//third_party/xla:revision.bzl", "XLA_COMMIT", "XLA_SHA256")
load("//:workspace.bzl", "XLA_PATCHES")

def repo(extra_patches = [], override_commit = ""):
    commit = XLA_COMMIT
    sha = XLA_SHA256
    if len(override_commit):
        commit = override_commit
        sha = ""
    http_archive(
        name = "xla",
        sha256 = sha,
        strip_prefix = "xla-" + commit,
        urls = ["https://github.com/wsmoses/xla/archive/{commit}.tar.gz".format(commit = commit)],
        patch_cmds = XLA_PATCHES + extra_patches,
    )
