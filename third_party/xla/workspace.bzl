"""Loads XLA."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@jax//third_party/xla:workspace.bzl", "XLA_COMMIT", "XLA_SHA256")
load("//:workspace.bzl", "XLA_PATCHES")

# XLA_COMMIT = "a76eb715696fd46adef8615769214ea68a297dad"
# XLA_SHA256 = ""

def repo():
    http_archive(
        name = "xla",
        sha256 = XLA_SHA256,
        strip_prefix = "xla-" + XLA_COMMIT,
        urls = ["https://github.com/wsmoses/xla/archive/{commit}.tar.gz".format(commit = XLA_COMMIT)],
        patch_cmds = XLA_PATCHES,
    )
