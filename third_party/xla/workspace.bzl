"""Loads XLA."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@jax//third_party/xla:workspace.bzl", "XLA_COMMIT", "XLA_SHA256")
load("//:workspace.bzl", "XLA_PATCHES")

def repo():
    http_archive(
        name = "xla",
        sha256 = XLA_SHA256,
        strip_prefix = "xla-" + XLA_COMMIT,
        urls = ["https://github.com/wsmoses/xla/archive/{commit}.tar.gz".format(commit = XLA_COMMIT)],
        patch_cmds = XLA_PATCHES,
    )

    load("@xla//:workspace4.bzl", "xla_workspace4")
    load("@xla//:workspace3.bzl", "xla_workspace3")
    load("@xla//:workspace2.bzl", "xla_workspace2")
    load("@xla//:workspace1.bzl", "xla_workspace1")
    load("@xla//:workspace0.bzl", "xla_workspace0")

    xla_workspace4()
    xla_workspace3()
    xla_workspace2()
    xla_workspace1()
    xla_workspace0()
