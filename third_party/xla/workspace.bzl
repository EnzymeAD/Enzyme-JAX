"""Loads XLA."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@jax//third_party/xla:revision.bzl", "XLA_COMMIT", "XLA_SHA256")
load("//:workspace.bzl", "XLA_PATCHES")

def repo(extra_patches = [], override_commit = ""):
    commit = XLA_COMMIT
    sha = XLA_SHA256
    commit = "bdb1261d41f99e2513ca137776a0d1f2a72cf552"
    sha = "feffbcdd9ba490b93f95d78f6cb260f7cb1abee6e6f5c5bf687d6eaaf597bdc6"
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
        patches = ["//:patches/xla.patch"],
        patch_args = ["-p1"],
    )
