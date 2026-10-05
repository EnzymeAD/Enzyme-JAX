"""Loads XLA."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@jax_deps//:deps.bzl", "XLA_COMMIT", "XLA_INTEGRITY")
load("//:workspace.bzl", "XLA_PATCHES")
load("//third_party/grpc:workspace.bzl", "xla_grpc_repository")

def repo(extra_patches = [], override_commit = ""):
    commit = XLA_COMMIT
    integrity = XLA_INTEGRITY
    if len(override_commit):
        commit = override_commit
        integrity = ""
    http_archive(
        name = "xla",
        integrity = integrity,
        type = "tar.gz",
        strip_prefix = "xla-{commit}".format(commit = commit),
        urls = ["https://github.com/openxla/xla/archive/{commit}.tar.gz".format(commit = commit)],
        patch_cmds = XLA_PATCHES + extra_patches,
        patches = ["//:patches/xla.patch", "//:patches/xla_win.patch", "//:patches/xla_trainium.patch"],
        patch_args = ["-p1"],
    )

    # XLA's workspace chain reaches google-cloud-cpp, which defines a gRPC of
    # its own under this name if nothing else has, too old to build against
    # this workspace's protobuf. Get in first, with the gRPC XLA itself uses.
    xla_grpc_repository(name = "com_github_grpc_grpc")
