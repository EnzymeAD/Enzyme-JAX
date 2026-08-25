"""Loads gRPC as @com_github_grpc_grpc."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//:workspace.bzl", "GRPC_PATCHES")

# Keep in sync with the version used by @xla (workspace2.bzl).
GRPC_VERSION = "1.81.0"
GRPC_SHA256 = "41b695614b26652ff9e97ce50cfd4a6c7a3d45a9fe598d1454407746499bbf2c"

def repo():
    """Defines @com_github_grpc_grpc, the canonical legacy name for gRPC.

    XLA now fetches gRPC as @grpc, but many WORKSPACE-mode consumers (including
    transitive loads through xla_workspace1) still reference the archive under
    the old name @com_github_grpc_grpc.  Defining it here — using the same
    1.81.0 tarball that XLA uses for @grpc — ensures a modern version with all
    known fixes (including the missing #include <algorithm> in
    src/core/util/glob.cc) is used instead of whatever stale copy might
    otherwise be pulled in.

    Uses native.existing_rules() so that an earlier explicit definition in the
    consuming WORKSPACE wins.
    """
    if "com_github_grpc_grpc" not in native.existing_rules():
        http_archive(
            name = "com_github_grpc_grpc",
            sha256 = GRPC_SHA256,
            strip_prefix = "grpc-" + GRPC_VERSION,
            urls = ["https://github.com/grpc/grpc/archive/refs/tags/v{version}.tar.gz".format(version = GRPC_VERSION)],
            patch_cmds = GRPC_PATCHES,
        )
