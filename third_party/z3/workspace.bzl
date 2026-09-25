"""Loads Z3."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo(build_file = "//third_party/z3:BUILD"):
    http_archive(
        name = "z3",
        build_file = build_file,
        strip_prefix = "z3-z3-5.1.0",
        url = "https://github.com/Z3Prover/z3/archive/refs/tags/z3-5.1.0.tar.gz",
    )
