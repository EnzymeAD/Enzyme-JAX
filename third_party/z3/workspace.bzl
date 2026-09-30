"""Loads Z3."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    # z3 ships a BUILD.bazel that drives its CMake build through
    # rules_foreign_cc. That does not cross compile well (prebuilt ninja binary,
    # MSVC-only output names, ...), so we build libz3 natively from z3.BUILD.
    http_archive(
        name = "z3",
        build_file = Label("//third_party/z3:z3.BUILD"),
        sha256 = "c433e1add0431c5edf1644bd9951c40588024d2d288f0e4215e5fcb6e3b4277d",
        strip_prefix = "z3-z3-5.1.0",
        url = "https://github.com/Z3Prover/z3/archive/refs/tags/z3-5.1.0.tar.gz",
    )
