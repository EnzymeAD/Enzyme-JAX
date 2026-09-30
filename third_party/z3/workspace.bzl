"""Loads Z3."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")

def repo():
    rules_foreign_cc_dependencies()
    http_archive(
        name = "z3",
        strip_prefix = "z3-z3-5.1.0",
        url = "https://github.com/Z3Prover/z3/archive/refs/tags/z3-5.1.0.tar.gz",
    )
