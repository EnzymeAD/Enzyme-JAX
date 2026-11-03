"""Loads Lean 4 and the libraries its runtime needs."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//third_party/lean:stage1.bzl", "lean_stage1_repo")

def repo():
    # Lean is bootstrapped natively from lean.BUILD: stage0's pregenerated C
    # gives a compiler that turns src/ into stage1, which is what we link.
    http_archive(
        name = "lean4",
        build_file = Label("//third_party/lean:lean.BUILD"),
        sha256 = "6747b099d03d3675f2716e7cf5924a16bfbf75822b13a3d36f51f933ffcc307b",
        strip_prefix = "lean4-4.35.0",
        url = "https://github.com/leanprover/lean4/archive/refs/tags/v4.35.0.tar.gz",
    )

    # Versions match the ones Lean's CMake fetches.
    http_archive(
        name = "lean_libuv",
        build_file = Label("//third_party/lean:libuv.BUILD"),
        sha256 = "8c253adb0f800926a6cbd1c6576abae0bc8eb86a4f891049b72f9e5b7dc58f33",
        strip_prefix = "libuv-1.48.0",
        url = "https://github.com/libuv/libuv/archive/refs/tags/v1.48.0.tar.gz",
    )

    http_archive(
        name = "lean_mimalloc",
        build_file = Label("//third_party/lean:mimalloc.BUILD"),
        sha256 = "19a43af0645c57d348e729d5b31e23e912582911bb1047f795790834d3416221",
        strip_prefix = "mimalloc-3.4.5",
        url = "https://github.com/microsoft/mimalloc/archive/refs/tags/v3.4.5.tar.gz",
    )

    lean_stage1_repo(name = "lean4_stage1")
