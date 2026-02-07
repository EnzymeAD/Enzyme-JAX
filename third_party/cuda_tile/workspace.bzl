"""Loads NVIDIA CUDA Tile."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

CUDA_TILE_COMMIT = "8a775693b18303d6c696be6ffd06dadad1b32a8e"  # v13.1.3
CUDA_TILE_SHA256 = ""

def repo(repo_name = ""):
    # When used as an external dependency, repo_name should be "@enzyme_ad"
    # When used standalone, repo_name should be "" (empty string)
    build_file_label = (repo_name + "//third_party/cuda_tile:cuda_tile.BUILD") if repo_name else "//third_party/cuda_tile:cuda_tile.BUILD"
    http_archive(
        name = "cuda_tile",
        sha256 = CUDA_TILE_SHA256,
        strip_prefix = "cuda-tile-" + CUDA_TILE_COMMIT,
        urls = ["https://github.com/NVIDIA/cuda-tile/archive/{commit}.tar.gz".format(commit = CUDA_TILE_COMMIT)],
        build_file = build_file_label,
    )

