"""Module extension providing the Enzyme and cuda-tile repositories.

Defined as a module extension (rather than `use_repo_rule`) so that every
module using it, root or not, shares the very same `@enzyme` and `@cuda_tile`
repositories, and so that the pinned commits can live in a loadable
`workspace.bzl` (MODULE.bazel files cannot `load()`).
"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:local.bzl", "local_repository")
load(
    "//:workspace.bzl",
    "CUDA_TILE_COMMIT",
    "CUDA_TILE_SHA256",
    "CUTILE_PATCHES",
    "ENZYME_COMMIT",
    "ENZYME_SHA256",
    "OVERRIDE_ENZYME_PATH",
)

def _enzyme_deps_impl(_mctx):
    if len(OVERRIDE_ENZYME_PATH) != 0:
        local_repository(
            name = "enzyme",
            path = OVERRIDE_ENZYME_PATH,
        )
    else:
        http_archive(
            name = "enzyme",
            sha256 = ENZYME_SHA256,
            strip_prefix = "Enzyme-" + ENZYME_COMMIT + "/enzyme",
            urls = ["https://github.com/EnzymeAD/Enzyme/archive/{commit}.tar.gz".format(commit = ENZYME_COMMIT)],
        )

    http_archive(
        name = "cuda_tile",
        build_file = Label("//third_party/cuda_tile:cuda_tile.BUILD"),
        patch_cmds = CUTILE_PATCHES,
        sha256 = CUDA_TILE_SHA256,
        strip_prefix = "cuda-tile-" + CUDA_TILE_COMMIT,
        urls = ["https://github.com/NVIDIA/cuda-tile/archive/{commit}.tar.gz".format(commit = CUDA_TILE_COMMIT)],
    )

enzyme_deps = module_extension(
    implementation = _enzyme_deps_impl,
)
