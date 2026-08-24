# Commits of the dependencies fetched by //third_party:extensions.bzl.
# The JAX / XLA / LLVM stack is pinned in xla_deps.MODULE.bazel instead.

ENZYME_COMMIT = "25933271ffb8a475d66766e0c75929e81d4df432"
ENZYME_SHA256 = ""

# If the empty string this will automatically use the commit above
# otherwise this should be a path to the folder containing the BUILD file for enzyme
# (alternatively pass --override_repository=enzyme=/path/to/Enzyme/enzyme).
OVERRIDE_ENZYME_PATH = ""

CUDA_TILE_COMMIT = "0c5ec1c5b72889d58b03cf43970984747680588c"
CUDA_TILE_SHA256 = ""

CUTILE_PATCHES = [
    """sed -i.bak "/usePropertiesForAttributes/d" include/cuda_tile/Dialect/CudaTile/IR/Dialect.td""",
]
