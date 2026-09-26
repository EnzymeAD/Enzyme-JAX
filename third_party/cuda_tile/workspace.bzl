"""Loads NVIDIA CUDA Tile."""

# XLA's workspace2() declares @cuda_tile itself, with a BUILD file and the
# patch series NVIDIA's sources need -- the archive ships no BUILD of its own.
#
# In WORKSPACE mode the first declaration of a repository wins, so declaring it
# here as well does not add anything; it only decides which definition is used,
# according to where a workspace happens to call this macro relative to
# xla_workspace2().  Enzyme-JAX calls it after (WORKSPACE:355 vs :192) and gets
# XLA's, while Reactant calls it before (WORKSPACE:140 vs :198) and got this
# one, which listed cuda_tile's sources by hand and so went stale whenever a
# release added a file -- as 13.4.0 did, breaking every Reactant_jll platform
# on a header it does not name.
#
# Leave the repository to XLA, which keeps it current.  This stays a macro
# because Reactant's WORKSPACE loads and calls it by name.

def repo(repo_name = ""):  # @unused
    pass
