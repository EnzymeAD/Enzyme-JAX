import glob
import os
from .primitives import *


def default_nowheel_resource(dn):
    # When running from the bazel output tree (not from the wheel), the clang
    # resource directory lives in the llvm-project external repository.  Its
    # directory name is the canonical repository name, which depends on the
    # Bazel version and on the module that defines it
    # (e.g. `xla+llvm_extension+llvm-project`), so glob for it.
    external = os.path.join(dn, "..", "..", "..", "external")
    candidates = sorted(
        glob.glob(os.path.join(external, "*llvm-project", "clang", "staging"))
    )
    if candidates:
        return candidates[0]
    return os.path.join(external, "llvm-project", "clang", "staging")


def default_linux_cflags():
    return ()
