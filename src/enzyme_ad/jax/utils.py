import os
from .primitives import *


def default_nowheel_resource(dn):
    return os.path.join(
        dn, "..", "..", "..", "external", "llvm-project", "clang", "staging"
    )


def default_linux_cflags():
    return ()
