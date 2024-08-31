load("@rules_cc//cc:defs.bzl", "cc_library")

def rust_cxx_bridge(name, src, deps = []):
    native.alias(
        name = "%s/header" % name,
        actual = src + ".h",
    )

    native.alias(
        name = "%s/source" % name,
        actual = src + ".cc",
    )

    native.genrule(
        name = "%s/generated" % name,
        srcs = [src],
        outs = [
            src + ".h",
            src + ".cc",
        ],
        cmd = "cxxbridge $(location %s) -o $(location %s.h) -o $(location %s.cc)" % (src, src, src),
    )

    cc_library(
        name = "%s/include" % name,
        hdrs = [src + ".h"],
        include_prefix = "cxxbridge",
        alwayslink = True,
        linkstatic = True,
    )

    cc_library(
        name = name,
        srcs = [src + ".cc"],
        deps = deps + [":%s/include" % name],
        alwayslink = True,
        linkstatic = True,
    )
