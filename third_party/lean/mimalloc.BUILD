# mimalloc for the Lean runtime.
#
# Lean does not link mimalloc as a library: its runtime/mimalloc.cpp
# #includes all of mimalloc (src/static.c) so the allocation fast path inlines
# into Lean's allocator. This only exposes the sources for that, plus
# mimalloc.h under the <lean/mimalloc.h> name lean.h uses.

licenses(["notice"])

exports_files(["LICENSE"])

cc_library(
    name = "static_c",
    includes = [
        "include",
        "src",
    ],
    textual_hdrs = glob([
        "include/**/*.h",
        "src/**/*.c",
        "src/**/*.h",
    ]),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "lean_mimalloc_h",
    hdrs = ["include/mimalloc.h"],
    include_prefix = "lean",
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)
