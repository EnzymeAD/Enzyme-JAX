cc_library(
    name = "Polly",
    hdrs = glob([
         "**/*.h"
    ]),
    strip_include_prefix = "include",
    # copts = [
    #     "-I/isl/include",
    # ],
    deps = [
        "//src/external/isl:Isl",
    ],
    visibility = ["//visibility:public"],
)
