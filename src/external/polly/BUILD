load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_library(
    name = "Polly",
    hdrs = glob([
        "**/*.h",
    ]),
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
    # copts = [
    #     "-I/isl/include",
    # ],
    deps = [
        "//src/external/isl:Isl",
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Passes",
        "@llvm-project//llvm:Support",
    ],
)
