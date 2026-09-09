load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_library(
    name = "jit",
    srcs = ["jit/jit.cpp"],
    hdrs = ["jit/jit.h"],
    copts = [
        "-Wno-implicit-fallthrough",
        "-Wno-non-virtual-dtor",
        "-Wno-pessimizing-move",
        "-Wno-unused-result",
        "-Wno-unused-variable",
        "-Wno-vla-cxx-extension",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@llvm-project//llvm:ExecutionEngine",
        "@llvm-project//llvm:OrcJIT",
        "@llvm-project//llvm:OrcTargetProcess",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:ToLLVMIRTranslation",
    ],
)
