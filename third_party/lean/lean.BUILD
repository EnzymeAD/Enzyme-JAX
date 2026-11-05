# Native Bazel bootstrap of Lean 4, replacing its CMake build.
#
# Lean's CMake build has two stages. stage0/ holds a copy of the C++ sources
# and the C that an earlier Lean emitted for the Lean-written compiler and
# standard library; building it gives a working `lean`. That `lean` then
# compiles src/'s Lean sources to C, which together with src/'s C++ sources
# is stage1, the Lean that we link. When bumping Lean, diff the flags below
# against src/CMakeLists.txt.

licenses(["notice"])

exports_files(["LICENSE"])

# Compiled by @lean4_stage1 (see stage1.bzl).
exports_files(glob(["src/**/*.lean"]))

################################################################################
# Generated headers
################################################################################

_VERSION = "4.35.0"

# Commit of the v4.35.0 tag.
_GITHASH = "eb6cafb7290e8bd5f2937522d8d804f58f02bafc"

[
    config_setting(
        name = "{}_{}".format(os, cpu),
        constraint_values = [
            "@platforms//os:" + os,
            "@platforms//cpu:" + cpu,
        ],
    )
    for os in [
        "linux",
        "macos",
    ]
    for cpu in [
        "aarch64",
        "x86_64",
    ]
]

# What `clang --print-target-triple` reports, as CMake records it.
_PLATFORM_TARGET = {
    ":linux_aarch64": "aarch64-unknown-linux-gnu",
    ":linux_x86_64": "x86_64-unknown-linux-gnu",
    ":macos_aarch64": "arm64-apple-darwin",
    ":macos_x86_64": "x86_64-apple-darwin",
}

# include/lean/{config,version}.h and githash.h for each stage, where CMake
# puts them. Only config.h differs between the stages, in LEAN_IS_STAGE0.
[
    genrule(
        name = stage + "_generated_headers",
        srcs = [
            src + "/config.h.in",
            src + "/githash.h.in",
            src + "/version.h.in",
        ],
        outs = [
            stage + "/generated/githash.h",
            stage + "/generated/include/lean/config.h",
            stage + "/generated/include/lean/version.h",
        ],
        cmd = select({
            setting: "PLATFORM_TARGET=" + target
            for setting, target in _PLATFORM_TARGET.items()
        }) + """
d=$(RULEDIR)/{stage}/generated/include/lean
sed -e 's|@LEAN_MIMALLOC@|#define LEAN_MIMALLOC|' \\
    -e 's|@LEAN_IS_STAGE0@|#define LEAN_IS_STAGE0 {is_stage0}|' \\
    $(location {src}/config.h.in) > $$d/config.h
sed -e 's|@GIT_SHA1@|{githash}|' $(location {src}/githash.h.in) \\
    > $(RULEDIR)/{stage}/generated/githash.h
sed -e 's|@LEAN_VERSION_MAJOR@|{major}|' \\
    -e 's|@LEAN_VERSION_MINOR@|{minor}|' \\
    -e 's|@LEAN_VERSION_PATCH@|{patch}|' \\
    -e 's|@LEAN_VERSION_IS_RELEASE@|1|' \\
    -e 's|@LEAN_SPECIAL_VERSION_DESC@||' \\
    -e 's|@LEAN_VERSION_STRING@|{version}|' \\
    -e 's|@LEAN_PLATFORM_TARGET@|'"$$PLATFORM_TARGET"'|' \\
    -e 's|@LEAN_MANUAL_ROOT@|https://lean-lang.org/doc/reference/{version}/|' \\
    $(location {src}/version.h.in) > $$d/version.h
""".format(
            githash = _GITHASH,
            is_stage0 = 1 if stage == "stage0" else 0,
            major = _VERSION.split(".")[0],
            minor = _VERSION.split(".")[1],
            patch = _VERSION.split(".")[2],
            src = src,
            stage = stage,
            version = _VERSION,
        ),
    )
    for stage, src in [
        ("stage0", "stage0/src"),
        ("stage1", "src"),
    ]
]

# util/ffi.cpp reports the flags `leanc` compiles and links with. We never
# compile Lean code through `leanc`, so they are left empty.
[
    genrule(
        name = stage + "_ffi_cpp",
        srcs = [src + "/util/ffi.cpp"],
        outs = [stage + "/generated/util/ffi.cpp"],
        cmd = "sed -e 's|@[A-Z_]*@||g' $< > $@",
    )
    for stage, src in [
        ("stage0", "stage0/src"),
        ("stage1", "src"),
    ]
]

################################################################################
# C++ libraries
################################################################################

# Flags CMake adds for Release builds on Linux and macOS.
_LEAN_COPTS = [
    # This must win over any -std= passed via --cxxopt.
    "-std=c++20",
    "-ffp-contract=off",
    "-ftls-model=initial-exec",
    "-fvisibility=hidden",
    "-fvisibility-inlines-hidden",
    "-w",
]

_LEAN_DEFINES = [
    "LEAN_BUILD_TYPE=Release",
    "LEAN_EXPORTING",
    "LEAN_MMAP",
    "LEAN_MULTI_THREAD",
]

# The C++ libraries and the C emitted for the Lean-written ones call each
# other, so CMake links them inside --start-group. We instead link every
# object (alwayslink), which the IR interpreter needs anyway: it looks up
# @[extern] implementations by symbol name at runtime.
[
    [
        cc_library(
            name = stage + "_headers",
            hdrs = glob([src + "/**/*.h"]) + [
                ":{}/generated/githash.h".format(stage),
                ":{}/generated/include/lean/config.h".format(stage),
                ":{}/generated/include/lean/version.h".format(stage),
            ],
            includes = [
                src,
                src + "/include",
                stage + "/generated",
                stage + "/generated/include",
            ],
            deps = ["@lean_mimalloc//:lean_mimalloc_h"],
        ),
        # runtime/mimalloc.cpp compiles all of mimalloc, so it gets mimalloc's
        # flags (MIMALLOC_OPTS in src/runtime/CMakeLists.txt).
        cc_library(
            name = stage + "_mimalloc",
            srcs = [src + "/runtime/mimalloc.cpp"],
            copts = _LEAN_COPTS + ["-O3"],
            local_defines = _LEAN_DEFINES + [
                "MI_MAX_ALIGN_SIZE=8",
                "MI_SECURE=0",
                "MI_SHARED_LIB",
                "MI_SHARED_LIB_EXPORT",
                "MI_WIN_NOREDIRECT",
                "NDEBUG",
            ],
            alwayslink = True,
            deps = [
                ":{}_headers".format(stage),
                "@lean_mimalloc//:static_c",
            ],
        ),
        cc_library(
            name = stage + "_leanrt",
            srcs = glob(
                [
                    src + "/runtime/*.cpp",
                    src + "/runtime/uv/*.cpp",
                ],
                exclude = [src + "/runtime/mimalloc.cpp"],
            ),
            copts = _LEAN_COPTS,
            linkopts = ["-pthread"] + select({
                "@platforms//os:macos": [],
                "//conditions:default": ["-ldl"],
            }),
            local_defines = _LEAN_DEFINES,
            alwayslink = True,
            deps = [
                ":{}_headers".format(stage),
                ":{}_mimalloc".format(stage),
                "@boringssl//:ssl",
                "@lean_libuv//:libuv",
            ],
        ),
        cc_library(
            name = stage + "_leancpp",
            srcs = glob(
                [
                    src + "/kernel/*.cpp",
                    src + "/library/*.cpp",
                    src + "/library/constructions/*.cpp",
                    src + "/util/*.cpp",
                ],
                exclude = [
                    # Part of the `lean` executable.
                    src + "/util/shell.cpp",
                    src + "/util/ffi.cpp",
                ],
            ) + [":{}/generated/util/ffi.cpp".format(stage)],
            copts = _LEAN_COPTS,
            local_defines = _LEAN_DEFINES,
            alwayslink = True,
            deps = [
                ":{}_headers".format(stage),
                ":{}_leanrt".format(stage),
            ],
        ),
        cc_library(
            name = stage + "_initialize",
            srcs = [src + "/initialize/init.cpp"],
            copts = _LEAN_COPTS,
            local_defines = _LEAN_DEFINES,
            alwayslink = True,
            deps = [":{}_leancpp".format(stage)],
        ),
    ]
    for stage, src in [
        ("stage0", "stage0/src"),
        ("stage1", "src"),
    ]
]

################################################################################
# stage0 `lean`
################################################################################

# Flags `leanc` compiles Lean's emitted C with (LEANC_EXTRA_CC_FLAGS in
# src/CMakeLists.txt); LEAN_EXPORTING as for the objects of libleanshared.
_LEANC_COPTS = [
    "-ffp-contract=off",
    "-fstack-clash-protection",
    "-fvisibility=hidden",
    "-fdata-sections",
    "-ffunction-sections",
    "-w",
]

# Init, Std and Lean are what `lean` needs; Lake and the other tools are
# not built.
cc_library(
    name = "stage0_stdlib",
    srcs = glob([
        "stage0/stdlib/Init.c",
        "stage0/stdlib/Init/**/*.c",
        "stage0/stdlib/Lean.c",
        "stage0/stdlib/Lean/**/*.c",
        "stage0/stdlib/Std.c",
        "stage0/stdlib/Std/**/*.c",
    ]),
    copts = _LEANC_COPTS,
    local_defines = ["LEAN_EXPORTING"],
    alwayslink = True,
    deps = [":stage0_headers"],
)

cc_library(
    name = "stage0_leanshell",
    srcs = ["stage0/src/util/shell.cpp"],
    copts = _LEAN_COPTS,
    local_defines = _LEAN_DEFINES,
    alwayslink = True,
    deps = [
        ":stage0_initialize",
        ":stage0_stdlib",
    ],
)

# Runs in the exec configuration when compiling src/, so it is optimized even
# in fastbuild.
cc_binary(
    name = "stage0_lean",
    srcs = ["stage0/src/shell/lean.cpp"],
    copts = _LEAN_COPTS,
    # The IR interpreter finds @[extern] implementations with dlsym.
    linkopts = select({
        "@platforms//os:macos": [],
        "//conditions:default": ["-rdynamic"],
    }),
    visibility = ["@lean4_stage1//:__subpackages__"],
    deps = [":stage0_leanshell"],
)

################################################################################
# stage1 libraries
################################################################################

# The C that :stage0_lean emitted for src/ (see stage1.bzl).
cc_library(
    name = "stage1_stdlib",
    srcs = ["@lean4_stage1//:c_srcs"],
    copts = _LEANC_COPTS,
    local_defines = ["LEAN_EXPORTING"],
    alwayslink = True,
    deps = [":stage1_headers"],
)

# The contents of libleanshared, minus the `lean` command line driver
# (util/shell.cpp).
cc_library(
    name = "leanshared_static",
    visibility = ["//visibility:public"],
    deps = [
        ":stage1_initialize",
        ":stage1_stdlib",
    ],
)
