load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")
load("@rules_license//rules:license.bzl", "license")

package(default_applicable_licenses = [":license"])

license(
    name = "license",
    license_kinds = ["@rules_license//licenses/spdx:MIT"],
    license_text = "LICENSE.txt",
)

exports_files(["LICENSE.txt"])

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
)

cmake(
    name = "z3_dynamic",
    env = {
        "LD_LIBRARY_PATH": "/usr/lib/csl-musl-x86_64:/usr/lib/csl-glibc-x86_64",
    },
    generate_args = [
        "-G Ninja",
    ],
    lib_source = ":all_files",
    out_binaries = ["z3"],
    out_include_dir = "include",
    out_shared_libs = select({
        "@platforms//os:linux": ["libz3.so"],
        # "@platforms//os:osx": ["libz3.dylib"],    # FIXME: this is not working, libz3<version>.dylib is not copied
        "@platforms//os:windows": ["libz3.dll"],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    visibility = ["//visibility:public"],
)

cmake(
    name = "z3_static_cmake",
    env = {
        "LD_LIBRARY_PATH": "/usr/lib/csl-musl-x86_64:/usr/lib/csl-glibc-x86_64",
    },
    generate_args = [
        "-G Ninja",
    ],
    lib_source = ":all_files",
    out_binaries = ["z3"],
    out_include_dir = "include",
    out_static_libs = select({
        "@platforms//os:linux": ["libz3.a"],
        "@platforms//os:osx": ["libz3.a"],
        "@platforms//os:windows": ["libz3.lib"],  # MSVC with Control Flow Guard enabled by default
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "z3_static",
    hdrs = glob([
        "src/api/*.h",
        "src/api/*.hpp",
        "src/api/c++/*.h",
        "src/api/c++/*.hpp",
    ]),
    include_prefix = "",
    includes = [
        "src/api",
        "src/api/c++",
    ],
    strip_include_prefix = "src/api",
    visibility = ["//visibility:public"],
    deps = [":z3_static_cmake"],
)

alias(
    name = "z3",
    actual = ":z3_static",
    visibility = ["//visibility:public"],
)
