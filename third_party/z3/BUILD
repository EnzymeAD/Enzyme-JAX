load("@python_version_repo//:py_version.bzl", "HERMETIC_PYTHON_VERSION")
load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")
load("@rules_license//rules:license.bzl", "license")

package(default_applicable_licenses = [":license"])

license(
    name = "license",
    license_kinds = ["@rules_license//licenses/spdx:MIT"],
    license_text = "LICENSE.txt",
)

exports_files(["LICENSE.txt"])

# The hermetic interpreter exposed by the workspace's python toolchain repos
# (also used as pip_parse's python_interpreter_target), needed by z3's CMake
# configure step and codegen scripts since CI runners don't provide python3.
_HERMETIC_PYTHON = "@python_{}_host//:python".format(HERMETIC_PYTHON_VERSION.replace(".", "_"))

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
)

cmake(
    name = "z3_dynamic",
    data = [
        _HERMETIC_PYTHON,
        "@pypi//standard_pipes:pkg",
    ],
    env = {
        "PYTHONPATH": "$$EXT_BUILD_ROOT$$/external/pypi_standard_pipes/site-packages",
        "Z3_PYTHON3_EXECUTABLE": "$(execpath {})".format(_HERMETIC_PYTHON),
    },
    generate_args = [
        "-G Ninja",
        "-D Z3_EXPORTED_TARGETS=",  # prevents installation, leaving symlinks between dylibs intact on copy
        "-D Python3_EXECUTABLE=$Z3_PYTHON3_EXECUTABLE",
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
    data = [
        _HERMETIC_PYTHON,
        "@pypi//standard_pipes:pkg",
    ],
    env = {
        "PYTHONPATH": "$$EXT_BUILD_ROOT$$/external/pypi_standard_pipes/site-packages",
        "Z3_PYTHON3_EXECUTABLE": "$(execpath {})".format(_HERMETIC_PYTHON),
    },
    generate_args = [
        "-G Ninja",
        "-D BUILD_SHARED_LIBS=OFF",
        "-D Z3_BUILD_LIBZ3_SHARED=OFF",
        "-D Python3_EXECUTABLE=$Z3_PYTHON3_EXECUTABLE",
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
