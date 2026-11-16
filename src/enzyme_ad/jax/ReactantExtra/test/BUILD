load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    default_applicable_licenses = [],
    default_visibility = ["//:__subpackages__"],
)

licenses(["notice"])

# Directories of libReactantExtra.so in the runfiles, relative to the working
# directory of a test, when this is the main repository and when it is the
# external repository @enzyme_ad. The dynamic linker has to find the library
# there to resolve the shared libraries the library itself depends on.
LIBREACTANTEXTRA_DIRS = "src/enzyme_ad/jax/ReactantExtra:../enzyme_ad/src/enzyme_ad/jax/ReactantExtra"

cc_test(
    name = "address_reservation_test",
    srcs = [
        "address_reservation_test.cpp",
        "//src/enzyme_ad/jax/ReactantExtra:libReactantExtra.so",
    ],
    data = ["//src/enzyme_ad/jax/ReactantExtra:libReactantExtra.so"],
    env = {"LD_LIBRARY_PATH": LIBREACTANTEXTRA_DIRS},
    linkstatic = True,
    deps = ["@com_google_googletest//:gtest_main"],
)

cc_test(
    name = "linkable_runtime_test",
    srcs = [
        "linkable_runtime_test.cpp",
        "reactant_extra_test.h",
        "//src/enzyme_ad/jax/ReactantExtra:libReactantExtra.so",
    ],
    data = ["//src/enzyme_ad/jax/ReactantExtra:libReactantExtra.so"],
    env = {"LD_LIBRARY_PATH": LIBREACTANTEXTRA_DIRS},
    linkstatic = True,
    deps = ["@com_google_googletest//:gtest_main"],
)

cc_test(
    name = "pjrt_ifrt_cpu_test",
    srcs = [
        "pjrt_ifrt_cpu_test.cpp",
        "reactant_extra_test.h",
        "//src/enzyme_ad/jax/ReactantExtra:libReactantExtra.so",
    ],
    data = ["//src/enzyme_ad/jax/ReactantExtra:libReactantExtra.so"],
    env = {"LD_LIBRARY_PATH": LIBREACTANTEXTRA_DIRS},
    linkstatic = True,
    deps = ["@com_google_googletest//:gtest_main"],
)

cc_test(
    name = "xla_ffi_test",
    srcs = [
        "reactant_extra_test.h",
        "xla_ffi_test.cpp",
        "//src/enzyme_ad/jax/ReactantExtra:libReactantExtra.so",
    ],
    data = ["//src/enzyme_ad/jax/ReactantExtra:libReactantExtra.so"],
    env = {"LD_LIBRARY_PATH": LIBREACTANTEXTRA_DIRS},
    linkstatic = True,
    deps = ["@com_google_googletest//:gtest_main"],
)

cc_test(
    name = "mlir_hlo_utils_test",
    srcs = [
        "mlir_hlo_utils_test.cpp",
        "reactant_extra_test.h",
        "//src/enzyme_ad/jax/ReactantExtra:libReactantExtra.so",
    ],
    data = ["//src/enzyme_ad/jax/ReactantExtra:libReactantExtra.so"],
    env = {"LD_LIBRARY_PATH": LIBREACTANTEXTRA_DIRS},
    linkstatic = True,
    deps = ["@com_google_googletest//:gtest_main"],
)

cc_test(
    name = "dlopen_abi_test",
    srcs = ["dlopen_abi_test.cpp"],
    data = ["//src/enzyme_ad/jax/ReactantExtra:libReactantExtra.so"],
    linkopts = ["-ldl"],
    linkstatic = True,
    deps = [
        "@com_google_absl//absl/debugging:leak_check",
        "@com_google_googletest//:gtest_main",
    ],
)
