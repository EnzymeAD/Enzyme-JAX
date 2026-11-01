load("@rules_cc//cc:cc_test.bzl", "cc_test")

# C++ unit tests, laid out like the sources under src/enzyme_ad/jax. Each area
# is one gtest binary over a glob, since every binary links the whole compiler
# library.

UNITTEST_DEPS = [
    "//src/enzyme_ad/jax:RegistryUtils",
    "//src/enzyme_ad/jax:XLADerivatives",
    "@com_google_googletest//:gtest_main",
]

cc_test(
    name = "axis_tests",
    srcs = glob(["Dialect/Axis/*Test.cpp"]),
    deps = UNITTEST_DEPS,
)

cc_test(
    name = "distributed_tests",
    srcs = glob(
        [
            "Dialect/Distributed/*Test.cpp",
            "Passes/Distributed/*Test.cpp",
        ],
        exclude = ["Dialect/Distributed/CollectiveDecomposerRegretTest.cpp"],
    ),
    deps = UNITTEST_DEPS,
)

# A regret harness for CollectiveDecomposer's budgeted planning, kept apart
# from the unit tests.
cc_test(
    name = "collective_decomposer_regret_test",
    srcs = ["Dialect/Distributed/CollectiveDecomposerRegretTest.cpp"],
    deps = UNITTEST_DEPS,
)
