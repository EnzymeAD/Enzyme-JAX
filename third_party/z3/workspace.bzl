"""Loads Z3."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")

def prereqs():
    http_archive(
        name = "bazel_features_1_47",
        # 1.47.1 is the floor required by rules_cc_autoconf (used by rules_foreign_cc
        # 0.16.0's pkgconf toolchain); matches rules_foreign_cc's own pin.
        sha256 = "6a727a78c0134b1b912c97c0937e1c956f35775934ae3e1f4af4156f8d5d1ff4",
        strip_prefix = "bazel_features-1.47.1",
        url = "https://github.com/bazel-contrib/bazel_features/releases/download/v1.47.1/bazel_features-v1.47.1.tar.gz",
    )
    # xla_workspace0() includes benchmark (specifically benchmark_deps.bzl).
    # at the version pinned in xla, benchmark in turn includes a version of rules_foreign_cc that sets up a very old ninja version
    # this ninja version includes a config script which depends on the python "pipes" module, which was deprecated after 3.11
    # this causes builds to fail (especially on macOS)

    # these rules ensure that rules_foreign_cc builds before xla_workspace0(), and does so with dependencies sufficiently recent to compile z3 via ninja
    # see: https://github.com/google/benchmark/blob/754ef08ab91767be54f56e8de3f00527aef3f779/bazel/benchmark_deps.bzl#L21
    # https://github.com/openxla/xla/blob/3a6a82438d93c3d1bc3709a9603275d51af9026e/workspace0.bzl#L23
    # and https://github.com/ninja-build/ninja/blob/b84b3501c63042e72b05c90c76d75e0381daa4cf/configure.py#L24

    http_archive(
        name = "bazel_lib_3_2_0",
        sha256 = "e733937de2f542436c5d3d618e22c638489b40dfd251284050357babe71103d7",
        strip_prefix = "bazel-lib-3.2.0",
        url = "https://github.com/bazel-contrib/bazel-lib/releases/download/v3.2.0/bazel-lib-v3.2.0.tar.gz",
    )

    http_archive(
        name = "rules_foreign_cc",
        repo_mapping = {
            "@bazel_lib": "@bazel_lib_3_2_0",
            "@bazel_features": "@bazel_features_1_47",
        },
        sha256 = "327b3fcacde97b9665424db2b6c37e6f8da59ecc783dc5b8683c69396f820a12",
        strip_prefix = "rules_foreign_cc-0.16.0",
        url = "https://github.com/bazel-contrib/rules_foreign_cc/releases/download/0.16.0/rules_foreign_cc-0.16.0.tar.gz",
    )

def repo(build_file = "//third_party/z3:BUILD"):
    prereqs()

    # Use the prebuilt cmake/ninja toolchains only: the source-built toolchains
    # (pkgconf/m4/make via rules_cc_autoconf) require
    # --incompatible_enable_cc_toolchain_resolution, which this build disables.
    rules_foreign_cc_dependencies(register_built_tools = False)
    http_archive(
        name = "z3",
        build_file = build_file,
        strip_prefix = "z3-z3-5.1.0",
        url = "https://github.com/Z3Prover/z3/archive/refs/tags/z3-5.1.0.tar.gz",
    )
