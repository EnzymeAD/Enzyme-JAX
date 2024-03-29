load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "rules_cc",
    sha256 = "85723d827f080c5e927334f1fb18a294c0b3f94fee6d6b45945f5cdae6ea0fd4",
    strip_prefix = "rules_cc-c8c38f8c710cbbf834283e4777916b68261b359c",
    urls = [
        "https://github.com/bazelbuild/rules_cc/archive/c8c38f8c710cbbf834283e4777916b68261b359c.tar.gz",
    ],
)

load("@rules_cc//cc:repositories.bzl", "rules_cc_dependencies")

rules_cc_dependencies()

LLVM_TARGETS = ["X86", "AArch64", "AMDGPU", "NVPTX"]

XLA_COMMIT = "35a563552b55129d3d91965bf5ede0cd2077a63a"
XLA_SHA256 = "a2e089231bb75198268927ba16a4dbfb21f4495818a451706b6c5ef6b2a9e6ea"

http_archive(
    name = "xla",
    sha256 = XLA_SHA256,
    strip_prefix = "xla-" + XLA_COMMIT,
    urls = ["https://github.com/wsmoses/xla/archive/{commit}.tar.gz".format(commit = XLA_COMMIT)],
    patch_cmds = [
        "find . -type f -name BUILD -exec sed -i 's/\\/\\/third_party\\/py\\/enzyme_ad\\/\\.\\.\\./public/g' {} +", 
        "find . -type f -name BUILD -exec sed -i 's/\\/\\/xla\\/mlir\\/memref:friends/\\/\\/visibility:public/g' {} +",
        "find xla/mlir -type f -name BUILD -exec sed -i 's/\\/\\/xla:internal/\\/\\/\\/\\/visibility:public/g' {} +"
    ]
    # patch_args = ["-p1"],
    # patches = ["//:patches/xla.patch", "//:patches/xla2.patch", ],
)

PYRULES_COMMIT = "fe33a4582c37499f3caeb49a07a78fc7948a8949"
PYRULES_SHA256 = "cfa6957832ae0e0c7ee2ccf455a888a291e8419ed8faf45f4420dd7414d5dd96"

http_archive(
    name = "rules_python",
    sha256 = PYRULES_SHA256,
    strip_prefix = "rules_python-" + PYRULES_COMMIT,
    urls = ["https://github.com/bazelbuild/rules_python/archive/{commit}.tar.gz".format(commit = PYRULES_COMMIT)]
)

load("@rules_python//python:repositories.bzl", "py_repositories")

py_repositories()

load("@rules_python//python/pip_install:repositories.bzl", "pip_install_dependencies")

pip_install_dependencies()

ENZYME_COMMIT = "9acbc0a667ec8ae76407b5708758667a65ff15aa"
ENZYME_SHA256 = "287143133ccf9501a02f1bdab351c34adcab3bbfc8648b180ebd79d0e058b3af"

http_archive(
    name = "enzyme",
    sha256 = ENZYME_SHA256,
    strip_prefix = "Enzyme-" + ENZYME_COMMIT + "/enzyme",
    urls = ["https://github.com/EnzymeAD/Enzyme/archive/{commit}.tar.gz".format(commit = ENZYME_COMMIT)],
)

JAX_COMMIT = "8815b236b656f494171131301d1d81e84cf4c67c"
JAX_SHA256 = "188787b8ec366dcda5805f24dcbfab7a349aa780498aeb9c3b728c9cec0a7e7d"

http_archive(
    name = "jax",
    sha256 = JAX_SHA256,
    strip_prefix = "jax-" + JAX_COMMIT,
    urls = ["https://github.com/google/jax/archive/{commit}.tar.gz".format(commit = JAX_COMMIT)],
    patch_args = ["-p1"],
    patches = ["//:patches/jax.patch"],
)

load("@jax//third_party/xla:workspace.bzl", jax_xla_workspace = "repo")
jax_xla_workspace()

load("@xla//:workspace4.bzl", "xla_workspace4")
xla_workspace4()

load("@xla//:workspace3.bzl", "xla_workspace3")
xla_workspace3()

load("@xla//:workspace2.bzl", "xla_workspace2")

load("@llvm-raw//utils/bazel:configure.bzl", "llvm_configure")
llvm_configure(name = "llvm-project", targets = LLVM_TARGETS)
xla_workspace2()

load("@xla//:workspace1.bzl", "xla_workspace1")
xla_workspace1()

load("@xla//:workspace0.bzl", "xla_workspace0")
xla_workspace0()

load("@jax//third_party/flatbuffers:workspace.bzl", flatbuffers = "repo")
flatbuffers()
