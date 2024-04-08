JAX_COMMIT = "8815b236b656f494171131301d1d81e84cf4c67c"
JAX_SHA256 = "188787b8ec366dcda5805f24dcbfab7a349aa780498aeb9c3b728c9cec0a7e7d"

ENZYME_COMMIT = "9acbc0a667ec8ae76407b5708758667a65ff15aa"
ENZYME_SHA256 = "287143133ccf9501a02f1bdab351c34adcab3bbfc8648b180ebd79d0e058b3af"

PYRULES_COMMIT = "fe33a4582c37499f3caeb49a07a78fc7948a8949"
PYRULES_SHA256 = "cfa6957832ae0e0c7ee2ccf455a888a291e8419ed8faf45f4420dd7414d5dd96"

XLA_PATCHES = [
    """
    sed -i.bak0 "s/strip_prefix/patch_cmds = [\\\"sed -i.bak0 's\\/HAVE_BACKTRACE=1\\/HAVE_BACKTRACE=0\\/g'\\\"], strip_prefix/g" third_party/llvm/workspace.bzl
    """,
    "find . -type f -name BUILD -exec sed -i 's/\\/\\/third_party\\/py\\/enzyme_ad\\/\\.\\.\\./public/g' {} +", 
    "find . -type f -name BUILD -exec sed -i 's/\\/\\/xla\\/mlir\\/memref:friends/\\/\\/visibility:public/g' {} +",
    "find xla/mlir -type f -name BUILD -exec sed -i 's/\\/\\/xla:internal/\\/\\/\\/\\/visibility:public/g' {} +"
]