"""Loads ML_TOOLCHAIN."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//:workspace.bzl", "ML_TOOLCHAIN_COMMIT", "ML_TOOLCHAIN_SHA256")

ML_TOOLCHAIN_PATCHES = [
    """
	sed -i.bak0 "/D_FORTIFY_SOURCE/d" cc/features/BUILD third_party/gpus/crosstool/cc_toolchain_config.bzl.tpl
	sed -i.bak0 "/DNDEBUG/d" cc/features/BUILD third_party/gpus/crosstool/cc_toolchain_config.bzl.tpl
	""",
]

def repo(extra_patches = [], override_commit = ""):
    commit = ML_TOOLCHAIN_COMMIT
    sha = ML_TOOLCHAIN_SHA256
    if len(override_commit):
        commit = override_commit
        sha = ""
    http_archive(
        name = "rules_ml_toolchain",
        sha256 = sha,
        strip_prefix = "rules_ml_toolchain-" + commit,
        urls = ["https://github.com/google-ml-infra/rules_ml_toolchain/archive/{commit}.tar.gz".format(commit = commit)],
        patch_cmds = ML_TOOLCHAIN_PATCHES + extra_patches,
    )
