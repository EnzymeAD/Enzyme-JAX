"""Loads Hedron's Compile Commands Extractor for Bazel."""
# https://github.com/hedronvision/bazel-compile-commands-extractor

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//:workspace.bzl", "HEDRON_COMPILE_COMMANDS_COMMIT", "HEDRON_COMPILE_COMMANDS_SHA256")

def repo():
    http_archive(
        name = "hedron_compile_commands",
        sha256 = HEDRON_COMPILE_COMMANDS_SHA256,
        # Replace the commit hash in both places (below) with the latest (https://github.com/hedronvision/bazel-compile-commands-extractor/commits/main), rather than using the stale one here.
        # Even better, set up Renovate and let it do the work for you (see "Suggestion: Updates" in the README).
        url = "https://github.com/hedronvision/bazel-compile-commands-extractor/archive/{commit}.tar.gz".format(commit = HEDRON_COMPILE_COMMANDS_COMMIT),
        strip_prefix = "bazel-compile-commands-extractor-" + HEDRON_COMPILE_COMMANDS_COMMIT,
        # When you first run this tool, it'll recommend a sha256 hash to put here with a message like: "DEBUG: Rule 'hedron_compile_commands' indicated that a canonical reproducible form can be obtained by modifying arguments sha256 = ..."
    )
