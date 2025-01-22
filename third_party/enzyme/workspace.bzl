"""Loads Enzyme."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//:workspace.bzl", "ENZYME_COMMIT", "ENZYME_SHA256", "OVERRIDE_ENZYME_PATH")

def repo():
    if len(OVERRIDE_ENZYME_PATH) != 0:
        native.local_repository(
            name = "enzyme",
            path = OVERRIDE_ENZYME_PATH
        )
    else:
        http_archive(
           name = "enzyme",
           sha256 = ENZYME_SHA256,
           strip_prefix = "Enzyme-" + ENZYME_COMMIT + "/enzyme",
           urls = ["https://github.com/EnzymeAD/Enzyme/archive/{commit}.tar.gz".format(commit = ENZYME_COMMIT)],
        )
