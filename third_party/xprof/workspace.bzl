"""Loads XPROF"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

XPROF_COMMIT = "a193bc6485ff2865f284362a038761165d833311"
XPROF_SHA256 = ""

def repo(repo_name = ""):
    http_archive(
        name = "com_github_googlecloudplatform_google_cloud_cpp",
        patch_args = ["-p1"],
        patches = ["//third_party:google_cloud_cpp.patch"],
        repo_mapping = {
            "@com_github_curl_curl": "@curl",
            "@com_github_nlohmann_json": "@nlohmann_json",
            "@nlohmann_json": "@nlohmann_json",
            "@abseil-cpp": "@com_google_absl",
        },
        sha256 = "e868bdb537121d2169fbc1ef69b81f4b4f96e97891c4567a6533d4adf62bffde",
        strip_prefix = "google-cloud-cpp-3.1.0",
        urls = ["https://github.com/googleapis/google-cloud-cpp/archive/v3.1.0.tar.gz"],
    )

    http_archive(
        name = "org_xprof",
        sha256 = XPROF_SHA256,
        strip_prefix = "xprof-" + XPROF_COMMIT,
        urls = ["https://github.com/openxla/xprof/archive/{commit}.tar.gz".format(commit = XPROF_COMMIT)],
    )
