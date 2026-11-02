"""Loads XPROF"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

XPROF_COMMIT = "e52e5ba901b996e411eb44e760e7ff9f95d9994b"
XPROF_SHA256 = ""

def repo(repo_name = ""):
    http_archive(
        name = "org_xprof",
        # The root BUILD loads @npm and @repository_configuration, which only
        # xprof's own WORKSPACE defines; keep just what the C++ targets use.
        patch_cmds = [
            """
cat > BUILD <<'EOF'
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("@rules_python//python:defs.bzl", "py_library")

licenses(["notice"])

exports_files(["LICENSE"])

bool_flag(
    name = "enable_embedded_features",
    build_setting_default = False,
    visibility = ["//visibility:public"],
)

config_setting(
    name = "embedded_features_enabled",
    flag_values = {":enable_embedded_features": "True"},
    visibility = ["//visibility:public"],
)

py_library(
    name = "expect_tensorflow_installed",
    visibility = ["//visibility:public"],
)
EOF
""",
        ],
        sha256 = XPROF_SHA256,
        strip_prefix = "xprof-" + XPROF_COMMIT,
        urls = ["https://github.com/openxla/xprof/archive/{commit}.tar.gz".format(commit = XPROF_COMMIT)],
    )

    http_archive(
        name = "nlohmann_json",
        build_file_content = """
cc_library(
    name = "json",
    hdrs = glob(["include/nlohmann/**/*.hpp"]),
    includes = ["include"],
    visibility = ["//visibility:public"],
)
""",
        sha256 = "0d8ef5af7f9794e3263480193c491549b2ba6cc74bb018906202ada498a79406",
        strip_prefix = "json-3.11.3",
        urls = ["https://github.com/nlohmann/json/archive/v3.11.3.tar.gz"],
    )

    http_archive(
        name = "opentelemetry-cpp",
        build_file_content = """
cc_library(
    name = "api",
    hdrs = glob(["api/include/**/*.h"]),
    includes = ["api/include"],
    visibility = ["//visibility:public"],
)
""",
        sha256 = "b149109d5983cf8290d614654a878899a68b0c8902b64c934d06f47cd50ffe2e",
        strip_prefix = "opentelemetry-cpp-1.18.0",
        urls = ["https://github.com/open-telemetry/opentelemetry-cpp/archive/v1.18.0.tar.gz"],
    )

    http_archive(
        name = "com_github_googlecloudplatform_google_cloud_cpp",
        patch_cmds = [
            """find . -type f -name '*.cc' -exec sed -i.bak -e 's/Windows.h/windows.h/g' -e 's/Ntstatus.h/ntstatus.h/g' {} +""",
        ],
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
