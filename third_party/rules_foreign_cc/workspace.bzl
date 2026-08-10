load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

RULES_FOREIGN_CC_COMMIT = "f68b351c4691e747f889dc5e4c2cac3cd3b66ea2"
RULES_FOREIGN_CC_SHA256 = ""

def repo():
    http_archive(
        name = "rules_foreign_cc",
        sha256 = RULES_FOREIGN_CC_SHA256,
        strip_prefix = "rules_foreign_cc-" + RULES_FOREIGN_CC_COMMIT,
        urls = ["https://github.com/bazel-contrib/rules_foreign_cc/archive/{commit}.tar.gz".format(commit = RULES_FOREIGN_CC_COMMIT)],
    )
