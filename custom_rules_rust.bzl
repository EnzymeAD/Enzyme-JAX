load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _fetch_rules_rust(repository_ctx):
    # Detect the OS
    os_name = repository_ctx.os.name
    patch_file = None

    if os_name == "linux":
        patch_file = repository_ctx.path(repository_ctx.attr.linux_patch)
    elif os_name == "mac os x":
        patch_file = repository_ctx.path(repository_ctx.attr.macos_patch)
    else:
        # TODO
        print("not linux or macos, defaulting to macos patch")
        patch_file = repository_ctx.path(repository_ctx.attr.macos_patch)

    # Download and extract the archive
    repository_ctx.download_and_extract(
        url = repository_ctx.attr.url,
    )

    # Apply the appropriate patch
    if patch_file:
        repository_ctx.patch(
            patch_file,
            strip = 1,
        )

# Define the custom repository rule
custom_rules_rust = repository_rule(
    implementation = _fetch_rules_rust,
    attrs = {
        "url": attr.string(mandatory = True),
        "linux_patch": attr.label(mandatory = True),
        "macos_patch": attr.label(mandatory = True),
    },
)

