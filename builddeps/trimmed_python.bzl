"""A filegroup minus the files whose paths contain any of the given strings."""

def _filtered_files_impl(ctx):
    files = [
        f
        for f in ctx.files.srcs
        if not any([s in f.short_path for s in ctx.attr.exclude_substrings])
    ]
    return [DefaultInfo(files = depset(files))]

filtered_files = rule(
    implementation = _filtered_files_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = True),
        "exclude_substrings": attr.string_list(),
    },
)
