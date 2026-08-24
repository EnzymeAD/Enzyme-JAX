def _py_package_impl(ctx):
    inputs = depset(
        transitive = [dep[DefaultInfo].data_runfiles.files for dep in ctx.attr.deps] +
                     [dep[DefaultInfo].default_runfiles.files for dep in ctx.attr.deps],
    )

    filtered_files = []

    packages = [package.label for package in ctx.attr.packages]

    # TODO: rewrite path
    for input_file in inputs.to_list():
        if input_file.owner in packages:
            filtered_files.append(input_file)
    filtered_inputs = depset(direct = filtered_files)

    return [DefaultInfo(
        files = filtered_inputs,
    )]

py_package_lib = struct(
    implementation = _py_package_impl,
    attrs = {
        "deps": attr.label_list(
            doc = "",
        ),
        "packages": attr.label_list(
            mandatory = False,
            allow_empty = True,
            doc = """\
List of targets whose files are included in the distribution.
""",
        ),
    },
)

py_package = rule(
    implementation = py_package_lib.implementation,
    doc = """\
A rule to select all files in transitive dependencies of deps which
belong to given set of Python packages.

This rule is intended to be used as data dependency to py_wheel rule.
""",
    attrs = py_package_lib.attrs,
)
