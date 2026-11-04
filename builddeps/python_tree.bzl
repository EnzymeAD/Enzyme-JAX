"""A Python runtime laid out as one directory artifact plus its interpreter.

The hermetic toolchain declares every file of the interpreter separately, so
each py_test's runfiles tree gets one symlink per file (thousands). Copying
lib/ once into a tree artifact turns that into a single directory symlink;
the interpreter binary is copied next to it so that it finds lib/ (its stdlib
and libpython) at ../lib as usual.
"""

def _python_tree_impl(ctx):
    interp = ctx.file.interpreter
    root = interp.path.rsplit("/bin/", 1)[0]
    lib = ctx.actions.declare_directory(ctx.attr.name + "/lib")
    bin = ctx.outputs.interpreter_out

    lines = []
    for f in ctx.files.srcs:
        if not f.path.startswith(root + "/lib/"):
            continue
        if any([s in f.path for s in ctx.attr.exclude_substrings]):
            continue
        lines.append(f.path + "\t" + f.path[len(root) + len("/lib/"):])
    manifest = ctx.actions.declare_file(ctx.attr.name + ".manifest")
    ctx.actions.write(manifest, "\n".join(lines) + "\n")

    ctx.actions.run_shell(
        inputs = ctx.files.srcs + [interp, manifest],
        outputs = [lib, bin],
        command = """
set -e
while IFS=$'\\t' read -r src rel; do
  [ -n "$rel" ] || continue
  mkdir -p "$LIB/$(dirname "$rel")"
  cp -L "$src" "$LIB/$rel"
done < "$MANIFEST"
cp -L "$INTERP" "$BIN"
chmod +x "$BIN"
""",
        env = {
            "BIN": bin.path,
            "INTERP": interp.path,
            "LIB": lib.path,
            "MANIFEST": manifest.path,
        },
        mnemonic = "PythonTree",
    )
    return [DefaultInfo(files = depset([lib, bin]))]

python_tree = rule(
    implementation = _python_tree_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = True),
        "interpreter": attr.label(allow_single_file = True, mandatory = True),
        "interpreter_out": attr.output(mandatory = True),
        "exclude_substrings": attr.string_list(),
    },
)
