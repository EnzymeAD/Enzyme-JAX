"""Compiles Lean's own Lean sources (src/) with the stage0 `lean`.

Lean builds these with Lake, which discovers the module graph by running
`lean --deps`. Bazel needs the graph before running anything, so
`lean_stage1_repo` reads it from the `import` lines at repository time and
writes one `lean_module` target per module.
"""

LeanModuleInfo = provider(
    doc = "A compiled Lean module.",
    fields = {
        "c_files": "depset of the C emitted for this module and its imports",
        "files": "depset of the .olean/.ir files of this module and its imports",
    },
)

# What `lean -o` writes for a file with a `module` header (see `writeModule`
# in src/Lean/Environment.lean); other files only get the .olean.
_MODULE_EXTENSIONS = [
    "olean",
    "olean.server",
    "olean.private",
    "ir",
    "ir.sig",
]

def _lean_module_impl(ctx):
    path = ctx.attr.module.replace(".", "/")
    extensions = _MODULE_EXTENSIONS if ctx.attr.is_module else ["olean"]
    outs = [
        ctx.actions.declare_file("lib/lean/{}.{}".format(path, ext))
        for ext in extensions
    ]
    c_file = ctx.actions.declare_file("lib/temp/{}.c".format(path))
    imports = [dep[LeanModuleInfo] for dep in ctx.attr.deps]
    imported_files = depset(transitive = [i.files for i in imports])

    olean = outs[0]
    args = ctx.actions.args()
    args.add(ctx.file.src)
    args.add(ctx.executable.lean)
    args.add("-o", olean)
    args.add("-c", c_file)
    # As CMake does for every stage: the stage0 `lean` must not run its own
    # native code in place of the stage1 definitions it is compiling.
    args.add("-Dinterpreter.prefer_native=false")
    args.add(ctx.file.src)

    # Lean derives the module name from the source path relative to -R, after
    # resolving symlinks in both. In the sandbox the source is a symlink into
    # the external repository, so -R is taken from its resolved path.
    ctx.actions.run_shell(
        command = 'root=$(realpath "$1") && lean=$2 && shift 2 && ' +
                  'exec "$lean" -R "${{root%{}.lean}}" "$@"'.format(path),
        arguments = [args],
        inputs = depset([ctx.file.src], transitive = [imported_files]),
        tools = [ctx.executable.lean],
        outputs = outs + [c_file],
        env = {
            "LEAN_ABORT_ON_PANIC": "1",
            "LEAN_PATH": olean.path[:-len(path + ".olean")],
        },
        mnemonic = "LeanCompile",
        progress_message = "Compiling Lean module {}".format(ctx.attr.module),
    )
    return [
        DefaultInfo(files = depset(outs + [c_file])),
        LeanModuleInfo(
            c_files = depset(
                [c_file],
                transitive = [i.c_files for i in imports],
            ),
            files = depset(outs, transitive = [imported_files]),
        ),
    ]

lean_module = rule(
    implementation = _lean_module_impl,
    attrs = {
        "deps": attr.label_list(providers = [LeanModuleInfo]),
        "is_module": attr.bool(mandatory = True),
        "lean": attr.label(
            default = "@lean4//:stage0_lean",
            executable = True,
            cfg = "exec",
        ),
        "module": attr.string(mandatory = True),
        "src": attr.label(allow_single_file = [".lean"], mandatory = True),
    },
)

def _lean_c_srcs_impl(ctx):
    return [DefaultInfo(files = depset(
        transitive = [dep[LeanModuleInfo].c_files for dep in ctx.attr.deps],
    ))]

lean_c_srcs = rule(
    implementation = _lean_c_srcs_impl,
    doc = "The C emitted for `deps` and everything they import.",
    attrs = {
        "deps": attr.label_list(providers = [LeanModuleInfo]),
    },
)

def _skip_block_comment(text, i):
    """Returns the index just past the (nestable) /- -/ comment at i."""
    depth = 0
    for _ in range(len(text)):
        if text.startswith("/-", i):
            depth += 1
            i += 2
        elif text.startswith("-/", i):
            depth -= 1
            i += 2
            if depth == 0:
                return i
        else:
            i += 1
    fail("unterminated comment")

def _parse_header(text):
    """Returns (is_module, imports) of a Lean file.

    Follows `header` in src/Lean/Parser/Module/Syntax.lean:
    `module`? `prelude`? (`public`? `meta`? `import` `all`? ident)*.
    """
    is_module = False
    imports = []
    after_import = False
    i = 0
    n = len(text)
    for _ in range(n):
        if i >= n:
            break
        if text[i] in " \t\r\n":
            i += 1
            continue
        if text.startswith("--", i):
            end = text.find("\n", i)
            i = n if end < 0 else end + 1
            continue
        if text.startswith("/-", i):
            i = _skip_block_comment(text, i)
            continue
        end = i
        for _ in range(n - i):
            if end >= n or text[end] in " \t\r\n":
                break
            end += 1
        token = text[i:end]
        i = end
        if after_import:
            if token != "all":
                # A module can be imported more than once, e.g. both with and
                # without `meta`.
                if token not in imports:
                    imports.append(token)
                after_import = False
        elif token == "import":
            after_import = True
        elif token == "module" and not imports:
            is_module = True
        elif token not in ("prelude", "public", "meta"):
            break
    return is_module, imports

def _lean_stage1_repo_impl(rctx):
    src = rctx.path(Label("@lean4//:src/Init.lean")).dirname
    files = []
    pending = [src.get_child(d) for d in ("Init", "Std", "Lean")]
    for _ in range(100000):
        if not pending:
            break
        d = pending.pop()
        for child in d.readdir():
            if child.is_dir:
                pending.append(child)
            elif child.basename.endswith(".lean"):
                files.append(child)
    files += [src.get_child(f) for f in ("Init.lean", "Std.lean", "Lean.lean")]

    src_prefix = str(src) + "/"
    targets = []
    for f in sorted(files, key = str):
        relative = str(f)[len(src_prefix):]
        module = relative[:-len(".lean")].replace("/", ".")
        is_module, imports = _parse_header(rctx.read(f))
        targets.append("""lean_module(
    name = "{module}",
    src = "@lean4//:src/{relative}",
    is_module = {is_module},
    module = "{module}",
    deps = {deps},
)
""".format(
            deps = repr([":" + i for i in imports]),
            is_module = is_module,
            module = module,
            relative = relative,
        ))

    rctx.file("BUILD.bazel", """load("{bzl}", "lean_c_srcs", "lean_module")

package(default_visibility = ["//visibility:public"])

# The Init, Std and Lean libraries of src/lakefile.toml.in, which make up
# libleanshared along with the C++ code.
lean_c_srcs(
    name = "c_srcs",
    deps = [
        ":Init",
        ":Std",
        ":Lean",
        ":Lean.Compiler.IR.EmitLLVM",
        ":Lean.Compiler.LCNF.Probing",
    ],
)

{targets}""".format(
        bzl = str(Label("//third_party/lean:stage1.bzl")),
        targets = "\n".join(targets),
    ))

lean_stage1_repo = repository_rule(
    implementation = _lean_stage1_repo_impl,
)
