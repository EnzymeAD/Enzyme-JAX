"""Repository rule for detecting ROCm and enabling MLIR ROCm conversions.

Reads ROCM_PATH env var (set by CI after running ReactantBuilder's
build_tarballs.jl), falls back to /opt/rocm. When ROCm is found, generates
a defs.bzl that:
  - exports if_enzyme_rocm_available(if_true, if_false=[])
  - exports ENZYME_ROCM_XLA_PATCHES containing a sed command to set
    MLIR_ENABLE_ROCM_CONVERSIONS=1 in the LLVM build
"""

def _enzyme_rocm_configure_impl(ctx):
    rocm_path = ctx.os.environ.get("ROCM_PATH", "")
    found = False

    if rocm_path:
        found = ctx.path(rocm_path).exists
    else:
        default_path = "/opt/rocm"
        if ctx.path(default_path).exists:
            rocm_path = default_path
            found = True

    ctx.file("BUILD", "")

    if found:
        # Helper characters for building the multi-escaped sed command.
        BS = "\\"  # single backslash
        DQ = '"'  # double quote

        # Build the sed patch command for defs.bzl.
        patch_line = (
            "sed -i.bak0 " + DQ +
            "s/patch_cmds = " + BS + BS + "[" +
            "/patch_cmds = " + BS + BS + "[" + BS + BS + BS + DQ +
            "find . -type f -name BUILD.bazel -exec sed -i.bak0 " +
            "'" + "s" + BS + BS + "/MLIR_ENABLE_ROCM_CONVERSIONS 0" +
            BS + BS + "/MLIR_ENABLE_ROCM_CONVERSIONS 1" +
            BS + BS + "/g'" +
            " {} +" + BS + BS + BS + DQ + ",/g" + DQ +
            " third_party/llvm/workspace.bzl"
        )

        content = (
            "ENZYME_ROCM_FOUND = True\n\n" +
            "def if_enzyme_rocm_available(if_true, if_false = []):\n" +
            "    return if_true\n\n" +
            'ENZYME_ROCM_XLA_PATCHES = ["""\n' +
            patch_line + "\n" +
            '"""]\n'
        )
    else:
        content = (
            "ENZYME_ROCM_FOUND = False\n\n" +
            "def if_enzyme_rocm_available(if_true, if_false = []):\n" +
            "    return if_false\n\n" +
            "ENZYME_ROCM_XLA_PATCHES = []\n"
        )

    ctx.file("defs.bzl", content)

enzyme_rocm_configure = repository_rule(
    implementation = _enzyme_rocm_configure_impl,
    local = True,
    environ = ["ROCM_PATH"],
)
