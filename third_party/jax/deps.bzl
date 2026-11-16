"""Reads the dependency versions JAX pins in its MODULE.bazel.

JAX no longer ships a WORKSPACE (nor the repository definitions that came
with it), so the XLA it builds against and the flatbuffers it generates with
now exist only as an `archive_override` and a `bazel_dep` in that file.
Parse them out of it, so that bumping `JAX_COMMIT` keeps pulling in the
versions JAX itself uses.
"""

def _attributes(text):
    """The `name = "value"` pairs of one call, by name.

    A call spells them one per line or all on one line, so take both apart.
    """
    attrs = {}
    for field in text.replace("\n", ",").split(","):
        if "=" not in field:
            continue
        name, value = field.split("=", 1)
        attrs[name.strip()] = value.strip().strip("\"")
    return attrs

def _calls(content, callee):
    return [_attributes(block.split(")")[0]) for block in content.split(callee + "(")[1:]]

def _xla_revision(content):
    for attrs in _calls(content, "archive_override"):
        if attrs.get("module_name") != "xla":
            continue
        prefix = attrs.get("strip_prefix", "")
        if not prefix.startswith("xla-"):
            fail("the XLA archive_override has no xla- strip_prefix")
        return prefix[len("xla-"):], attrs.get("integrity", "")
    fail("could not find the XLA archive_override")

def _module_version(content, module):
    for attrs in _calls(content, "bazel_dep"):
        if attrs.get("name") == module:
            return attrs.get("version", "")
    fail("could not find a bazel_dep on " + module)

def _jax_deps_impl(repository_ctx):
    content = repository_ctx.read(Label("@jax//:MODULE.bazel"))
    commit, integrity = _xla_revision(content)
    repository_ctx.file("BUILD", "")
    repository_ctx.file("deps.bzl", "\n".join([
        "\"\"\"Generated from JAX's MODULE.bazel; do not edit.\"\"\"",
        "",
        "XLA_COMMIT = \"{}\"".format(commit),
        "XLA_INTEGRITY = \"{}\"".format(integrity),
        "FLATBUFFERS_VERSION = \"{}\"".format(_module_version(content, "flatbuffers")),
        "",
    ]))

jax_deps_repository = repository_rule(
    implementation = _jax_deps_impl,
    doc = "Exposes the dependency versions pinned by the JAX repository.",
)
