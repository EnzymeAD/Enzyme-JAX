"""Loads the gRPC that google-cloud-cpp asks for, as XLA describes it.

XLA binds its gRPC to the repository name `grpc`, and maps
`@com_github_grpc_grpc` onto it -- but a `repo_mapping` only rewrites the
labels written inside that repository. google-cloud-cpp, which gRPC's own
grpc_deps() pulls in, looks for a repository actually named
`com_github_grpc_grpc` and defines one itself when it finds none, at a gRPC
old enough that its sources no longer compile against the protobuf and upb
this workspace builds with.

Define that name first, from the archive XLA declares -- version, hash and
patches -- so that both names describe the same gRPC and neither has to be
tracked here.
"""

def _call(content, callee, name):
    """The text of the `callee(...)` call that names `name`."""
    for block in content.split(callee + "(")[1:]:
        block = block.split("\n)")[0]
        if "name = \"{}\"".format(name) in block:
            return block
    fail("could not find {}(name = \"{}\")".format(callee, name))

def _strings(text):
    """Every double-quoted string in `text`, in order."""
    return [part for i, part in enumerate(text.split("\"")) if i % 2 == 1]

def _attr(block, attr, optional = False):
    """The strings of `block`'s `attr = ...`, whatever shape it is written in."""
    for i, field in enumerate(block.split(attr + " =")):
        if i == 0:
            continue
        return _strings(field.split("\n    ")[0])
    if optional:
        return []
    fail("could not find the {} of the archive".format(attr))

def _xla_grpc_impl(repository_ctx):
    workspace = Label("@xla//:workspace2.bzl")
    archive = _call(repository_ctx.read(workspace), "tf_http_archive", "grpc")
    repository_ctx.download_and_extract(
        url = _attr(archive, "urls"),
        sha256 = _attr(archive, "sha256")[0],
        stripPrefix = _attr(archive, "strip_prefix")[0],
    )

    # tf_http_archive applies these the same way.
    for patch in _attr(archive, "patch_file", optional = True):
        repository_ctx.patch(Label("@xla" + patch), strip = 1)

xla_grpc_repository = repository_rule(
    implementation = _xla_grpc_impl,
    doc = "Fetches the gRPC archive XLA declares, under a name of your choosing.",
)
