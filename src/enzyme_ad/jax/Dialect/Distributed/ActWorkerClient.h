#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_ACTWORKERCLIENT_H
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_ACTWORKERCLIENT_H

#include "Dialect.h"

#include <string>

namespace mlir::enzyme::distributed {

// Deployment-specific knobs for dispatching a kernel body to ACT (or reusing
// a cached result from a previous dispatch). One config is shared across
// every kernel a single compilation costs: the fields here are exactly the
// launch-time parameters a pass threads through as pass options, and none
// of them varies per kernel or per request.
struct ActWorkerConfig {
  // Root directory under which a work directory keyed by (target, calling
  // user) is created: requests/, responses/, cache/, and the container
  // lifecycle lockfile all live under
  // "<workdirRoot>/<target>-uid<getuid()>/". Keying on target keeps a worker
  // for one accelerator from ever being silently reused for another, so a
  // cost comparison across targets can't be corrupted by a stale result from
  // the wrong one; keying on uid keeps separate users on this shared machine
  // from colliding on the same container name or cache.
  std::string workdirRoot;

  // Accelerator target name (e.g. "eva2", "eva2-fine", "eva2-coarse"),
  // passed through to act_worker.py's --target. A worker process serves
  // exactly one target for its whole lifetime, so this is a launch-time
  // parameter, not part of the per-request wire protocol, and is assumed to
  // already be safe to use verbatim in a docker container name (true of
  // every real target name today).
  std::string target;

  // Directory containing act_worker.py, bind-mounted read-only into the
  // container at /worker. The worker script's filename itself is part of
  // the fixed wire contract the client and act_worker.py agree on, so only
  // its directory is configurable here.
  std::string workerScriptDir;

  // Path to the act-eva-egan checkout, bind-mounted read-only at /workspace
  // and added to PYTHONPATH inside the container (act_worker.py's own doc
  // comment explains why the image's baked-in PYTHONPATH doesn't already
  // cover this).
  std::string actRepoPath = "/mnt/eganj/act-eva-egan";

  // Docker image containing the built ACT toolchain.
  std::string dockerImage = "act-eva-egan:latest-amd64";

  // Seconds of no new requests after which a launched worker exits on its
  // own (act_worker.py's --idle-timeout). This only bounds how long an idle
  // container lingers before self-terminating; a lazy-start on the next
  // request restarts one regardless, so it has no bearing on correctness.
  double idleTimeoutSeconds = 300.0;

  // How long to wait for a single request's response before treating the
  // worker as unresponsive and, after re-verifying (and if needed
  // restarting) the container, retrying once. See ensureActPatchedKernel's
  // doc comment for the full reasoning.
  double requestTimeoutSeconds = 60.0;
};

// Ensures kernelOp's body has been costed and rewritten by ACT for
// config.target, splicing the result in via spliceKernelBodyModule.
//
// Three-tier lookup, cheapest first:
//   1. Content-hash cache: buildKernelBodyModule's printed text, hashed with
//      a real cryptographic hash (a collision here would silently misattribute
//      one kernel's ACT-compiled body/cost to a different kernel, which is a
//      correctness bug, not just a wasted lookup). A hit reads
//      "<workdir>/cache/<hash>.mlir" straight off disk, with no container or
//      worker involved at all.
//   2. On a miss, a persistent worker container is lazily started (or reused
//      if already running) and the kernel body is dispatched to it over the
//      request/response file protocol act_worker.py implements. A successful
//      response is written to the cache for every future lookup, this run or
//      a later one, before being spliced in.
//   3. If a response doesn't appear within config.requestTimeoutSeconds (the
//      container may have crashed, or hit its own idle-timeout, since it was
//      last confirmed running), the container is re-verified/restarted and
//      the request is retried exactly once. A second timeout, or an ACT
//      compile error (a real "<uid>.error" response: the kernel body itself
//      was rejected, which restarting the container cannot fix), is reported
//      as a real diagnostic on kernelOp, not silently swallowed.
//
// Returns failure with a diagnostic already emitted on kernelOp for any of:
// a local filesystem problem, a docker/container-lifecycle problem, an ACT
// compile error, or a patched module that fails to parse or splice.
::mlir::LogicalResult ensureActPatchedKernel(DistributedKernelOp kernelOp,
                                             const ActWorkerConfig &config);

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_ACTWORKERCLIENT_H
