#!/usr/bin/env python3
"""Persistent ACT worker: services StableHLO kernel-compile requests over a file protocol.

`act_e2e.compile.compile_text` does real, useful work in
~0.14s, but a fresh `docker run` plus a fresh JAX import to reach it costs ~1.1s more on
top of that, almost entirely fixed overhead rather than compute. This worker amortizes
that overhead across many kernel-compile calls by staying up inside one already-running
container with `act_e2e` (and JAX) imported exactly once, and taking requests over a
polled file-based protocol instead of a process-per-call. It is one half of that protocol;
the other half is the C++ client in ActWorkerClient.cpp, built against the exact wire
shape documented here, so this file *is* the contract between them. Don't change file
names, suffixes, or the success/error split without updating that client too.

Directory layout under --workdir (each created on startup if missing):
  requests/   <uid>.mlir   one kernel-body StableHLO module per request. Written by the
                           client via write-then-rename, so a `*.mlir.tmp` here is always
                           mid-write on the client's side and is never polled or touched.
  claimed/    <uid>.mlir   a request the poller has picked up and handed to the worker
                           pool, moved here (atomically, via os.rename) so the same file
                           is never dispatched twice and the poller's own listing of
                           requests/ never race with an in-flight compile.
  responses/  <uid>.mlir   the patched module's text, on success.
              <uid>.error  the failing exception's text (traceback included), on failure.
              Both are written by this worker the same way requests are: to a `.tmp`
              sibling first, then renamed into place, so the client never observes a
              partially written response either.
  kernels/<uid>/  passed to `compile_text` as `build_dir`; ACT's own per-kernel build
                  artifacts land here, one subdirectory per request so concurrently
                  handled requests (which may legitimately name the same kernel) never
                  overwrite each other's build outputs.

A request's uid is its filename with the `.mlir` suffix stripped; the response (or error)
for uid `X` is always named `X.mlir` or `X.error`. The claimed request file is deleted
once handled, successfully or not, so it is never reprocessed.

Requests are dispatched to a bounded thread pool (`--workers` threads) rather than
handled one at a time: `compile_text` is most of the wall-clock time this worker spends,
and it is CPU-bound Python/MLIR-parsing work with no known in-process thread-safety
hazard (the one piece of shared state, `target.load_cost_model`'s `functools.cache`, is
benign to race — worst case two threads both load and cache the same module). Handing
several requests to the pool at once is what actually gets concurrent kernel compiles
off the critical path of the search that's waiting on them.

The worker exits cleanly (not a crash) after `--idle-timeout` seconds pass with no
request seen AND none in flight; a request still being handled by the pool when the
timeout would otherwise fire keeps the worker alive until that request's response is
written. Nothing else in this design stops the container this runs in from outliving its
usefulness once a caller is done with it, so self-termination is the leak backstop; the
client's lazy-start just restarts a fresh worker on the next request.

`act_e2e` is not installed as a package; it resolves via act-eva-egan's own repo root
being on `PYTHONPATH` (see that repo's `docker/Dockerfile.amd64`, which mounts the repo at
/workspace but does not itself put /workspace on PYTHONPATH, only its `act`/`taidl`/
`act-backend`/`act-oracle` subdirectories). Whoever launches this worker must add the
repo root to PYTHONPATH themselves; this script does not assume a particular cwd.

Usage:
  python act_worker.py --workdir /path/to/workdir --target eva2 --idle-timeout 300 \
      --workers 6
"""

import argparse
import glob
import os
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from act_e2e import compile as act_compile

# Cheap relative to the ~0.14s/kernel real-compute floor this worker exists to approach,
# and idle CPU cost of a listdir this often is negligible.
POLL_INTERVAL_SECONDS = 0.05


def _atomic_write(final_path, text):
    """Write `text` to `final_path` without a reader ever observing a partial file.

    `os.rename` is atomic within one directory/filesystem on POSIX, so a poller only
    ever sees the fully-written file appear, never a truncated or in-progress one.
    """
    tmp_path = final_path + ".tmp"
    with open(tmp_path, "w") as f:
        f.write(text)
    os.rename(tmp_path, final_path)


def _handle_request(request_path, responses_dir, target, kernels_dir):
    uid = os.path.basename(request_path)[: -len(".mlir")]
    with open(request_path) as f:
        text = f.read()
    build_dir = os.path.join(kernels_dir, uid)
    os.makedirs(build_dir, exist_ok=True)

    # compile_text's failure modes are whatever act_e2e/act-backend/the accelerator
    # toolchain choose to raise. This is a genuine external-tool boundary, so it is
    # reported to the client as-is rather than swallowed or narrowed to some assumed
    # subset of exception types.
    try:
        # A fresh per-request log directory keeps two concurrently handled requests
        # from deleting each other's act-backend PII-graph dumps: the backend deletes
        # and recreates whatever `--log` names on every invocation.
        with tempfile.TemporaryDirectory(prefix="act-log-" + uid + "-") as log_dir:
            patched, _hits, _coverage, _detail = act_compile.compile_text(
                text, target, build_dir=build_dir, log_dir=log_dir
            )
    except Exception:
        _atomic_write(
            os.path.join(responses_dir, uid + ".error"), traceback.format_exc()
        )
    else:
        _atomic_write(os.path.join(responses_dir, uid + ".mlir"), patched)
    finally:
        os.remove(request_path)


def run(workdir, target, idle_timeout, workers):
    requests_dir = os.path.join(workdir, "requests")
    claimed_dir = os.path.join(workdir, "claimed")
    responses_dir = os.path.join(workdir, "responses")
    kernels_dir = os.path.join(workdir, "kernels")
    for d in (requests_dir, claimed_dir, responses_dir, kernels_dir):
        os.makedirs(d, exist_ok=True)

    # Guards in_flight: the poller thread increments it before submitting to the pool
    # and each pool thread decrements it on the way out, so the idle check below can
    # tell "no new requests" apart from "no new requests, but one is still compiling".
    in_flight_lock = threading.Lock()
    in_flight = 0

    def run_claimed(request_path):
        nonlocal in_flight
        try:
            _handle_request(request_path, responses_dir, target, kernels_dir)
        finally:
            with in_flight_lock:
                in_flight -= 1

    last_activity = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while True:
            # glob (not listdir) so a name that merely ends in "mlir" without the dot,
            # or a "*.mlir.tmp" mid-write on the client's side, can never match.
            requests = sorted(glob.glob(os.path.join(requests_dir, "*.mlir")))
            claimed_any = False
            for request_path in requests:
                claimed_path = os.path.join(claimed_dir, os.path.basename(request_path))
                try:
                    # Atomic claim: rename (not copy-then-delete) so exactly one poller
                    # iteration ever succeeds in claiming a given request, even though
                    # nothing else here currently polls requests/ concurrently with
                    # this loop.
                    os.rename(request_path, claimed_path)
                except FileNotFoundError:
                    continue
                claimed_any = True
                # Counted in flight from the moment the pool accepts it, not from
                # when a worker thread actually starts running it, so a burst that
                # outruns `--workers` (queued inside the pool, not yet executing)
                # still counts as active work and can't trip the idle timeout.
                with in_flight_lock:
                    in_flight += 1
                pool.submit(run_claimed, claimed_path)
            with in_flight_lock:
                idle = in_flight == 0
            if claimed_any or not idle:
                last_activity = time.monotonic()
            elif time.monotonic() - last_activity >= idle_timeout:
                return

            time.sleep(POLL_INTERVAL_SECONDS)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--workdir",
        required=True,
        help="directory holding requests/, claimed/, responses/, kernels/",
    )
    ap.add_argument(
        "--target",
        required=True,
        help="accelerator target name, passed to compile_text",
    )
    ap.add_argument(
        "--idle-timeout",
        required=True,
        type=float,
        help="exit cleanly after this many seconds with no request seen and none in flight",
    )
    ap.add_argument(
        "--workers",
        required=True,
        type=int,
        help="thread pool size for concurrently dispatched compile_text calls",
    )
    args = ap.parse_args()
    run(args.workdir, args.target, args.idle_timeout, args.workers)


if __name__ == "__main__":
    main()
