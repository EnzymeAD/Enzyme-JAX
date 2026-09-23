#include "ActWorkerClient.h"
#include "Utilities.h"

#include "mlir/Parser/Parser.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/file.h>
#include <thread>
#include <unistd.h>

namespace mlir::enzyme::distributed {

namespace {

std::string formatSeconds(double seconds) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%g", seconds);
  return buf;
}

struct CommandResult {
  int exitCode = -1;
  std::string stdOut;
  std::string stdErr;
};

// Runs `docker <args>`, capturing stdout/stderr through temp files:
// ExecuteAndWait redirects to paths, not pipes, so this is the portable way
// to capture a subprocess's output with the facilities Support already
// gives us (no new dependency beyond the LLVM Support this target already
// links).
llvm::Expected<CommandResult> runDocker(llvm::ArrayRef<llvm::StringRef> args,
                                        unsigned timeoutSeconds) {
  auto dockerPath = llvm::sys::findProgramByName("docker");
  if (!dockerPath) {
    return llvm::errorCodeToError(dockerPath.getError());
  }

  llvm::SmallString<128> outPath, errPath;
  if (auto ec = llvm::sys::fs::createTemporaryFile("act-worker-client", "out",
                                                   outPath)) {
    return llvm::errorCodeToError(ec);
  }
  if (auto ec = llvm::sys::fs::createTemporaryFile("act-worker-client", "err",
                                                   errPath)) {
    return llvm::errorCodeToError(ec);
  }
  auto cleanupTemps = llvm::scope_exit([&] {
    llvm::sys::fs::remove(outPath);
    llvm::sys::fs::remove(errPath);
  });

  llvm::SmallVector<llvm::StringRef> fullArgs;
  fullArgs.push_back(*dockerPath);
  fullArgs.append(args.begin(), args.end());

  std::optional<llvm::StringRef> redirects[] = {
      llvm::StringRef(), // stdin -> /dev/null
      llvm::StringRef(outPath), llvm::StringRef(errPath)};
  std::string errMsg;
  bool executionFailed = false;
  int rc = llvm::sys::ExecuteAndWait(
      *dockerPath, fullArgs, /*Env=*/std::nullopt, redirects, timeoutSeconds,
      /*MemoryLimit=*/0, &errMsg, &executionFailed);
  if (executionFailed) {
    return llvm::createStringError(llvm::Twine("failed to execute 'docker': ") +
                                   errMsg);
  }

  CommandResult result;
  result.exitCode = rc;
  if (auto buf = llvm::MemoryBuffer::getFile(outPath)) {
    result.stdOut = (*buf)->getBuffer().str();
  }
  if (auto buf = llvm::MemoryBuffer::getFile(errPath)) {
    result.stdErr = (*buf)->getBuffer().str();
  }
  return result;
}

// Rootless Docker (the expected setup on a shared-machine-per-user host,
// which the rest of this repo also assumes) maps container uid 0 to the
// host user who started the daemon, so leaving a container at its default
// root is what makes files land on the bind mount owned by the real host
// user rather than root; an explicit `--user $(id -u):$(id -g)` instead
// resolves through /etc/subuid to a different, unrelated, permission-less
// host uid, so writes come back "Permission denied". A non-rootless
// (system) daemon has the opposite behavior: container-root really is
// host-root there, so --user is the right fix in that case. Hence the
// runtime check below rather than picking one unconditionally.
bool isRootlessDocker() {
  static bool cached = [] {
    auto result = runDocker({"info", "--format", "{{json .SecurityOptions}}"},
                            /*timeoutSeconds=*/10);
    if (!result) {
      llvm::consumeError(result.takeError());
      return false;
    }
    return result->exitCode == 0 &&
           llvm::StringRef(result->stdOut).contains("name=rootless");
  }();
  return cached;
}

// Idempotent lazy-start-or-reuse of the worker container for this
// (target, uid) key, guarded end-to-end by an advisory flock on a lockfile
// inside workdir. flock (rather than a stale-PID-checking lockfile scheme)
// is the right tool here specifically because it releases automatically if
// this process dies mid-section: a crashed enzymexlamlir-opt must never
// leave a permanently-stuck lock blocking every future search.
::mlir::LogicalResult ensureContainerRunning(DistributedKernelOp kernelOp,
                                             const ActWorkerConfig &config,
                                             llvm::StringRef containerName,
                                             llvm::StringRef workdir) {
  llvm::SmallString<128> lockPath(workdir);
  llvm::sys::path::append(lockPath, "lifecycle.lock");
  int lockFd = ::open(lockPath.c_str(), O_CREAT | O_RDWR, 0644);
  if (lockFd < 0) {
    return kernelOp.emitError() << "failed to open ACT worker lifecycle lock '"
                                << lockPath << "': " << std::strerror(errno);
  }
  if (::flock(lockFd, LOCK_EX) != 0) {
    int savedErrno = errno;
    ::close(lockFd);
    return kernelOp.emitError()
           << "failed to acquire ACT worker lifecycle lock '" << lockPath
           << "': " << std::strerror(savedErrno);
  }
  auto unlockGuard = llvm::scope_exit([&] {
    ::flock(lockFd, LOCK_UN);
    ::close(lockFd);
  });

  auto statusResult = runDocker(
      {"inspect", "--format", "{{.State.Status}}", containerName}, 10);
  if (!statusResult) {
    return kernelOp.emitError()
           << "failed to invoke docker to check the ACT worker container's "
              "status: "
           << llvm::toString(statusResult.takeError());
  }
  // A nonzero exit here almost always just means "no such container" (the
  // expected first-run case), not a genuine docker problem: a real
  // docker/daemon failure surfaces just as loudly on the `docker run` below,
  // so it isn't worth distinguishing the two cases here.
  std::string status;
  if (statusResult->exitCode == 0) {
    status = llvm::StringRef(statusResult->stdOut).trim().str();
  }

  if (status == "running") {
    return success();
  }

  if (!status.empty()) {
    // Leak mitigation: a worker that crashed or was reaped leaves a
    // stopped/dead container registered under this name; `docker run
    // --name` fails outright on that collision rather than replacing it, so
    // always clear it first, regardless of how the previous instance died.
    auto rmResult = runDocker({"rm", "-f", containerName}, 10);
    if (!rmResult || rmResult->exitCode != 0) {
      return kernelOp.emitError()
             << "failed to remove stale ACT worker container '" << containerName
             << "' (status was '" << status << "')";
    }
  }

  std::string uidGid =
      (llvm::Twine(getuid()) + ":" + llvm::Twine(getgid())).str();
  std::string workspaceMount =
      (llvm::Twine(config.actRepoPath) + ":/workspace:ro").str();
  std::string workerMount =
      (llvm::Twine(config.workerScriptDir) + ":/worker:ro").str();
  std::string workdirMount = (llvm::Twine(workdir) + ":/workdir:rw").str();
  std::string idleTimeoutStr = formatSeconds(config.idleTimeoutSeconds);

  llvm::SmallVector<llvm::StringRef> runArgs = {
      "run", "-d",           "--name", containerName,
      "-v",  workspaceMount, "-v",     workerMount,
      "-v",  workdirMount,   "-e",     "PYTHONPATH=/workspace",
      "-w",  "/workspace",
  };
  if (!isRootlessDocker()) {
    runArgs.push_back("--user");
    runArgs.push_back(uidGid);
  }
  runArgs.push_back(config.dockerImage);
  runArgs.push_back("python");
  runArgs.push_back("/worker/act_worker.py");
  runArgs.push_back("--workdir");
  runArgs.push_back("/workdir");
  runArgs.push_back("--target");
  runArgs.push_back(config.target);
  runArgs.push_back("--idle-timeout");
  runArgs.push_back(idleTimeoutStr);

  auto runResult = runDocker(runArgs, 30);
  if (!runResult) {
    return kernelOp.emitError()
           << "failed to start ACT worker container '" << containerName
           << "': " << llvm::toString(runResult.takeError());
  }
  if (runResult->exitCode != 0) {
    return kernelOp.emitError() << "failed to start ACT worker container '"
                                << containerName << "': " << runResult->stdErr;
  }

  // `docker run -d` returns once the container process has launched, which
  // should already be "running" by the time it returns; poll briefly rather
  // than assuming, since this is exactly the kind of external-tool boundary
  // worth confirming rather than asserting past.
  for (int attempt = 0; attempt < 40; ++attempt) {
    auto check = runDocker(
        {"inspect", "--format", "{{.State.Status}}", containerName}, 10);
    if (check && check->exitCode == 0 &&
        llvm::StringRef(check->stdOut).trim() == "running") {
      return success();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  return kernelOp.emitError() << "ACT worker container '" << containerName
                              << "' did not reach the 'running' state after "
                                 "starting";
}

enum class DispatchOutcome { Success, ActError, Timeout };

struct DispatchAttempt {
  DispatchOutcome outcome;
  // The patched module text on Success, or ACT's own error text on ActError.
  std::string text;
};

// One request/response round trip against whatever worker is running (or
// times out waiting for one). Does not itself decide whether a timeout is
// recoverable: dispatchWithRetry owns that policy.
::mlir::FailureOr<DispatchAttempt> dispatchOnce(DistributedKernelOp kernelOp,
                                                const ActWorkerConfig &config,
                                                llvm::StringRef workdir,
                                                llvm::StringRef bodyText) {
  llvm::SmallString<128> requestsDir(workdir), responsesDir(workdir);
  llvm::sys::path::append(requestsDir, "requests");
  llvm::sys::path::append(responsesDir, "responses");
  if (auto ec = llvm::sys::fs::create_directories(requestsDir)) {
    return kernelOp.emitError() << "failed to create ACT requests directory '"
                                << requestsDir << "': " << ec.message();
  }
  if (auto ec = llvm::sys::fs::create_directories(responsesDir)) {
    return kernelOp.emitError() << "failed to create ACT responses directory '"
                                << responsesDir << "': " << ec.message();
  }

  // Mirrors act_worker.py's own write-then-rename convention.
  // createUniqueFile both mints a collision-free uid and atomically creates
  // the ".tmp" file in one step, so no concurrently-polling worker can ever
  // observe a partially-written request.
  llvm::SmallString<128> tmpModel(requestsDir);
  llvm::sys::path::append(tmpModel, "request-%%%%%%.mlir.tmp");
  int fd;
  llvm::SmallString<128> tmpPath;
  if (auto ec = llvm::sys::fs::createUniqueFile(tmpModel, fd, tmpPath)) {
    return kernelOp.emitError()
           << "failed to create an ACT request file under '" << requestsDir
           << "': " << ec.message();
  }
  {
    llvm::raw_fd_ostream os(fd, /*shouldClose=*/true);
    os << bodyText;
  }
  // Strip the literal ".tmp" this function itself appended above. This
  // isn't parsing untrusted external input, so a fixed-length drop_back is
  // exact.
  std::string finalPath = llvm::StringRef(tmpPath).drop_back(4).str();
  if (auto ec = llvm::sys::fs::rename(tmpPath, finalPath)) {
    return kernelOp.emitError() << "failed to publish ACT request file '"
                                << finalPath << "': " << ec.message();
  }

  std::string uid =
      llvm::sys::path::stem(llvm::sys::path::filename(finalPath)).str();
  llvm::SmallString<128> respOk(responsesDir), respErr(responsesDir);
  llvm::sys::path::append(respOk, uid + ".mlir");
  llvm::sys::path::append(respErr, uid + ".error");

  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::duration<double>(config.requestTimeoutSeconds);
  while (std::chrono::steady_clock::now() < deadline) {
    if (llvm::sys::fs::exists(respOk)) {
      auto buf = llvm::MemoryBuffer::getFile(respOk);
      llvm::sys::fs::remove(respOk); // client cleans up what it consumes.
      if (!buf) {
        return kernelOp.emitError() << "failed to read ACT response '" << respOk
                                    << "': " << buf.getError().message();
      }
      return DispatchAttempt{DispatchOutcome::Success,
                             (*buf)->getBuffer().str()};
    }
    if (llvm::sys::fs::exists(respErr)) {
      auto buf = llvm::MemoryBuffer::getFile(respErr);
      llvm::sys::fs::remove(respErr);
      return DispatchAttempt{DispatchOutcome::ActError,
                             buf ? (*buf)->getBuffer().str()
                                 : "<failed to read ACT error response>"};
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  // Timed out: the request file (and any response that eventually shows up
  // for this uid) is left alone. The worker may still be midway through it,
  // and a stale, unread response arriving later is small, harmless litter,
  // not worth racing to clean up here. dispatchWithRetry resubmits under a
  // fresh uid rather than trying to reclaim this one.
  return DispatchAttempt{DispatchOutcome::Timeout, {}};
}

// One retry, each preceded by re-verifying (and if needed restarting) the
// container: a response can fail to appear because the worker crashed or
// hit its own idle-timeout sometime after it was last confirmed running, so
// a single "confirmed running" check up front can't be trusted to hold for
// the whole wait. Bounding this to a fixed number of attempts turns that
// race into a real, reported failure instead of an indefinite hang or a
// silently-swallowed miss.
::mlir::FailureOr<std::string> dispatchWithRetry(DistributedKernelOp kernelOp,
                                                 const ActWorkerConfig &config,
                                                 llvm::StringRef containerName,
                                                 llvm::StringRef workdir,
                                                 llvm::StringRef bodyText) {
  constexpr int kMaxAttempts = 2;
  for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
    if (failed(
            ensureContainerRunning(kernelOp, config, containerName, workdir))) {
      return failure(); // diagnostic already emitted
    }

    auto attemptResult = dispatchOnce(kernelOp, config, workdir, bodyText);
    if (failed(attemptResult)) {
      // A local filesystem problem, already diagnosed: restarting the
      // container wouldn't fix it, so don't spend the retry on it.
      return failure();
    }

    switch (attemptResult->outcome) {
    case DispatchOutcome::Success:
      return attemptResult->text;
    case DispatchOutcome::ActError:
      // ACT itself ran and rejected this kernel body: the container is
      // fine, so this is terminal, not something a restart-and-retry helps.
      return kernelOp.emitError() << "ACT failed to compile this kernel body:\n"
                                  << attemptResult->text;
    case DispatchOutcome::Timeout:
      break; // fall through to the next attempt.
    }
  }
  return kernelOp.emitError()
         << "ACT worker '" << containerName << "' did not respond within "
         << config.requestTimeoutSeconds
         << "s, even after re-verifying/restarting the container once; "
            "giving up";
}

::mlir::LogicalResult writeFileAtomically(DistributedKernelOp kernelOp,
                                          llvm::StringRef finalPath,
                                          llvm::StringRef text) {
  std::string tmpPath = (llvm::Twine(finalPath) + ".tmp").str();
  std::error_code ec;
  {
    llvm::raw_fd_ostream os(tmpPath, ec, llvm::sys::fs::OF_None);
    if (ec) {
      return kernelOp.emitError() << "failed to open '" << tmpPath
                                  << "' for writing: " << ec.message();
    }
    os << text;
  }
  if (auto renameEc = llvm::sys::fs::rename(tmpPath, finalPath)) {
    return kernelOp.emitError()
           << "failed to publish '" << finalPath << "': " << renameEc.message();
  }
  return success();
}

} // namespace

::mlir::LogicalResult ensureActPatchedKernel(DistributedKernelOp kernelOp,
                                             const ActWorkerConfig &config) {
  auto bodyModuleOrFailure = buildKernelBodyModule(kernelOp);
  if (failed(bodyModuleOrFailure)) {
    return failure(); // diagnostic already emitted
  }
  ModuleOp bodyModule = *bodyModuleOrFailure;
  auto eraseBodyModule = llvm::scope_exit([&] { bodyModule.erase(); });

  std::string bodyText;
  {
    llvm::raw_string_ostream os(bodyText);
    bodyModule.print(os);
  }

  // A real cryptographic hash, not a cheap rolling/positional one: a
  // collision here would silently splice one kernel's ACT-compiled body and
  // cost into an unrelated kernel with the same hash, a correctness bug,
  // not just a wasted cache lookup.
  std::string hash =
      llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(bodyText)),
                  /*LowerCase=*/true);

  std::string workerKey =
      (llvm::Twine(config.target) + "-uid" + llvm::Twine(getuid())).str();
  llvm::SmallString<128> workdir(config.workdirRoot);
  llvm::sys::path::append(workdir, workerKey);
  llvm::SmallString<128> cacheDir(workdir);
  llvm::sys::path::append(cacheDir, "cache");
  llvm::SmallString<128> cachePath(cacheDir);
  llvm::sys::path::append(cachePath, hash + ".mlir");

  bool cacheHit = llvm::sys::fs::exists(cachePath);
  std::string patchedText;
  if (cacheHit) {
    auto buf = llvm::MemoryBuffer::getFile(cachePath);
    if (!buf) {
      return kernelOp.emitError()
             << "failed to read ACT cost cache entry '" << cachePath
             << "': " << buf.getError().message();
    }
    patchedText = (*buf)->getBuffer().str();
  } else {
    if (auto ec = llvm::sys::fs::create_directories(workdir)) {
      return kernelOp.emitError() << "failed to create ACT worker directory '"
                                  << workdir << "': " << ec.message();
    }
    std::string containerName = "enzyme-act-worker-" + workerKey;
    auto dispatched =
        dispatchWithRetry(kernelOp, config, containerName, workdir, bodyText);
    if (failed(dispatched)) {
      return failure(); // diagnostic already emitted
    }
    patchedText = *dispatched;

    if (auto ec = llvm::sys::fs::create_directories(cacheDir)) {
      return kernelOp.emitError()
             << "failed to create ACT cost cache directory '" << cacheDir
             << "': " << ec.message();
    }
    if (failed(writeFileAtomically(kernelOp, cachePath, patchedText))) {
      return failure(); // diagnostic already emitted
    }
  }

  ParserConfig parserConfig(kernelOp.getContext());
  OwningOpRef<ModuleOp> patchedModule =
      parseSourceString<ModuleOp>(patchedText, parserConfig);
  if (!patchedModule) {
    return kernelOp.emitError()
           << "failed to parse ACT's patched module (from "
           << (cacheHit ? "the cost cache" : "a live ACT dispatch") << ")";
  }

  return spliceKernelBodyModule(kernelOp, *patchedModule);
}

} // namespace mlir::enzyme::distributed
