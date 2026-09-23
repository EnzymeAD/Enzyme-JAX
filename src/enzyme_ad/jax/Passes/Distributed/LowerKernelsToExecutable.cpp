#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include "src/enzyme_ad/jax/Dialect/Distributed/ActWorkerClient.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Utilities.h"
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_LOWERKERNELSTOEXECUTABLEPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

// Writes the standalone module a kernel would be dispatched to the external
// compiler as -- the same func.func-wrapped view buildKernelBodyModule
// builds for LowerKernelsPass, minus any sharding attributes, since there is
// no sharding left to describe by this point in the pipeline.
//
// This pass runs once per search candidate (LowerKernelsToExecutablePass
// sits at the end of buildDistributedSearchLoweringPipeline), so a naming
// scheme keyed only on a per-invocation kernel index would collide and
// overwrite dumps across candidates, across sibling kernels in the same
// walk, and across separate enzymexlamlir-opt runs sharing a directory.
// createUniqueFile sidesteps all of that with an atomically-created,
// randomly-suffixed name; nothing here needs to map a dump back to the
// candidate or kernel that produced it.
static LogicalResult dumpKernelModule(DistributedKernelOp kernelOp,
                                      StringRef directory) {
  if (auto ec = llvm::sys::fs::create_directories(directory)) {
    return kernelOp.emitError() << "failed to create kernel dump directory '"
                                << directory << "': " << ec.message();
  }

  SmallString<128> model(directory);
  llvm::sys::path::append(model, "kernel-%%%%%%.mlir");

  int fd;
  SmallString<128> path;
  if (auto ec = llvm::sys::fs::createUniqueFile(model, fd, path)) {
    return kernelOp.emitError()
           << "failed to create a unique kernel dump file under '" << directory
           << "': " << ec.message();
  }
  llvm::raw_fd_ostream os(fd, /*shouldClose=*/true);

  auto moduleOrFailure = buildKernelBodyModule(kernelOp);
  if (failed(moduleOrFailure)) {
    return failure(); // diagnostic already emitted
  }
  ModuleOp module = *moduleOrFailure;
  module.print(os);
  os << "\n";
  module.erase();
  return success();
}

// By now distributed-lower-kernels has already shrunk every kernel to its
// local (per-device) sizes, so a kernel body is just a plain, unsharded HLO
// subgraph, ready to be handed to ACT for actual executable lowering
// (compilation is per-kernel). This pass's job is the walk-and-dispatch
// boilerplate around that:
//   - dump-kernel-modules-to writes the wire format each kernel is
//     dispatched as, independent of whether real dispatch is also active.
//   - act-target/act-worker-dir-root/act-worker-script-dir drive real
//     dispatch through ensureActPatchedKernel (ActWorkerClient.h).
//   - act-patched-module-file splices a literal patched-module file back in
//     through spliceKernelBodyModule instead, for tests that want the
//     splice behavior without a real ACT/Docker round trip.
struct LowerKernelsToExecutablePass
    : public impl::LowerKernelsToExecutablePassBase<
          LowerKernelsToExecutablePass> {
  using LowerKernelsToExecutablePassBase::LowerKernelsToExecutablePassBase;

  void runOnOperation() override {
    bool anyRealDispatchOption = !actTarget.empty() ||
                                 !actWorkerDirRoot.empty() ||
                                 !actWorkerScriptDir.empty();
    bool realDispatch = !actTarget.empty() && !actWorkerDirRoot.empty() &&
                        !actWorkerScriptDir.empty();

    // Both of these are CLI-option misuse, a genuine input boundary (nothing
    // upstream validates pass options), so they get a real diagnostic rather
    // than an implicit "one wins" precedence.
    if (!actPatchedModuleFile.empty() && anyRealDispatchOption) {
      getOperation()->emitError()
          << "act-patched-module-file (a test-only seam that bypasses real "
             "ACT dispatch) is mutually exclusive with act-target/"
             "act-worker-dir-root/act-worker-script-dir (real dispatch); set "
             "one or the other, not both";
      signalPassFailure();
      return;
    }
    if (anyRealDispatchOption && !realDispatch) {
      getOperation()->emitError()
          << "act-target, act-worker-dir-root, and act-worker-script-dir "
             "must all be set together to activate real ACT dispatch (got "
             "act-target='"
          << actTarget << "', act-worker-dir-root='" << actWorkerDirRoot
          << "', act-worker-script-dir='" << actWorkerScriptDir << "')";
      signalPassFailure();
      return;
    }

    OwningOpRef<ModuleOp> testPatchedModule;
    if (!actPatchedModuleFile.empty()) {
      ParserConfig parserConfig(&getContext());
      testPatchedModule =
          parseSourceFile<ModuleOp>(actPatchedModuleFile, parserConfig);
      if (!testPatchedModule) {
        // parseSourceFile already emitted a diagnostic through the context's
        // error handler.
        signalPassFailure();
        return;
      }
    }

    // Shared across every kernel this pass walks: the worker/cache these
    // fields point at is keyed by (target, calling user), not by kernel.
    ActWorkerConfig config;
    if (realDispatch) {
      config.target = actTarget;
      config.workdirRoot = actWorkerDirRoot;
      config.workerScriptDir = actWorkerScriptDir;
    }

    auto result = getOperation()->walk([&](DistributedKernelOp kernelOp) {
      if (!dumpKernelModulesTo.empty() &&
          failed(dumpKernelModule(kernelOp, dumpKernelModulesTo))) {
        return WalkResult::interrupt();
      }
      if (testPatchedModule &&
          failed(spliceKernelBodyModule(kernelOp, *testPatchedModule))) {
        return WalkResult::interrupt();
      }
      if (realDispatch && failed(ensureActPatchedKernel(kernelOp, config))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
