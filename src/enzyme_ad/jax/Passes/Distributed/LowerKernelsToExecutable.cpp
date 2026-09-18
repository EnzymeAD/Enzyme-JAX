#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

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

// Placeholder for the next lowering stage. By now distributed-lower-kernels
// has already shrunk every kernel to its local (per-device) sizes, so a
// kernel body is just a plain, unsharded HLO subgraph -- ready to be handed
// to an external compiler for actual executable lowering. That dispatch
// isn't wired up yet; this pass is only the walk that will drive it,
// compilation being per-kernel, plus a debug dump of the wire format each
// kernel would be sent as.
struct LowerKernelsToExecutablePass
    : public impl::LowerKernelsToExecutablePassBase<
          LowerKernelsToExecutablePass> {
  using LowerKernelsToExecutablePassBase::LowerKernelsToExecutablePassBase;

  void runOnOperation() override {
    auto result = getOperation()->walk([&](DistributedKernelOp kernelOp) {
      if (!dumpKernelModulesTo.empty() &&
          failed(dumpKernelModule(kernelOp, dumpKernelModulesTo))) {
        return WalkResult::interrupt();
      }
      // TODO: dispatch kernelOp's body to the external compiler and splice
      // the resulting executable back in.
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::enzyme::distributed
