#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_DUMPDISTRIBUTEDLOGGINGCHECKPOINTPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

struct DumpDistributedLoggingCheckpointPass
    : public impl::DumpDistributedLoggingCheckpointPassBase<
          DumpDistributedLoggingCheckpointPass> {
  using DumpDistributedLoggingCheckpointPassBase::
      DumpDistributedLoggingCheckpointPassBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Logging needs both attributes:
    // - distributed.logging_root: set by --set-distributed-logging, and
    //   inherited by every scoring clone through ordinary attribute cloning.
    // - distributed.logging_candidate_uid: set by cloneAndApplyDecisions on
    //   each scoring clone only.
    //
    // The winner-finalization run at the end of distributed-search-strategies
    // applies the winning decisions in place on the original module, which
    // carries logging_root but no candidate uid. That is the expected shape
    // once logging is enabled: the winner's checkpoints were already captured
    // while it was a scoring clone, and finalizing it again must not produce
    // a second log entry. So a missing uid (with root present) is a no-op,
    // the same as logging being off.
    auto loggingRoot =
        moduleOp->getAttrOfType<StringAttr>("distributed.logging_root");
    if (!loggingRoot)
      return;
    auto candidateUid = moduleOp->getAttrOfType<IntegerAttr>(
        "distributed.logging_candidate_uid");
    if (!candidateUid)
      return;

    llvm::SmallString<128> irDir(loggingRoot.getValue());
    llvm::sys::path::append(irDir, "Candidates",
                            llvm::Twine(candidateUid.getInt()), "ir");
    if (auto ec = llvm::sys::fs::create_directories(irDir)) {
      moduleOp.emitError()
          << "dump-distributed-logging-checkpoint: failed to create '" << irDir
          << "': " << ec.message();
      return signalPassFailure();
    }

    llvm::SmallString<128> irPath(irDir);
    llvm::sys::path::append(irPath, checkpointName.getValue() + ".mlir");
    // Only one writer ever produces a given
    // Candidates/<uid>/ir/<checkpoint-name>.mlir, so a plain overwrite is
    // enough; no tmp-then-rename as in writeJsonAtomically.
    std::error_code ec;
    llvm::raw_fd_ostream os(irPath, ec, llvm::sys::fs::OF_None);
    if (ec) {
      moduleOp.emitError()
          << "dump-distributed-logging-checkpoint: failed to open '" << irPath
          << "' for writing: " << ec.message();
      return signalPassFailure();
    }
    moduleOp.print(os);
    os << "\n";

    if (countCollectivesAs.empty())
      return;

    int64_t numCollectives = 0;
    moduleOp.walk([&](DistributedCollectiveOp) { ++numCollectives; });
    Builder builder(moduleOp.getContext());
    moduleOp->setAttr(countCollectivesAs.getValue(),
                      builder.getI64IntegerAttr(numCollectives));
  }
};

} // namespace
} // namespace mlir::enzyme::distributed
