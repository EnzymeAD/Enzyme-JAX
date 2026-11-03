#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <ctime>
#include <string>

namespace mlir::enzyme::distributed {

#define GEN_PASS_DEF_SETDISTRIBUTEDLOGGINGPASS
#include "src/enzyme_ad/jax/Passes/Distributed/Passes.h.inc"

namespace {

// A sortable, human-legible disambiguator alongside createUniqueDirectory's
// own random suffix; no precision or timezone-awareness is needed beyond that.
static std::string currentTimestamp() {
  std::time_t now = std::time(nullptr);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", std::localtime(&now));
  return std::string(buf);
}

struct SetDistributedLoggingPass
    : public impl::SetDistributedLoggingPassBase<SetDistributedLoggingPass> {
  using SetDistributedLoggingPassBase::SetDistributedLoggingPassBase;

  void runOnOperation() override {
    if (root.empty())
      return; // Logging off.

    ModuleOp moduleOp = getOperation();

    llvm::SmallString<128> absoluteRoot(root);
    // createUniqueDirectory resolves a relative model against the system temp
    // directory (see its createUniquePath), so a relative `root` is made
    // absolute against the cwd here first.
    if (auto ec = llvm::sys::fs::make_absolute(absoluteRoot)) {
      moduleOp.emitError() << "set-distributed-logging: failed to resolve '"
                           << root << "' to an absolute path: " << ec.message();
      return signalPassFailure();
    }
    // createUniqueDirectory creates only the leaf it mints and fails on a
    // missing parent, so the master logging directory must exist first
    // (create_directories is idempotent). dumpKernelModule
    // (AnnotateKernelLatency.cpp) does the same for its dump directory.
    if (auto ec = llvm::sys::fs::create_directories(absoluteRoot)) {
      moduleOp.emitError() << "set-distributed-logging: failed to create "
                              "master logging directory '"
                           << absoluteRoot << "': " << ec.message();
      return signalPassFailure();
    }

    std::string namePart = moduleName.empty() ? "run" : moduleName.getValue();
    llvm::SmallString<128> prefix(absoluteRoot);
    llvm::sys::path::append(prefix, namePart + "-" + currentTimestamp());

    llvm::SmallString<128> runDir;
    if (auto ec = llvm::sys::fs::createUniqueDirectory(prefix, runDir)) {
      moduleOp.emitError()
          << "set-distributed-logging: failed to create a unique run "
             "directory under '"
          << prefix << "': " << ec.message();
      return signalPassFailure();
    }

    for (llvm::StringRef sub : {"Main", "Candidates"}) {
      llvm::SmallString<128> subDir(runDir);
      llvm::sys::path::append(subDir, sub);
      if (auto ec = llvm::sys::fs::create_directories(subDir)) {
        moduleOp.emitError() << "set-distributed-logging: failed to create '"
                             << subDir << "': " << ec.message();
        return signalPassFailure();
      }
    }

    moduleOp->setAttr("distributed.logging_root",
                      StringAttr::get(moduleOp.getContext(), runDir));
    if (!moduleName.empty())
      moduleOp->setAttr("distributed.logging_module_name",
                        StringAttr::get(moduleOp.getContext(), moduleName));
    if (!commit.empty())
      moduleOp->setAttr("distributed.logging_commit",
                        StringAttr::get(moduleOp.getContext(), commit));
  }
};

} // namespace
} // namespace mlir::enzyme::distributed
