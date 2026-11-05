#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "src/enzyme_ad/jax/Dialect/Perfify/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Perfify/Passes.h"
#include "src/enzyme_ad/jax/Passes/Perfify/PerfifyLeanEmitter.h"
#include "llvm/Support/ToolOutputFile.h"

namespace mlir {
namespace enzyme {
namespace perfify {
#define GEN_PASS_DEF_PERFIFYTOLEANPASS
#include "src/enzyme_ad/jax/Passes/Perfify/Passes.h.inc"
} // namespace perfify
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzyme::perfify;

namespace {

struct PerfifyToLeanPass
    : public enzyme::perfify::impl::PerfifyToLeanPassBase<PerfifyToLeanPass> {
  using PerfifyToLeanPassBase::PerfifyToLeanPassBase;

  LogicalResult writeLeanFile(StringRef path, StringRef lean) {
    std::string error;
    std::unique_ptr<llvm::ToolOutputFile> file = openOutputFile(path, &error);
    if (!file)
      return getOperation().emitError("cannot open Lean output file: ")
             << error;
    file->os() << lean;
    file->keep();
    return success();
  }

  void runOnOperation() override {
    if (failed(writeLeanFile(output, emitLean(getOperation()).text)))
      signalPassFailure();
  }
};

} // namespace
