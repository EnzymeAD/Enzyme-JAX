#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Dialect/Perfify/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Perfify/Passes.h"
#include "llvm/ADT/MapVector.h"

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

/// A conditions op is unproven if any of its assumptions failed to check.
/// An assumption without a satres has not been checked, so it does not count.
static bool isUnproven(ConditionsOp cond) {
  return cond
      .walk([](AssumeOp assume) {
        std::optional<bool> satres = assume.getSatres();
        return satres && !*satres ? WalkResult::interrupt()
                                  : WalkResult::advance();
      })
      .wasInterrupted();
}

struct PerfifyToLeanPass
    : public enzyme::perfify::impl::PerfifyToLeanPassBase<PerfifyToLeanPass> {
  using PerfifyToLeanPassBase::PerfifyToLeanPassBase;

  // Unproven conditions, keyed by the function they constrain, in the order
  // they appear in the module.
  llvm::MapVector<StringAttr, SmallVector<ConditionsOp>> unprovenConds;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    module.walk([&](ConditionsOp cond) {
      if (cond.getVerifyHuh() && isUnproven(cond))
        unprovenConds[cond.getFuncHandleAttr().getAttr()].push_back(cond);
    });

    llvm::outs() << "-- Unproven conditions:";
    for (auto &[func, conds] : unprovenConds)
      llvm::outs() << " @" << func.getValue();
    llvm::outs() << "\n";
  }
};

} // namespace
