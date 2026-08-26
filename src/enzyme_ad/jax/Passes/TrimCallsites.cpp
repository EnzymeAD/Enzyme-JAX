#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"

#include "mlir/Interfaces/FunctionInterfaces.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "trim-callsites"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_TRIMCALLSITES
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;

namespace {

struct TrimCallsites : public enzyme::impl::TrimCallsitesBase<TrimCallsites> {
  using Base::Base;

  mlir::Location trim(mlir::Location loc, SmallVector<StringRef, 1> &prefixes) {
    if (auto csl = dyn_cast<CallSiteLoc>(loc)) {
      auto callee = trim(csl.getCallee(), prefixes);
      auto caller = trim(csl.getCaller(), prefixes);
      if (isa<UnknownLoc>(callee)) {
        return caller;
      }
      if (auto nl = dyn_cast<mlir::NameLoc>(callee)) {
        for (auto prefix : prefixes)
          if (nl.getName().getValue().starts_with(prefix))
            return caller;
      }
      return CallSiteLoc::get(callee, caller);
    }
    return loc;
  }

  void runOnOperation() override {
    SmallVector<StringRef, 1> prefixes;
    StringRef(to_trim).split(prefixes, ";");

    getOperation()->walk([&](Operation *op) {
      op->setLoc(trim(op->getLoc(), prefixes));
      for (auto &reg : op->getRegions()) {
        for (auto &blk : reg.getBlocks()) {
          for (auto &arg : blk.getArguments()) {
            arg.setLoc(trim(arg.getLoc(), prefixes));
          }
        }
      }
    });
  }
};

} // namespace
