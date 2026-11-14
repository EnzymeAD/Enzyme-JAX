#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Dialect/Perfify/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Perfify/Passes.h"

namespace mlir {
namespace enzyme {
namespace perfify {
#define GEN_PASS_DEF_STRIPPERFIFYPASS
#include "src/enzyme_ad/jax/Passes/Perfify/Passes.h.inc"
} // namespace perfify
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzyme::perfify;

namespace {

struct StripPerfifyPass
    : public enzyme::perfify::impl::StripPerfifyPassBase<StripPerfifyPass> {
  using StripPerfifyPassBase::StripPerfifyPassBase;

  void runOnOperation() override {
    SmallVector<Operation *> toErase;
    getOperation()->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (!isa_and_nonnull<PerfifyDialect>(op->getDialect()))
        return WalkResult::advance();
      toErase.push_back(op);
      return WalkResult::skip();
    });

    for (Operation *op : llvm::reverse(toErase)) {
      if (!op->use_empty()) {
        op->emitOpError("result is used outside of Perfify ops");
        return signalPassFailure();
      }
      op->erase();
    }
  }
};
} // namespace
