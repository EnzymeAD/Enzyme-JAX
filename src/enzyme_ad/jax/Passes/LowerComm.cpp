#include "src/enzyme_ad/jax/Dialect/Comm/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Comm/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

namespace mlir::enzymexla::comm {
#define GEN_PASS_DEF_LOWERCOMMPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace mlir::enzymexla::comm

using namespace mlir;

struct LowerCommPass
    : public mlir::enzymexla::comm::impl::LowerCommPassBase<LowerCommPass> {
  using Base::Base;

  void runOnOperation() override {
    auto *context = getOperation()->getContext();

    RewritePatternSet patterns(context);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};