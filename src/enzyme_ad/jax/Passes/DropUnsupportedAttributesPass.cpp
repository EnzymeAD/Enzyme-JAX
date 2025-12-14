#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"

#include "mlir/Interfaces/FunctionInterfaces.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "drop-unsupported-attributes"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_DROPUNSUPPORTEDATTRIBUTESPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzymexla;

struct RemoveMemoryEffectAttributes
    : public OpInterfaceRewritePattern<FunctionOpInterface> {
  using OpInterfaceRewritePattern<
      FunctionOpInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(FunctionOpInterface funcOp,
                                PatternRewriter &rewriter) const override {
    bool updated = false;

    // Drop argument attributes first
    for (int i = 0; i < funcOp.getNumArguments(); i++) {
      auto argAttrs = funcOp.getArgAttrs(i);
      if (argAttrs.empty())
        continue;

      SmallVector<NamedAttribute> newArgAttrs;
      for (auto attr : argAttrs) {
        if (attr.getName().getValue() != "enzymexla.memory_effects") {
          newArgAttrs.push_back(attr);
        } else {
          updated = true;
        }
      }

      funcOp.setArgAttrs(i, newArgAttrs);
    }

    // Drop enzymexla.memory_effects attribute
    if (funcOp.getOperation()->hasAttr(
            StringAttr::get(funcOp.getContext(), "enzymexla.memory_effects"))) {
      updated = true;
      funcOp.getOperation()->removeAttr("enzymexla.memory_effects");
    }

    return updated ? success() : failure();
  }
};

namespace {

struct DropUnsupportedAttributesPass
    : public enzyme::impl::DropUnsupportedAttributesPassBase<
          DropUnsupportedAttributesPass> {
  using Base::Base;

  void runOnOperation() override {
    auto context = getOperation()->getContext();
    RewritePatternSet patterns(context);

    if (enzymexla_memory_effects) {
      patterns.add<RemoveMemoryEffectAttributes>(context);
    }

    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }

    if (enzymexla_analysis_result) {
      auto moduleOp = getOperation();
      SmallVector<StringRef, 4> enzymexlaAnalysisResultAttrs = {
          "enzymexla.symmetric_matrix", "enzymexla.non_negative",
          "enzymexla.finite", "enzymexla.no_nan"};

      moduleOp.walk([&](Operation *op) {
        for (auto removeAttr : enzymexlaAnalysisResultAttrs) {
          if (op->hasAttr(removeAttr)) {
            op->removeAttr(removeAttr);
          }
        }
        return WalkResult::advance();
      });
    }
  }
};

} // namespace
