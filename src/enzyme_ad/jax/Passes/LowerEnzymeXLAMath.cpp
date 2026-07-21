// This must come first for windows builds
#define _USE_MATH_DEFINES

#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"

#include "mhlo/IR/hlo_ops.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/ChloDecompositionUtils.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "lower-enzymexla-math"

#include <functional>

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERENZYMEXLAMATHPASS
#define GEN_PASS_DEF_LOWERENZYMEXLAMLPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::stablehlo;

namespace {
#include "src/enzyme_ad/jax/Passes/LowerEnzymeXLAMathPatterns.cpp.inc"

struct LowerBinomialProgressOpToStableHLO
    : public OpRewritePattern<enzymexla::BinomialProgressOp> {
  using OpRewritePattern<enzymexla::BinomialProgressOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(enzymexla::BinomialProgressOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value numIters = op.getNumSteps();
    Value budget = op.getBudget();

    Value one = stablehlo::ConstantOp::create(
        rewriter, loc, cast<ElementsAttr>(makeAttr(op.getType(), 1)));
    Value cond = stablehlo::OrOp::create(
        rewriter, loc,
        stablehlo::CompareOp::create(rewriter, loc, numIters, one,
                                     stablehlo::ComparisonDirection::EQ),
        stablehlo::CompareOp::create(rewriter, loc, budget, one,
                                     stablehlo::ComparisonDirection::EQ));

    auto ifOp =
        stablehlo::IfOp::create(rewriter, loc, TypeRange{one.getType()}, cond);

    { // true block
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.createBlock(&ifOp.getTrueBranch(), {}, {}, {});
      stablehlo::ReturnOp::create(rewriter, loc, one);
    }

    { // false block
      OpBuilder::InsertionGuard guard2(rewriter);
      rewriter.createBlock(&ifOp.getFalseBranch(), {}, {}, {});

      auto whileOp = stablehlo::WhileOp::create(rewriter, loc, {one, budget});
      SmallVector<Location> locs;
      locs.push_back(one.getLoc());
      locs.push_back(budget.getLoc());
      {
        OpBuilder::InsertionGuard guard(rewriter);
        Block *cond = rewriter.createBlock(&whileOp.getCond(), {},
                                           whileOp->getResultTypes(), locs);
        rewriter.setInsertionPointToEnd(cond);

        Value cmp = stablehlo::CompareOp::create(
            rewriter, loc, cond->getArgument(1), numIters,
            stablehlo::ComparisonDirection::LT);
        stablehlo::ReturnOp::create(rewriter, loc, cmp);
      }

      {
        OpBuilder::InsertionGuard guard(rewriter);
        Block *body = rewriter.createBlock(&whileOp.getBody(), {},
                                           whileOp->getResultTypes(), locs);
        rewriter.setInsertionPointToEnd(body);

        Value j =
            stablehlo::AddOp::create(rewriter, loc, body->getArgument(0), one);
        Value binom = stablehlo::DivOp::create(
            rewriter, loc,
            stablehlo::MulOp::create(
                rewriter, loc, body->getArgument(1),
                stablehlo::SubtractOp::create(
                    rewriter, loc,
                    stablehlo::AddOp::create(rewriter, loc, j, budget), one)),
            j);

        stablehlo::ReturnOp::create(rewriter, loc, ValueRange{j, binom});
      }

      Value binom = whileOp->getResult(1), j = whileOp->getResult(0);

      Value result = stablehlo::SelectOp::create(
          rewriter, loc,
          stablehlo::CompareOp::create(rewriter, loc, binom, numIters,
                                       stablehlo::ComparisonDirection::EQ),
          j, stablehlo::SubtractOp::create(rewriter, loc, j, one));

      stablehlo::ReturnOp::create(rewriter, loc, result);
    }

    Value val = ifOp.getResult(0);
    rewriter.replaceOp(op, val);

    return success();
  }
};

void lowerEnzymeXLAMath(Operation *op,
                        std::function<void()> signalPassFailure) {
  auto context = op->getContext();
  RewritePatternSet patterns(context);

  populateWithGenerated(patterns);

  patterns.add<LowerBinomialProgressOpToStableHLO>(context);

  GreedyRewriteConfig config;
  config.enableFolding();
  if (failed(applyPatternsGreedily(op, std::move(patterns), config))) {
    signalPassFailure();
  }

  // Verify that all illegal ops have been lowered
  auto walkResult = op->walk([&](Operation *local_op) {
    if (local_op->getName().getStringRef().starts_with("enzymexla.math.")) {
      local_op->emitError("Failed to lower enzymexla math operation");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted()) {
    signalPassFailure();
  }
}
} // namespace

struct LowerEnzymeXLAMathPass
    : public enzyme::impl::LowerEnzymeXLAMathPassBase<LowerEnzymeXLAMathPass> {
  using Base::Base;

  void runOnOperation() override {
    lowerEnzymeXLAMath(getOperation(), [this]() { signalPassFailure(); });
  }
};

// TODO: delete this once Reactant uses `lower-enzymexla-math` instead of
// `lower-enzymexla-ml`
struct LowerEnzymeXLAMLPass
    : public enzyme::impl::LowerEnzymeXLAMLPassBase<LowerEnzymeXLAMLPass> {
  using Base::Base;

  void runOnOperation() override {
    lowerEnzymeXLAMath(getOperation(), [this]() { signalPassFailure(); });
  }
};
