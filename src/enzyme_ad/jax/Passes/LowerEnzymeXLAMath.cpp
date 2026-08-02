// This must come first for windows builds
#define _USE_MATH_DEFINES

#include "Enzyme/MLIR/Dialect/Dialect.h"
#include "Enzyme/MLIR/Dialect/Ops.h"
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

template <typename T>
static stablehlo::ConstantOp
createConstantOpFromScalar(PatternRewriter &rewriter, Location loc, Type type,
                           T value) {
  return stablehlo::ConstantOp::create(
      rewriter, loc, type,
      cast<ElementsAttr>(mlir::enzyme::makeAttr(type, value)));
}

namespace {
#include "src/enzyme_ad/jax/Passes/LowerEnzymeXLAMathPatterns.cpp.inc"

// Lower a tensor-typed enzyme.binomial_progress(n, s) to the Revolve advance
// distance. Enzyme's own lower-enzyme-binomial-progress handles the scalar form
// on scf/arith and skips tensors, leaving them to this pattern; the two mirror
// each other op for op and must agree with BinomialProgressConstProp in
// EnzymeHLOOpt.cpp:
//
//   %r = if (n <= 1) or (s <= 1) {
//     // n <= 1 yields n (0 or 1); otherwise s <= 1 advances the whole
//     // remainder, which is also n. Both degenerate cases are just n.
//     return n
//   } else {
//     // smallest t with beta = C(s+t, t) >= n
//     %w:2 = while (%t = 0, %beta = 1) cond(%beta < n) {
//       %t2 = %t + 1; %beta2 = %beta * (s + %t2) / %t2
//     }
//     // window [n - beta(s-1,t), beta(s,t-1)], clamped, then the midpoint
//     %lo = max(n - %beta * s / (s + %t), 1)
//     %hi = min(%beta * %t / (s + %t), n - 1)
//     return max(min((%lo + %hi) / 2, n - (s - 1)), 1)
//   }
//
// The guard must be a branch, not a select: for s <= 1 the update leaves %beta
// at 1 and the loop would spin forever.
struct LowerBinomialProgressOpToStableHLO
    : public OpRewritePattern<enzyme::BinomialProgressOp> {
  using OpRewritePattern<enzyme::BinomialProgressOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(enzyme::BinomialProgressOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value n = op.getNumSteps();
    Value s = op.getBudget();

    // Scalar operands are Enzyme's to lower, onto scf/arith.
    if (!isa<TensorType>(op.getType()))
      return failure();

    auto constOfType = [&](int64_t v) -> Value {
      return stablehlo::ConstantOp::create(
          rewriter, loc, cast<ElementsAttr>(makeAttr(op.getType(), v)));
    };

    Value zero = constOfType(0);
    Value one = constOfType(1);

    // Guard both degenerate cases.
    Value nSmall = stablehlo::CompareOp::create(
        rewriter, loc, n, one, stablehlo::ComparisonDirection::LE);
    Value sSmall = stablehlo::CompareOp::create(
        rewriter, loc, s, one, stablehlo::ComparisonDirection::LE);
    Value cond = stablehlo::OrOp::create(rewriter, loc, nSmall, sSmall);

    auto ifOp =
        stablehlo::IfOp::create(rewriter, loc, TypeRange{op.getType()}, cond);

    { // true block
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.createBlock(&ifOp.getTrueBranch(), {}, {}, {});
      stablehlo::ReturnOp::create(rewriter, loc, n);
    }

    { // false block
      OpBuilder::InsertionGuard guard2(rewriter);
      rewriter.createBlock(&ifOp.getFalseBranch(), {}, {}, {});

      // Smallest t with beta = C(s + t, t) >= n, carrying (t, beta).
      auto whileOp = stablehlo::WhileOp::create(rewriter, loc, {zero, one});
      SmallVector<Location> locs{zero.getLoc(), one.getLoc()};
      { // continue while beta < n
        OpBuilder::InsertionGuard guard(rewriter);
        Block *cond = rewriter.createBlock(&whileOp.getCond(), {},
                                           whileOp->getResultTypes(), locs);
        rewriter.setInsertionPointToEnd(cond);

        Value cmp =
            stablehlo::CompareOp::create(rewriter, loc, cond->getArgument(1), n,
                                         stablehlo::ComparisonDirection::LT);
        stablehlo::ReturnOp::create(rewriter, loc, cmp);
      }

      // t' = t + 1; beta' = beta * (s + t') / t'. That steps C(s+t-1, t-1) to
      // C(s+t, t), so the division is exact.
      {
        OpBuilder::InsertionGuard guard(rewriter);
        Block *body = rewriter.createBlock(&whileOp.getBody(), {},
                                           whileOp->getResultTypes(), locs);
        rewriter.setInsertionPointToEnd(body);

        Value t2 =
            stablehlo::AddOp::create(rewriter, loc, body->getArgument(0), one);
        Value beta2 = stablehlo::DivOp::create(
            rewriter, loc,
            stablehlo::MulOp::create(
                rewriter, loc, body->getArgument(1),
                stablehlo::AddOp::create(rewriter, loc, s, t2)),
            t2);

        stablehlo::ReturnOp::create(rewriter, loc, ValueRange{t2, beta2});
      }

      Value t = whileOp->getResult(0), beta = whileOp->getResult(1);

      // beta(s-1, t) = beta * s / (s + t) and beta(s, t-1) = beta * t / (s +
      // t), both exact in integers. Every advance between n - beta(s-1,t) and
      // beta(s, t-1) attains the optimal repetition count; clamp that window to
      // [1, n-1] -- so the caller always progresses and always leaves a tail --
      // and take its midpoint, since either edge can collapse onto the clamp
      // and waste a checkpoint on a one-step advance.
      Value sPlusT = stablehlo::AddOp::create(rewriter, loc, s, t);
      Value loRaw = stablehlo::SubtractOp::create(
          rewriter, loc, n,
          stablehlo::DivOp::create(
              rewriter, loc, stablehlo::MulOp::create(rewriter, loc, beta, s),
              sPlusT));
      Value hiRaw = stablehlo::DivOp::create(
          rewriter, loc, stablehlo::MulOp::create(rewriter, loc, beta, t),
          sPlusT);

      Value lo = stablehlo::MaxOp::create(rewriter, loc, loRaw, one);
      Value hi = stablehlo::MinOp::create(
          rewriter, loc, hiRaw,
          stablehlo::SubtractOp::create(rewriter, loc, n, one));
      Value two = constOfType(2);
      Value sum = stablehlo::AddOp::create(rewriter, loc, lo, hi);
      Value mid = stablehlo::DivOp::create(rewriter, loc, sum, two);

      // Leave a step for each of the s-1 checkpoints still to be placed.
      // Without this the advances can exhaust the interval before the slots run
      // out, and a caller walking one slot per iteration then records slots at
      // a step past the end, holding the final state rather than a checkpoint.
      Value cap = stablehlo::SubtractOp::create(
          rewriter, loc, n,
          stablehlo::SubtractOp::create(rewriter, loc, s, one));
      Value result = stablehlo::MaxOp::create(
          rewriter, loc, stablehlo::MinOp::create(rewriter, loc, mid, cap),
          one);

      stablehlo::ReturnOp::create(rewriter, loc, result);
    }

    rewriter.replaceOp(op, ifOp.getResult(0));

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

  // Verify that all illegal ops have been lowered. A tensor-typed
  // enzyme.binomial_progress counts: nothing downstream of here lowers it,
  // since Enzyme's own pass only handles the scalar form.
  auto walkResult = op->walk([&](Operation *local_op) {
    if (auto bp = dyn_cast<enzyme::BinomialProgressOp>(local_op)) {
      if (!isa<TensorType>(bp.getType()))
        return WalkResult::advance();
      bp->emitError("Failed to lower enzyme.binomial_progress");
      return WalkResult::interrupt();
    }
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
