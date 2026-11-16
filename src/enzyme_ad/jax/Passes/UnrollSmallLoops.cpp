//===- UnrollSmallLoops.cpp - Fully unroll loops of few iterations --------===//
//
// Fully unrolls every scf.for and affine.for whose trip count is a compile
// time constant of at most `max_trip_count` whose unrolled body is at most
// `max_nested_ops` operations, and always one of no iteration or a single
// one. A loop over the components of a vector, `for (int c = 0;
// c < 3; ++c)`, written out leaves each copy with `c` a constant: a bound
// chosen by the component, `(c == 2) ? D1D : D1D - 1`, is then a symbol,
// where over the loop it was a function of the induction variable that no
// affine bound expresses.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_UNROLLSMALLLOOPS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;

namespace {
struct UnrollSmallLoops
    : public enzyme::impl::UnrollSmallLoopsBase<UnrollSmallLoops> {
  using UnrollSmallLoopsBase::UnrollSmallLoopsBase;

  void runOnOperation() override {
    // A loop of no iteration is its initial values, and one of a single
    // iteration is its body, whatever the maximum: neither grows anything.
    uint64_t most = std::max<int64_t>(max_trip_count, 1);
    // Innermost first: an outer loop then copies its inner loops already
    // unrolled.
    SmallVector<std::pair<Operation *, uint64_t>> loops;
    getOperation()->walk([&](Operation *op) {
      std::optional<uint64_t> trips;
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        if (std::optional<APInt> count = forOp.getStaticTripCount())
          if (count->getActiveBits() <= 64)
            trips = count->getZExtValue();
      } else if (auto forOp = dyn_cast<affine::AffineForOp>(op)) {
        trips = affine::getConstantTripCount(forOp);
      }
      if (trips && *trips <= most)
        loops.emplace_back(op, *trips);
    });
    for (auto [op, trips] : loops) {
      // What unrolling leaves in place of the loop: its body, inner loops
      // already unrolled, once per iteration. A nest of small loops
      // multiplies, so the size is what is bounded, not each trip count.
      if (trips >= 2 && max_nested_ops >= 0) {
        int64_t nested = 0;
        op->walk([&](Operation *inner) {
          if (inner != op && !inner->hasTrait<OpTrait::IsTerminator>())
            ++nested;
        });
        if ((int64_t)trips * nested > max_nested_ops)
          continue;
      }
      if (trips == 0) {
        auto loop = cast<LoopLikeOpInterface>(op);
        SmallVector<Value> inits(loop.getInits());
        op->replaceAllUsesWith(inits);
        op->erase();
      } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        (void)loopUnrollFull(forOp);
      } else {
        (void)affine::loopUnrollFull(cast<affine::AffineForOp>(op));
      }
    }
  }
};
} // namespace
