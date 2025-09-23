#ifndef ENZYME_AD_JAX_PASSES_BARRIER_ELIMINATION_H
#define ENZYME_AD_JAX_PASSES_BARRIER_ELIMINATION_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/BarrierOpt.h"
#include "src/enzyme_ad/jax/Utils.h"

namespace mlir {
namespace enzymexla {

template <bool NotTopLevel = false>
class BarrierElim final
    : public mlir::OpRewritePattern<mlir::enzymexla::BarrierOp> {
public:
  using mlir::OpRewritePattern<mlir::enzymexla::BarrierOp>::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::enzymexla::BarrierOp barrier,
                  mlir::PatternRewriter &rewriter) const override {
    using namespace mlir;
    using namespace enzymexla;
    if (!BarrierOpt)
      return failure();
    // Remove if it only sync's constant indices.
    if (llvm::all_of(barrier.getOperands(), [](mlir::Value v) {
          IntegerAttr constValue;
          return matchPattern(v, m_Constant(&constValue));
        })) {
      rewriter.eraseOp(barrier);
      return success();
    }

    Operation *op = barrier;
    if (NotTopLevel &&
        isa<mlir::scf::ParallelOp, mlir::affine::AffineParallelOp>(
            barrier->getParentOp()))
      return failure();

    {
      SmallVector<mlir::MemoryEffects::EffectInstance> beforeEffects;
      mlir::enzyme::getEffectsBefore(op, beforeEffects, /*stopAtBarrier*/ true);

      SmallVector<mlir::MemoryEffects::EffectInstance> afterEffects;
      mlir::enzyme::getEffectsAfter(op, afterEffects, /*stopAtBarrier*/ false);

      bool conflict = false;
      for (auto before : beforeEffects)
        for (auto after : afterEffects) {
          if (mlir::enzyme::mayAlias(before, after)) {
            // Read, read is okay
            if (isa<mlir::MemoryEffects::Read>(before.getEffect()) &&
                isa<mlir::MemoryEffects::Read>(after.getEffect())) {
              continue;
            }

            // Write, write is not okay because may be different offsets and the
            // later must subsume other conflicts are invalid.
            conflict = true;
            break;
          }
        }

      if (!conflict) {
        rewriter.eraseOp(barrier);
        return success();
      }
    }

    {
      SmallVector<mlir::MemoryEffects::EffectInstance> beforeEffects;
      mlir::enzyme::getEffectsBefore(op, beforeEffects,
                                     /*stopAtBarrier*/ false);

      SmallVector<mlir::MemoryEffects::EffectInstance> afterEffects;
      mlir::enzyme::getEffectsAfter(op, afterEffects, /*stopAtBarrier*/ true);

      bool conflict = false;
      for (auto before : beforeEffects)
        for (auto after : afterEffects) {
          if (mlir::enzyme::mayAlias(before, after)) {
            // Read, read is okay
            if (isa<mlir::MemoryEffects::Read>(before.getEffect()) &&
                isa<mlir::MemoryEffects::Read>(after.getEffect())) {
              continue;
            }
            // Write, write is not okay because may be different offsets and the
            // later must subsume other conflicts are invalid.
            conflict = true;
            break;
          }
        }

      if (!conflict) {
        rewriter.eraseOp(barrier);
        return success();
      }
    }

    return failure();
  }
};
} // namespace enzymexla
} // namespace mlir

#endif