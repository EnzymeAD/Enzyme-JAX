//===- RaisingTransformOps.cpp - Definition of raising transform extension ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/enzyme_ad/jax/TransformOps/RaisingTransformOps.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "src/enzyme_ad/jax/Passes/EnzymeHLOPatterns.h"

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/TransformOps/RaisingTransformOps.cpp.inc"
#include "src/enzyme_ad/jax/TransformOps/RaisingTransformOpsImpl.cpp.inc"

using namespace mlir;

namespace mlir {
namespace transform {

struct RemoveIVs : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    if (!forOp.getRegion().hasOneBlock())
      return failure();
    unsigned numIterArgs = forOp.getNumRegionIterArgs();
    auto loc = forOp->getLoc();
    bool changed = false;
    llvm::SetVector<unsigned> removed;
    llvm::MapVector<unsigned, Value> steps;
    auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    for (unsigned i = 0; i < numIterArgs; i++) {
      auto ba = forOp.getRegionIterArgs()[i];
      auto init = forOp.getInits()[i];
      auto next = yield->getOperand(i);

      auto increment = next.getDefiningOp<arith::AddIOp>();
      if (!increment)
        continue;

      Value step = nullptr;
      if (increment.getLhs() == ba) {
        step = increment.getRhs();
      } else {
        step = increment.getLhs();
      }
      if (!step)
        continue;

      // If it dominates the loop entry
      if (!step.getParentRegion()->isProperAncestor(&forOp.getRegion()))
        continue;

      rewriter.setInsertionPointToStart(forOp.getBody());
      Value iterNum = rewriter.create<arith::SubIOp>(
          loc, forOp.getInductionVar(), forOp.getLowerBound());
      iterNum = rewriter.create<arith::DivSIOp>(loc, iterNum, forOp.getStep());

      Value replacementIV = rewriter.create<arith::MulIOp>(loc, iterNum, step);
      replacementIV = rewriter.create<arith::AddIOp>(loc, replacementIV, init);

      rewriter.replaceAllUsesWith(ba, replacementIV);

      removed.insert(i);
      steps.insert({i, step});
      changed = true;
    }

    if (!changed)
      return failure();

    SmallVector<Value> newInits;
    for (unsigned i = 0; i < numIterArgs; i++)
      if (!removed.contains(i))
        newInits.push_back(forOp.getInits()[i]);

    rewriter.setInsertionPoint(forOp);
    auto newForOp = rewriter.create<scf::ForOp>(loc, forOp.getLowerBound(),
                                                forOp.getUpperBound(),
                                                forOp.getStep(), newInits);
    if (!newForOp.getRegion().empty())
      newForOp.getRegion().front().erase();
    assert(newForOp.getRegion().empty());
    rewriter.inlineRegionBefore(forOp.getRegion(), newForOp.getRegion(),
                                newForOp.getRegion().begin());

    SmallVector<Value> newYields;
    for (unsigned i = 0; i < numIterArgs; i++)
      if (!removed.contains(i))
        newYields.push_back(yield->getOperand(i));

    rewriter.setInsertionPoint(yield);
    rewriter.replaceOpWithNewOp<scf::YieldOp>(yield, newYields);

    llvm::BitVector toDelete(numIterArgs + 1);
    for (unsigned i = 0; i < numIterArgs; i++)
      if (removed.contains(i))
        toDelete[i + 1] = true;
    newForOp.getBody()->eraseArguments(toDelete);

    rewriter.setInsertionPoint(newForOp);
    unsigned curNewRes = 0;
    for (unsigned i = 0; i < numIterArgs; i++) {
      auto result = forOp->getResult(i);
      if (removed.contains(i)) {
        if (result.use_empty())
          continue;

        rewriter.setInsertionPointAfter(forOp.getOperation());
        Value iterNum = rewriter.create<arith::SubIOp>(
            loc, forOp.getUpperBound(), forOp.getLowerBound());
        iterNum =
            rewriter.create<arith::DivSIOp>(loc, iterNum, forOp.getStep());

        Value afterLoop =
            rewriter.create<arith::MulIOp>(loc, iterNum, steps[i]);
        afterLoop =
            rewriter.create<arith::AddIOp>(loc, afterLoop, forOp.getInits()[i]);

        rewriter.replaceAllUsesWith(result, afterLoop);
      } else {
        rewriter.replaceAllUsesWith(result, newForOp->getResult(curNewRes++));
      }
    }

    forOp->getParentOp()->dump();
    rewriter.eraseOp(forOp);

    return success();
  }
};

} // namespace transform
} // namespace mlir

#include "src/enzyme_ad/jax/TransformOps/RaisingTransformPatterns.cpp.inc"

namespace {
class RaisingTransformExtension
    : public transform::TransformDialectExtension<RaisingTransformExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RaisingTransformExtension)
  using Base::Base;

  void init() {
    registerTransformOps<
#define GET_OP_LIST
#include "src/enzyme_ad/jax/TransformOps/RaisingTransformOps.cpp.inc"
        >();
  }
};
} // namespace

void mlir::enzyme::registerRaisingTransformExtension(
    DialectRegistry &registry) {
  registry.addExtensions<RaisingTransformExtension>();
}
