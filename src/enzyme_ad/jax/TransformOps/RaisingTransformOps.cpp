//===- RaisingTransformOps.cpp - Definition of raising transform extension ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/enzyme_ad/jax/TransformOps/RaisingTransformOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/TransformOps/RaisingTransformOps.cpp.inc"
#include "src/enzyme_ad/jax/TransformOps/RaisingTransformOpsImpl.cpp.inc"

using namespace mlir;

namespace mlir {
namespace transform {

LogicalResult RemoveIVs::matchAndRewrite(scf::ForOp forOp,
                                         PatternRewriter &rewriter) const {
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
    Value iterNum = rewriter.create<arith::SubIOp>(loc, forOp.getInductionVar(),
                                                   forOp.getLowerBound());
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
      Value iterNum = rewriter.create<arith::SubIOp>(loc, forOp.getUpperBound(),
                                                     forOp.getLowerBound());
      iterNum = rewriter.create<arith::DivSIOp>(loc, iterNum, forOp.getStep());

      Value afterLoop = rewriter.create<arith::MulIOp>(loc, iterNum, steps[i]);
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

static inline void clearBlock(mlir::Block *block,
                              mlir::RewriterBase &rewriter) {
  for (auto &op : llvm::make_early_inc_range(llvm::reverse(*block))) {
    assert(op.use_empty() && "expected 'op' to have no uses");
    rewriter.eraseOp(&op);
  }
}

static mlir::Value createConstantInt(RewriterBase &rewriter, Location loc,
                                     Type ty, int64_t v) {
  if (ty.isIndex())
    return rewriter.create<arith::ConstantIndexOp>(loc, v);
  else
    return rewriter.create<arith::ConstantIntOp>(loc, v, ty);
}

static std::optional<int64_t> getConstant(Operation *op) {
  if (auto cst = dyn_cast_or_null<arith::ConstantIntOp>(op)) {
    return cst.value();
  } else if (auto cst = dyn_cast_or_null<arith::ConstantIndexOp>(op)) {
    return cst.value();
  } else if (auto cst = dyn_cast_or_null<LLVM::ConstantOp>(op)) {
    if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue()))
      return intAttr.getValue().getSExtValue();
  }
  return {};
}

static std::optional<int64_t> getConstant(Value v) {
  Operation *op = v.getDefiningOp();
  if (op)
    return getConstant(op);
  return {};
}

/// Returns `true` if the loop has a form expected by interchange patterns.
static bool isNormalized(scf::ForOp op) {
  auto lb = getConstant(op.getLowerBound());
  auto step = getConstant(op.getStep());
  if (!lb || !step)
    return false;
  return *lb == 0 && *step == 1;
}

#define DEBUG_TYPE "normalize-loop"
#define DBGS llvm::dbgs

LogicalResult NormalizeLoop::matchAndRewrite(scf::ForOp op,
                                             PatternRewriter &rewriter) const {
  using namespace arith;
  if (isNormalized(op) ||
      !isa<scf::ParallelOp, affine::AffineParallelOp>(op->getParentOp())) {
    LLVM_DEBUG(DBGS() << "[normalize-loop] loop already normalized\n");
    return failure();
  }

  rewriter.setInsertionPoint(op);
  Value zero = createConstantInt(rewriter, op.getLoc(),
                                 op.getInductionVar().getType(), 0);
  Value one = createConstantInt(rewriter, op.getLoc(),
                                op.getInductionVar().getType(), 1);

  Value difference = rewriter.create<SubIOp>(op.getLoc(), op.getUpperBound(),
                                             op.getLowerBound());
  Value tripCount = rewriter.create<AddIOp>(
      op.getLoc(),
      rewriter.create<DivUIOp>(
          op.getLoc(), rewriter.create<SubIOp>(op.getLoc(), difference, one),
          op.getStep()),
      one);
  auto newForOp = rewriter.create<scf::ForOp>(op.getLoc(), zero, tripCount, one,
                                              op.getInits());
  clearBlock(newForOp.getBody(), rewriter);
  rewriter.setInsertionPointToStart(newForOp.getBody());
  Value scaled = rewriter.create<MulIOp>(
      op.getLoc(), newForOp.getInductionVar(), op.getStep());
  Value iv = rewriter.create<AddIOp>(op.getLoc(), op.getLowerBound(), scaled);
  SmallVector<Value> newArgs(newForOp.getRegion().args_begin(),
                             newForOp.getRegion().args_end());
  newArgs[0] = iv;
  rewriter.inlineBlockBefore(op.getBody(), newForOp.getBody(),
                             newForOp.getBody()->end(), newArgs);
  rewriter.replaceOp(op, newForOp->getResults());
  return success();
}

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
