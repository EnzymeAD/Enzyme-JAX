//===- CanonicalizeLoops.cpp - canonicalize affine loops ------------------- //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include <numeric>

#define DEBUG_TYPE "affine-int-range-analysis"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_CANONICALIZELOOPSPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::affine;
using namespace mlir::dataflow;
using namespace mlir::enzyme;

namespace {

struct RemoveAffineParallelSingleIter
    : public OpRewritePattern<affine::AffineParallelOp> {
  using OpRewritePattern<affine::AffineParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineParallelOp op,
                                PatternRewriter &rewriter) const override {
    // Reductions are not supported yet.
    if (!op.getReductions().empty())
      return failure();

    SmallVector<AffineExpr> lbounds(op.getLowerBoundsMap().getResults().begin(),
                                    op.getLowerBoundsMap().getResults().end());
    SmallVector<AffineExpr> ubounds(op.getUpperBoundsMap().getResults().begin(),
                                    op.getUpperBoundsMap().getResults().end());

    SmallVector<int32_t> lboundGroup;
    SmallVector<int32_t> uboundGroup;
    for (auto lb : op.getLowerBoundsGroups())
      lboundGroup.push_back(lb.getZExtValue());
    for (auto ub : op.getUpperBoundsGroups())
      uboundGroup.push_back(ub.getZExtValue());

    SmallVector<int64_t> steps;
    for (auto step : op.getSteps())
      steps.push_back(step);

    Block *tmpBlk = new Block();
    SmallVector<Value> replacements;

    bool changed = false;
    for (ssize_t idx = steps.size() - 1; idx >= 0; idx--) {
      replacements.insert(replacements.begin(),
                          tmpBlk->insertArgument((unsigned)0,
                                                 op.getIVs()[idx].getType(),
                                                 op.getIVs()[idx].getLoc()));
      if (lboundGroup[idx] != 1)
        continue;
      if (uboundGroup[idx] != 1)
        continue;
      size_t loff = 0;
      for (size_t i = 0; i < idx; i++)
        loff += lboundGroup[i];

      size_t uoff = 0;
      for (size_t i = 0; i < idx; i++)
        uoff += uboundGroup[i];

      auto lb = dyn_cast<AffineConstantExpr>(lbounds[loff]);
      if (!lb)
        continue;
      auto ub = dyn_cast<AffineConstantExpr>(ubounds[uoff]);
      if (!ub)
        continue;
      if (lb.getValue() >= ub.getValue())
        continue;
      if (lb.getValue() + steps[idx] >= ub.getValue()) {
        tmpBlk->eraseArgument(0);
        replacements[0] =
            rewriter.create<arith::ConstantIndexOp>(op.getLoc(), lb.getValue());
        lboundGroup.erase(lboundGroup.begin() + idx);
        uboundGroup.erase(uboundGroup.begin() + idx);
        lbounds.erase(lbounds.begin() + loff);
        ubounds.erase(ubounds.begin() + uoff);
        steps.erase(steps.begin() + idx);
        changed = true;
        continue;
      }
      continue;
    }
    if (!changed) {
      delete tmpBlk;
      return failure();
    }

    if (steps.size() == 0) {
      delete tmpBlk;

      auto yld = cast<affine::AffineYieldOp>(op.getBody()->getTerminator());
      SmallVector<Value> toRet(yld.getOperands());
      rewriter.eraseOp(yld);
      rewriter.inlineBlockBefore(op.getBody(), op, replacements);
      rewriter.replaceOp(op, toRet);
    } else {

      affine::AffineParallelOp affineLoop =
          rewriter.create<affine::AffineParallelOp>(
              op.getLoc(), op.getResultTypes(), op.getReductions(),
              AffineMapAttr::get(
                  AffineMap::get(op.getLowerBoundsMap().getNumDims(),
                                 op.getLowerBoundsMap().getNumSymbols(),
                                 lbounds, op.getContext())),
              rewriter.getI32TensorAttr(lboundGroup),
              AffineMapAttr::get(
                  AffineMap::get(op.getUpperBoundsMap().getNumDims(),
                                 op.getUpperBoundsMap().getNumSymbols(),
                                 ubounds, op.getContext())),
              rewriter.getI32TensorAttr(uboundGroup),
              rewriter.getI64ArrayAttr(steps), op.getOperands());

      affineLoop.getRegion().getBlocks().push_back(tmpBlk);

      rewriter.mergeBlocks(op.getBody(), affineLoop.getBody(), replacements);
      rewriter.replaceOp(op, affineLoop->getResults());
    }

    return success();
  }
};

namespace {

/// Integer range analysis determines the integer value range of SSA values
/// using operations that define `InferIntRangeInterface` and also sets the
/// range of iteration indices of loops with known bounds.
///
/// This analysis depends on DeadCodeAnalysis, and will be a silent no-op
/// if DeadCodeAnalysis is not loaded in the same solver context.
class AffineIntegerRangeAnalysis
    : public SparseForwardDataFlowAnalysis<IntegerValueRangeLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  /// At an entry point, we cannot reason about interger value ranges.
  void setToEntryState(IntegerValueRangeLattice *lattice) override {
    propagateIfChanged(lattice, lattice->join(IntegerValueRange::getMaxRange(
                                    lattice->getAnchor())));
  }

  /// Visit an operation. Invoke the transfer function on each operation that
  /// implements `InferIntRangeInterface`.
  LogicalResult
  visitOperation(Operation *op,
                 ArrayRef<const IntegerValueRangeLattice *> operands,
                 ArrayRef<IntegerValueRangeLattice *> results) override;

  /// Visit block arguments or operation results of an operation with region
  /// control-flow for which values are not defined by region control-flow. This
  /// function calls `InferIntRangeInterface` to provide values for block
  /// arguments or tries to reduce the range on loop induction variables with
  /// known bounds.
  void
  visitNonControlFlowArguments(Operation *op, const RegionSuccessor &successor,
                               ArrayRef<IntegerValueRangeLattice *> argLattices,
                               unsigned firstIndex) override;

  /// Gets the constant lower and upper bounds for a given index of an
  /// AffineParallelOp. The upper bound is adjusted to be inclusive (subtracts 1
  /// from the exclusive bound).
  ///
  /// If the bounds cannot be determined statically, returns [SignedMinValue,
  /// SignedMaxValue].
  ///
  /// Example:
  ///   affine.parallel (%i) = (0) to (10) {
  ///     // getBoundsFromAffineParallel(op, 0) returns {0, 9}
  ///   }
  ConstantIntRanges getBoundsFromAffineParallel(affine::AffineParallelOp loop,
                                                size_t idx) {
    SmallVector<AffineExpr> lbounds(
        loop.getLowerBoundsMap().getResults().begin(),
        loop.getLowerBoundsMap().getResults().end());
    SmallVector<AffineExpr> ubounds(
        loop.getUpperBoundsMap().getResults().begin(),
        loop.getUpperBoundsMap().getResults().end());

    SmallVector<int32_t> lboundGroup;
    SmallVector<int32_t> uboundGroup;
    for (auto lb : loop.getLowerBoundsGroups())
      lboundGroup.push_back(lb.getZExtValue());
    for (auto ub : loop.getUpperBoundsGroups())
      uboundGroup.push_back(ub.getZExtValue());

    // Calculate offsets into the bounds arrays
    size_t loff = 0;
    for (size_t j = 0; j < idx; j++)
      loff += lboundGroup[j];

    size_t uoff = 0;
    for (size_t j = 0; j < idx; j++)
      uoff += uboundGroup[j];

    // Get the constant bounds if available
    auto lb = dyn_cast<AffineConstantExpr>(lbounds[loff]);
    auto ub = dyn_cast<AffineConstantExpr>(ubounds[uoff]);

    if (lb && ub) {
      // Create APInt values with 64 bit.
      return ConstantIntRanges::fromSigned(
          APInt(/*numBits=*/64, lb.getValue(), /*isSigned=*/true),
          APInt(/*numBits=*/64, ub.getValue() - 1, /*isSigned=*/true));
    }
    // Return sentinel values if bounds cannot be determined
    return ConstantIntRanges::maxRange(64);
  }
};

void AffineIntegerRangeAnalysis::visitNonControlFlowArguments(
    Operation *op, const RegionSuccessor &successor,
    ArrayRef<IntegerValueRangeLattice *> argLattices, unsigned firstIndex) {
  LLVM_DEBUG(llvm::dbgs() << "Inferring ranges for " << op->getName() << "\n");
  if (auto inferrable = dyn_cast<InferIntRangeInterface>(op)) {
    auto argRanges = llvm::map_to_vector(op->getOperands(), [&](Value value) {
      return getLatticeElementFor(getProgramPointAfter(op), value)->getValue();
    });

    auto joinCallback = [&](Value v, const IntegerValueRange &attrs) {
      auto arg = dyn_cast<BlockArgument>(v);
      if (!arg)
        return;
      if (!llvm::is_contained(successor.getSuccessor()->getArguments(), arg))
        return;

      LLVM_DEBUG(llvm::dbgs() << "Inferred range " << attrs << "\n");
      IntegerValueRangeLattice *lattice = argLattices[arg.getArgNumber()];
      IntegerValueRange oldRange = lattice->getValue();

      ChangeResult changed = lattice->join(attrs);

      // Catch loop results with loop variant bounds and conservatively make
      // them [-inf, inf] so we don't circle around infinitely often (because
      // the dataflow analysis in MLIR doesn't attempt to work out trip counts
      // and often can't).
      bool isYieldedValue = llvm::any_of(v.getUsers(), [](Operation *op) {
        return op->hasTrait<OpTrait::IsTerminator>();
      });
      if (isYieldedValue && !oldRange.isUninitialized() &&
          !(lattice->getValue() == oldRange)) {
        LLVM_DEBUG(llvm::dbgs() << "Loop variant loop result detected\n");
        changed |= lattice->join(IntegerValueRange::getMaxRange(v));
      }
      propagateIfChanged(lattice, changed);
    };

    inferrable.inferResultRangesFromOptional(argRanges, joinCallback);
    return;
  } // InferIntRangeInterface

  // Infer bounds for loop arguments that have static bounds.
  // TODO: (lorenzo) This should just work. But upstream AffineParallelOp does
  // not expose all the necessary interfaces/methods.
  if (auto loop = dyn_cast<affine::AffineParallelOp>(op)) {
    for (Value iv : loop.getIVs()) {
      ConstantIntRanges ivRange = getBoundsFromAffineParallel(
          loop, cast<BlockArgument>(iv).getArgNumber());
      IntegerValueRangeLattice *ivEntry = getLatticeElement(iv);
      propagateIfChanged(ivEntry, ivEntry->join(IntegerValueRange{ivRange}));
    }
    return;
  } // AffineParallelOp

  return SparseForwardDataFlowAnalysis::visitNonControlFlowArguments(
      op, successor, argLattices, firstIndex);
}

LogicalResult AffineIntegerRangeAnalysis::visitOperation(
    Operation *op, ArrayRef<const IntegerValueRangeLattice *> operands,
    ArrayRef<IntegerValueRangeLattice *> results) {

  auto inferrable = dyn_cast<InferIntRangeInterface>(op);
  if (!inferrable) {
    setAllToEntryStates(results);
    return success();
  }

  LLVM_DEBUG(llvm::dbgs() << "Inferring ranges for " << *op << "\n");
  auto argRanges = llvm::map_to_vector(
      operands, [](const IntegerValueRangeLattice *lattice) {
        return lattice->getValue();
      });

  auto joinCallback = [&](Value v, const IntegerValueRange &attrs) {
    auto result = dyn_cast<OpResult>(v);
    if (!result)
      return;
    assert(llvm::is_contained(op->getResults(), result));

    LLVM_DEBUG(llvm::dbgs() << "Inferred range " << attrs << "\n");
    IntegerValueRangeLattice *lattice = results[result.getResultNumber()];
    IntegerValueRange oldRange = lattice->getValue();

    ChangeResult changed = lattice->join(attrs);

    // Catch loop results with loop variant bounds and conservatively make
    // them [-inf, inf] so we don't circle around infinitely often (because
    // the dataflow analysis in MLIR doesn't attempt to work out trip counts
    // and often can't).
    bool isYieldedResult = llvm::any_of(v.getUsers(), [](Operation *op) {
      return op->hasTrait<OpTrait::IsTerminator>();
    });
    if (isYieldedResult && !oldRange.isUninitialized() &&
        !(lattice->getValue() == oldRange)) {
      LLVM_DEBUG(llvm::dbgs() << "Loop variant loop result detected\n");
      changed |= lattice->join(IntegerValueRange::getMaxRange(v));
    }
    propagateIfChanged(lattice, changed);
  };

  inferrable.inferResultRangesFromOptional(argRanges, joinCallback);
  return success();
}

std::optional<int64_t> maxSize(mlir::Value v) {
  if (auto ba = dyn_cast<BlockArgument>(v)) {
    if (auto par =
            dyn_cast<affine::AffineParallelOp>(ba.getOwner()->getParentOp())) {
      // Reductions are not supported yet.
      if (!par.getReductions().empty())
        return {};

      auto idx = ba.getArgNumber();
      SmallVector<int32_t> uboundGroup;
      for (auto ub : par.getUpperBoundsGroups())
        uboundGroup.push_back(ub.getZExtValue());

      if (uboundGroup[idx] != 1)
        return {};

      size_t uoff = 0;
      for (size_t i = 0; i < idx; i++)
        uoff += uboundGroup[i];

      SmallVector<AffineExpr> ubounds(
          par.getUpperBoundsMap().getResults().begin(),
          par.getUpperBoundsMap().getResults().end());

      auto ub = dyn_cast<AffineConstantExpr>(ubounds[uoff]);
      if (!ub)
        return {};
      return ub.getValue() - 1;
    }
  }
  if (auto shr = v.getDefiningOp<arith::ShRUIOp>()) {
    auto lhs = maxSize(shr.getLhs());
    if (!lhs)
      return {};

    IntegerAttr constValue;
    if (!matchPattern(shr.getRhs(), m_Constant(&constValue)))
      return lhs;

    return (*lhs) >> constValue.getValue().getZExtValue();
  }
  if (auto rem = v.getDefiningOp<arith::RemUIOp>()) {
    auto lhs = maxSize(rem.getLhs());

    APInt constValue;
    if (!matchPattern(rem.getRhs(), m_ConstantInt(&constValue)))
      return lhs;

    if (!lhs)
      return constValue.getZExtValue();

    return *lhs < constValue.getZExtValue() ? *lhs : constValue.getZExtValue();
  }
  if (auto rem = v.getDefiningOp<arith::MulIOp>()) {
    auto lhs = maxSize(rem.getLhs());
    if (!lhs)
      return lhs;

    APInt constValue;
    if (!matchPattern(rem.getRhs(), m_ConstantInt(&constValue)))
      return lhs;

    if (constValue.isNegative())
      return {};

    return (*lhs) * constValue.getZExtValue();
  }
  if (auto shr = v.getDefiningOp<arith::DivUIOp>()) {
    auto lhs = maxSize(shr.getLhs());
    if (!lhs)
      return {};

    IntegerAttr constValue;
    if (!matchPattern(shr.getRhs(), m_Constant(&constValue)))
      return lhs;

    if (constValue.getValue().isNonNegative())
      return (*lhs) >> constValue.getValue().getZExtValue();
  }
  if (auto shr = v.getDefiningOp<arith::AddIOp>()) {
    auto lhs = maxSize(shr.getLhs());
    if (!lhs)
      return {};

    IntegerAttr constValue;
    if (!matchPattern(shr.getRhs(), m_Constant(&constValue)))
      return {};

    if (constValue.getValue().isNonNegative())
      return (*lhs) + constValue.getValue().getZExtValue();
  }
  return {};
}

class ExtUIOfIndexUI final : public OpRewritePattern<arith::ExtUIOp> {
public:
  using OpRewritePattern<arith::ExtUIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ExtUIOp ext,
                                PatternRewriter &rewriter) const override {
    auto operand = ext.getOperand().getDefiningOp<arith::IndexCastUIOp>();
    if (!operand)
      return failure();
    auto maxSizeOpt = maxSize(operand.getOperand());
    if (!maxSizeOpt)
      return failure();
    if (APInt::getMaxValue(operand.getType().getIntOrFloatBitWidth())
            .ult(*maxSizeOpt))
      return failure();

    rewriter.replaceOpWithNewOp<arith::IndexCastUIOp>(ext, ext.getType(),
                                                      operand.getOperand());
    return success();
  }
};

class TruncIOfIndexUI final : public OpRewritePattern<arith::TruncIOp> {
public:
  using OpRewritePattern<arith::TruncIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::TruncIOp ext,
                                PatternRewriter &rewriter) const override {
    auto operand = ext.getOperand().getDefiningOp<arith::IndexCastUIOp>();
    if (!operand)
      return failure();
    auto maxSizeOpt = maxSize(operand.getOperand());
    if (!maxSizeOpt)
      return failure();
    if (APInt::getMaxValue(ext.getType().getIntOrFloatBitWidth())
            .ult(*maxSizeOpt))
      return failure();

    rewriter.replaceOpWithNewOp<arith::IndexCastUIOp>(ext, ext.getType(),
                                                      operand.getOperand());
    return success();
  }
};

class ShrUIOfIndexUI final : public OpRewritePattern<arith::ShRUIOp> {
public:
  using OpRewritePattern<arith::ShRUIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ShRUIOp ext,
                                PatternRewriter &rewriter) const override {
    auto operand = ext.getLhs().getDefiningOp<arith::IndexCastUIOp>();
    if (!operand)
      return failure();
    auto maxSizeOpt = maxSize(operand.getOperand());
    if (!maxSizeOpt)
      return failure();
    if (APInt::getMaxValue(operand.getType().getIntOrFloatBitWidth())
            .ult(*maxSizeOpt))
      return failure();

    IntegerAttr constValue;
    if (!matchPattern(ext.getRhs(), m_Constant(&constValue)))
      return failure();

    auto rhs = rewriter.create<arith::ConstantIndexOp>(
        ext.getRhs().getLoc(), constValue.getValue().getZExtValue());
    auto idxshr = rewriter.create<arith::ShRUIOp>(ext.getLoc(),
                                                  operand.getOperand(), rhs);
    rewriter.replaceOpWithNewOp<arith::IndexCastUIOp>(ext, ext.getType(),
                                                      idxshr);
    return success();
  }
};

class DivUIOfIndexUI final : public OpRewritePattern<arith::DivUIOp> {
public:
  using OpRewritePattern<arith::DivUIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::DivUIOp ext,
                                PatternRewriter &rewriter) const override {
    auto operand = ext.getLhs().getDefiningOp<arith::IndexCastUIOp>();
    if (!operand)
      return failure();
    auto maxSizeOpt = maxSize(operand.getOperand());
    if (!maxSizeOpt)
      return failure();
    if (APInt::getMaxValue(operand.getType().getIntOrFloatBitWidth())
            .ult(*maxSizeOpt))
      return failure();

    IntegerAttr constValue;
    if (!matchPattern(ext.getRhs(), m_Constant(&constValue)))
      return failure();

    if (constValue.getValue().isNegative())
      return failure();

    auto rhs = rewriter.create<arith::ConstantIndexOp>(
        ext.getRhs().getLoc(), constValue.getValue().getZExtValue());
    auto idxshr = rewriter.create<arith::DivUIOp>(ext.getLoc(),
                                                  operand.getOperand(), rhs);
    rewriter.replaceOpWithNewOp<arith::IndexCastUIOp>(ext, ext.getType(),
                                                      idxshr);
    return success();
  }
};

class DivMul final : public OpRewritePattern<arith::DivUIOp> {
public:
  using OpRewritePattern<arith::DivUIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::DivUIOp ext,
                                PatternRewriter &rewriter) const override {
    auto operand = ext.getLhs().getDefiningOp<arith::MulIOp>();
    if (!operand)
      return failure();
    auto maxSizeOpt = maxSize(operand);
    if (!maxSizeOpt)
      return failure();
    if (!operand.getType().isIndex())
      if (APInt::getMaxValue(operand.getType().getIntOrFloatBitWidth())
              .ult(*maxSizeOpt))
        return failure();
    if (operand.getRhs() != ext.getRhs())
      return failure();
    rewriter.replaceOp(ext, operand.getLhs());
    return success();
  }
};

class AddIOfIndexUI final : public OpRewritePattern<arith::AddIOp> {
public:
  using OpRewritePattern<arith::AddIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddIOp ext,
                                PatternRewriter &rewriter) const override {
    auto operand = ext.getLhs();
    auto operandOp = operand.getDefiningOp();
    if (!operandOp || !isa<arith::IndexCastUIOp, arith::IndexCastOp>(operandOp))
      return failure();
    auto maxSizeOpt = maxSize(operandOp->getOperand(0));
    if (!maxSizeOpt)
      return failure();
    if (APInt::getMaxValue(operand.getType().getIntOrFloatBitWidth())
            .ult(*maxSizeOpt))
      return failure();

    IntegerAttr constValue;
    if (!matchPattern(ext.getRhs(), m_Constant(&constValue)))
      return failure();

    if (constValue.getValue().isNegative()) {
      if (auto add2 = operandOp->getOperand(0).getDefiningOp<arith::AddIOp>()) {
        IntegerAttr constValue2;
        if (matchPattern(add2.getRhs(), m_Constant(&constValue2))) {
          auto v2 = constValue.getValue() + constValue2.getValue();
          if (!v2.isNegative()) {
            auto rhs = rewriter.create<arith::ConstantIndexOp>(
                ext.getRhs().getLoc(), v2.getZExtValue());
            auto idxshr = rewriter.create<arith::AddIOp>(ext.getLoc(),
                                                         add2.getLhs(), rhs);
            rewriter.replaceOpWithNewOp<arith::IndexCastUIOp>(
                ext, ext.getType(), idxshr);
            return success();
          }
        }
      }
      return failure();
    }

    auto rhs = rewriter.create<arith::ConstantIndexOp>(
        ext.getRhs().getLoc(), constValue.getValue().getZExtValue());
    auto idxshr = rewriter.create<arith::AddIOp>(ext.getLoc(),
                                                 operandOp->getOperand(0), rhs);
    rewriter.replaceOpWithNewOp<arith::IndexCastUIOp>(ext, ext.getType(),
                                                      idxshr);
    return success();
  }
};

class SubIOfIndexUI final : public OpRewritePattern<arith::SubIOp> {
public:
  using OpRewritePattern<arith::SubIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SubIOp ext,
                                PatternRewriter &rewriter) const override {
    auto operand = ext.getRhs().getDefiningOp<arith::IndexCastUIOp>();
    if (!operand)
      return failure();

    auto maxSizeOpt = maxSize(operand.getOperand());
    if (!maxSizeOpt)
      return failure();
    if (APInt::getMaxValue(operand.getType().getIntOrFloatBitWidth())
            .ult(*maxSizeOpt))
      return failure();

    APInt constValue;
    if (!matchPattern(ext.getLhs(), m_ConstantInt(&constValue)))
      return failure();

    if (!constValue.isZero())
      return failure();

    auto lhs2 = rewriter.create<arith::ConstantIndexOp>(
        ext.getLhs().getLoc(), constValue.getSExtValue());
    auto sub2 = rewriter.create<arith::SubIOp>(ext.getLoc(), lhs2,
                                               operand.getOperand());
    rewriter.replaceOpWithNewOp<arith::IndexCastOp>(ext, ext.getType(), sub2);
    return success();
  }
};

class MulIOfIndexUI final : public OpRewritePattern<arith::MulIOp> {
public:
  using OpRewritePattern<arith::MulIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::MulIOp ext,
                                PatternRewriter &rewriter) const override {
    auto operand = ext.getLhs().getDefiningOp();
    if (!operand || !isa<arith::IndexCastUIOp, arith::IndexCastOp>(operand))
      return failure();
    auto maxSizeOpt = maxSize(operand->getOperand(0));
    if (!maxSizeOpt)
      return failure();
    if (APInt::getMaxValue(
            operand->getResult(0).getType().getIntOrFloatBitWidth())
            .ult(*maxSizeOpt))
      return failure();

    IntegerAttr constValue;
    if (!matchPattern(ext.getRhs(), m_Constant(&constValue)))
      return failure();

    auto rhs = rewriter.create<arith::ConstantIndexOp>(
        ext.getRhs().getLoc(), constValue.getValue().isNegative()
                                   ? constValue.getValue().getSExtValue()
                                   : constValue.getValue().getZExtValue());
    auto idxshr = rewriter.create<arith::MulIOp>(ext.getLoc(),
                                                 operand->getOperand(0), rhs);
    if (constValue.getValue().isNegative())
      rewriter.replaceOpWithNewOp<arith::IndexCastOp>(ext, ext.getType(),
                                                      idxshr);
    else
      rewriter.replaceOpWithNewOp<arith::IndexCastUIOp>(ext, ext.getType(),
                                                        idxshr);
    return success();
  }
};

class ShLIOfIndexUI final : public OpRewritePattern<arith::ShLIOp> {
public:
  using OpRewritePattern<arith::ShLIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ShLIOp ext,
                                PatternRewriter &rewriter) const override {
    auto operand = ext.getLhs().getDefiningOp<arith::IndexCastUIOp>();
    if (!operand)
      return failure();
    auto maxSizeOpt = maxSize(operand.getOperand());
    if (!maxSizeOpt)
      return failure();
    if (APInt::getMaxValue(operand.getType().getIntOrFloatBitWidth())
            .ult(*maxSizeOpt))
      return failure();

    IntegerAttr constValue;
    if (!matchPattern(ext.getRhs(), m_Constant(&constValue)))
      return failure();

    auto rhs = rewriter.create<arith::ConstantIndexOp>(
        ext.getRhs().getLoc(), constValue.getValue().getZExtValue());
    auto idxshr =
        rewriter.create<arith::ShLIOp>(ext.getLoc(), operand.getOperand(), rhs);
    rewriter.replaceOpWithNewOp<arith::IndexCastUIOp>(ext, ext.getType(),
                                                      idxshr);
    return success();
  }
};

class AddIOfDoubleIndex final : public OpRewritePattern<arith::AddIOp> {
public:
  using OpRewritePattern<arith::AddIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddIOp ext,
                                PatternRewriter &rewriter) const override {
    if (!ext.getLhs().getDefiningOp<arith::IndexCastUIOp>() &&
        !ext.getLhs().getDefiningOp<arith::IndexCastOp>())
      return failure();

    if (!ext.getRhs().getDefiningOp<arith::IndexCastUIOp>() &&
        !ext.getRhs().getDefiningOp<arith::IndexCastOp>())
      return failure();

    if (!ext.getType().isInteger(64))
      return failure();

    bool sign = ext.getLhs().getDefiningOp<arith::IndexCastOp>() ||
                ext.getRhs().getDefiningOp<arith::IndexCastOp>();

    auto add = rewriter.create<arith::AddIOp>(
        ext.getLoc(), ext.getLhs().getDefiningOp()->getOperand(0),
        ext.getRhs().getDefiningOp()->getOperand(0));
    if (sign)
      rewriter.replaceOpWithNewOp<arith::IndexCastOp>(ext, ext.getType(), add);
    else
      rewriter.replaceOpWithNewOp<arith::IndexCastUIOp>(ext, ext.getType(),
                                                        add);
    return success();
  }
};

class ToRem final : public OpRewritePattern<arith::AddIOp> {
public:
  using OpRewritePattern<arith::AddIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddIOp ext,
                                PatternRewriter &rewriter) const override {
    for (int i = 0; i < 2; i++) {
      auto val = ext->getOperand(i);
      auto val2 = ext->getOperand(1 - i);
      auto mul = val2.getDefiningOp<arith::MulIOp>();
      if (!mul)
        continue;
      APInt factor;
      if (!matchPattern(mul.getRhs(), m_ConstantInt(&factor)))
        continue;
      auto div = mul.getLhs().getDefiningOp<arith::DivUIOp>();
      if (!div)
        continue;
      APInt divisor;
      if (!matchPattern(div.getRhs(), m_ConstantInt(&divisor)))
        continue;
      if (factor != -divisor)
        continue;
      if (div.getLhs() != val)
        continue;
      rewriter.replaceOpWithNewOp<arith::RemUIOp>(
          ext, val, factor.isNegative() ? div.getRhs() : mul.getRhs());
      return success();
    }
    return failure();
  }
};

class SwitchToIf final : public OpRewritePattern<scf::IndexSwitchOp> {
public:
  using OpRewritePattern<scf::IndexSwitchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::IndexSwitchOp switchOp,
                                PatternRewriter &rewriter) const override {
    if (switchOp.getNumCases() != 1)
      return failure();

    ArrayRef<int64_t> cases = switchOp.getCases();
    Value caseValue = rewriter.create<arith::ConstantIndexOp>(switchOp.getLoc(),
                                                              cases.front());

    Value cmpResult = rewriter.create<arith::CmpIOp>(
        switchOp.getLoc(), arith::CmpIPredicate::eq, switchOp.getArg(),
        caseValue);

    scf::IfOp ifOp =
        rewriter.create<scf::IfOp>(switchOp.getLoc(), switchOp.getResultTypes(),
                                   cmpResult, /*withElseRegion=*/true);

    // Move the first case block into the then region
    Block &firstBlock = switchOp.getCaseBlock(0);
    rewriter.mergeBlocks(&firstBlock, ifOp.thenBlock(),
                         firstBlock.getArguments());

    // Move the second case block into the else region
    Block &secondBlock = switchOp.getDefaultBlock();
    rewriter.mergeBlocks(&secondBlock, ifOp.elseBlock(),
                         secondBlock.getArguments());

    // Replace the switch with the if
    rewriter.replaceOp(switchOp, ifOp.getResults());
    return success();
  }
};

class SimplifyIfByRemovingEmptyThen final : public OpRewritePattern<scf::IfOp> {
public:
  using OpRewritePattern<scf::IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    if (ifOp.getNumResults() != 0)
      return failure();

    // Check if the then block is empty (contains only the terminator)
    Block &thenBlock = ifOp.getThenRegion().front();
    if (!llvm::hasSingleElement(thenBlock))
      return failure();

    // Check if there is an else region
    bool hasElse = !ifOp.getElseRegion().empty();
    if (!hasElse)
      return failure();

    // Get the condition and negate it
    Value cond = ifOp.getCondition();
    Value negatedCond = rewriter.create<arith::XOrIOp>(
        ifOp.getLoc(), cond,
        rewriter.create<arith::ConstantIntOp>(ifOp.getLoc(), 1,
                                              rewriter.getI1Type()));

    // Create new if operation with negated condition and no else region
    auto newIf = rewriter.create<scf::IfOp>(ifOp.getLoc(),
                                            ifOp.getResultTypes(), negatedCond,
                                            /*withElseRegion=*/false);

    // Move operations from old else block to new then block
    Block &elseBlock = ifOp.getElseRegion().front();
    Block &newThenBlock = newIf.getThenRegion().front();

    // Move all operations except the terminator before the new block's
    // terminator
    auto &oldOps = elseBlock.getOperations();
    auto &newOps = newThenBlock.getOperations();
    auto terminator = newThenBlock.getTerminator();
    newOps.splice(terminator->getIterator(), oldOps, oldOps.begin(),
                  std::prev(oldOps.end()));

    // Replace old if with new if
    rewriter.replaceOp(ifOp, newIf.getResults());
    return success();
  }
};

class IfToSelect final : public OpRewritePattern<scf::IfOp> {
public:
  using OpRewritePattern<scf::IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    // Check if if has both then and else regions
    bool hasElse = !ifOp.getElseRegion().empty();
    if (!hasElse)
      return failure();

    Location loc = ifOp.getLoc();
    Value condition = ifOp.getCondition();

    // Get the yield ops from both branches
    Block *thenBlock = ifOp.thenBlock();
    Block *elseBlock = ifOp.elseBlock();
    auto thenYield = cast<scf::YieldOp>(thenBlock->getTerminator());
    auto elseYield = cast<scf::YieldOp>(elseBlock->getTerminator());

    // Check if all operations in both blocks are pure
    if (llvm::any_of(thenBlock->getOperations(),
                     [](Operation &op) { return !isPure(&op); }))
      return failure();
    if (llvm::any_of(elseBlock->getOperations(),
                     [](Operation &op) { return !isPure(&op); }))
      return failure();

    // Clone all operations from both branches before their yields
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(ifOp);

    // Clone then block operations
    IRMapping thenMapping;
    for (auto &op : ifOp.thenBlock()->without_terminator()) {
      Operation *clonedOp = rewriter.clone(op, thenMapping);
      for (auto [orig, clone] :
           llvm::zip(op.getResults(), clonedOp->getResults())) {
        thenMapping.map(orig, clone);
      }
    }

    // Clone else block operations
    IRMapping elseMapping;
    for (auto &op : ifOp.elseBlock()->without_terminator()) {
      Operation *clonedOp = rewriter.clone(op, elseMapping);
      for (auto [orig, clone] :
           llvm::zip(op.getResults(), clonedOp->getResults())) {
        elseMapping.map(orig, clone);
      }
    }

    // Create selects for each result
    SmallVector<Value, 4> results;
    for (auto [thenVal, elseVal] :
         llvm::zip(thenYield.getOperands(), elseYield.getOperands())) {
      // Map the yield operands through our value mappings
      Value mappedThenVal = thenMapping.lookupOrDefault(thenVal);
      Value mappedElseVal = elseMapping.lookupOrDefault(elseVal);

      // Create select op with same attributes as original if op
      auto select = rewriter.create<arith::SelectOp>(
          loc, condition, mappedThenVal, mappedElseVal);
      results.push_back(select);
    }

    rewriter.replaceOp(ifOp, results);
    return success();
  }
};

} // end namespace

struct CanonicalizeLoopsPass
    : public enzyme::impl::CanonicalizeLoopsPassBase<CanonicalizeLoopsPass> {
  void runOnOperation() override {

    // Step 0: Canonicalize loops when possible.
    {
      RewritePatternSet patterns(&getContext());
      patterns.add<RemoveAffineParallelSingleIter, SwitchToIf,
                   SimplifyIfByRemovingEmptyThen, IfToSelect>(&getContext());

      if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }

    // Step 1: Run data flow analysis and do additional simplifications.
    DataFlowSolver solver;
    solver.load<DeadCodeAnalysis>();
    solver.load<AffineIntegerRangeAnalysis>();

    if (failed(solver.initializeAndRun(getOperation()))) {
      signalPassFailure();
      return;
    }

    OpBuilder b(getOperation()->getContext());

    // Rule from CMPI
    getOperation()->walk([&](Operation *op) {
      if (auto cmpiOp = dyn_cast<arith::CmpIOp>(op)) {
        auto lhs = cmpiOp.getLhs();
        auto *lattice = solver.lookupState<IntegerValueRangeLattice>(lhs);
        if (!lattice || lattice->getValue().isUninitialized())
          return;
        auto cst = cmpiOp.getRhs().getDefiningOp<arith::ConstantOp>();
        if (!cst)
          return;
        IntegerValueRange range = lattice->getValue();
        if (range.isUninitialized())
          return;
        ConstantIntRanges cstRange = range.getValue();
        // Let's evaluate the range of the lhs and try to figure out if the
        // condition is true or false.
        auto pred = cmpiOp.getPredicate();
        APInt cstRhs = cast<IntegerAttr>(cst.getValue()).getValue();
        if (pred == arith::CmpIPredicate::ne) {
          std::optional<APInt> constantRangeValue =
              range.getValue().getConstantValue();
          if (!constantRangeValue.has_value())
            return;
          b.setInsertionPoint(cmpiOp);
          auto cst = b.create<arith::ConstantOp>(
              cmpiOp.getLoc(), b.getI1Type(),
              IntegerAttr::get(b.getI1Type(), !constantRangeValue->eq(cstRhs)));
          cmpiOp.getResult().replaceAllUsesWith(cst);
        }
        if (pred == arith::CmpIPredicate::eq) {
          std::optional<APInt> constantRangeValue =
              range.getValue().getConstantValue();
          if (!constantRangeValue.has_value())
            return;
          b.setInsertionPoint(cmpiOp);
          auto cst = b.create<arith::ConstantOp>(
              cmpiOp.getLoc(), b.getI1Type(),
              IntegerAttr::get(b.getI1Type(), constantRangeValue->eq(cstRhs)));
          cmpiOp.getResult().replaceAllUsesWith(cst);
        }
        if (pred == arith::CmpIPredicate::ult) {
          const APInt umax = cstRange.umax();
          const APInt umin = cstRange.umin();
          if (umax.ult(cstRhs)) {
            // Condition always true.
            b.setInsertionPoint(cmpiOp);
            auto cst = b.create<arith::ConstantOp>(
                cmpiOp.getLoc(), b.getI1Type(),
                IntegerAttr::get(b.getI1Type(), true));
            cmpiOp.getResult().replaceAllUsesWith(cst);
          }
          // range < cst -> !(range >= cst)
          if (umin.uge(cstRhs)) {
            // Condition always false.
            b.setInsertionPoint(cmpiOp);
            auto cst = b.create<arith::ConstantOp>(
                cmpiOp.getLoc(), b.getI1Type(),
                IntegerAttr::get(b.getI1Type(), false));
            cmpiOp.getResult().replaceAllUsesWith(cst);
          }
        }
        if (pred == arith::CmpIPredicate::ule) {
          const APInt umax = cstRange.umax();
          const APInt umin = cstRange.umin();
          if (umax.ule(cstRhs)) {
            // Condition always true.
            b.setInsertionPoint(cmpiOp);
            auto cst = b.create<arith::ConstantOp>(
                cmpiOp.getLoc(), b.getI1Type(),
                IntegerAttr::get(b.getI1Type(), true));
            cmpiOp.getResult().replaceAllUsesWith(cst);
          }
          // range <= cst -> !(range > cst)
          if (umin.ugt(cstRhs)) {
            // Condition always false.
            b.setInsertionPoint(cmpiOp);
            auto cst = b.create<arith::ConstantOp>(
                cmpiOp.getLoc(), b.getI1Type(),
                IntegerAttr::get(b.getI1Type(), false));
            cmpiOp.getResult().replaceAllUsesWith(cst);
          }
        }
        if (pred == arith::CmpIPredicate::ugt) {
          const APInt umax = cstRange.umax();
          const APInt umin = cstRange.umin();
          if (umin.ugt(cstRhs)) {
            // Condition always true.
            b.setInsertionPoint(cmpiOp);
            auto cst = b.create<arith::ConstantOp>(
                cmpiOp.getLoc(), b.getI1Type(),
                IntegerAttr::get(b.getI1Type(), true));
            cmpiOp.getResult().replaceAllUsesWith(cst);
          }
          // range > cst -> !(range <= cst)
          if (umax.ule(cstRhs)) {
            // Condition always false.
            b.setInsertionPoint(cmpiOp);
            auto cst = b.create<arith::ConstantOp>(
                cmpiOp.getLoc(), b.getI1Type(),
                IntegerAttr::get(b.getI1Type(), false));
            cmpiOp.getResult().replaceAllUsesWith(cst);
          }
        }
        if (pred == arith::CmpIPredicate::uge) {
          const APInt umax = cstRange.umax();
          const APInt umin = cstRange.umin();
          if (umin.uge(cstRhs)) {
            // Condition always true.
            b.setInsertionPoint(cmpiOp);
            auto cst = b.create<arith::ConstantOp>(
                cmpiOp.getLoc(), b.getI1Type(),
                IntegerAttr::get(b.getI1Type(), true));
            cmpiOp.getResult().replaceAllUsesWith(cst);
          }
          // range >= cst -> !(range < cst)
          if (umax.ult(cstRhs)) {
            // Condition always false.
            b.setInsertionPoint(cmpiOp);
            auto cst = b.create<arith::ConstantOp>(
                cmpiOp.getLoc(), b.getI1Type(),
                IntegerAttr::get(b.getI1Type(), false));
            cmpiOp.getResult().replaceAllUsesWith(cst);
          }
        }

        if (pred == arith::CmpIPredicate::slt) {
          const APInt smax = cstRange.smax();
          const APInt smin = cstRange.smin();
          if (smax.slt(cstRhs)) {
            // Condition always true.
            b.setInsertionPoint(cmpiOp);
            auto cst = b.create<arith::ConstantOp>(
                cmpiOp.getLoc(), b.getI1Type(),
                IntegerAttr::get(b.getI1Type(), true));
            cmpiOp.getResult().replaceAllUsesWith(cst);
          }
          // range < cst -> !(range >= cst)
          if (smin.sge(cstRhs)) {
            // Condition always false.
            b.setInsertionPoint(cmpiOp);
            auto cst = b.create<arith::ConstantOp>(
                cmpiOp.getLoc(), b.getI1Type(),
                IntegerAttr::get(b.getI1Type(), false));
            cmpiOp.getResult().replaceAllUsesWith(cst);
          }
        }
        if (pred == arith::CmpIPredicate::sle) {
          const APInt smax = cstRange.smax();
          const APInt smin = cstRange.smin();
          if (smax.sle(cstRhs)) {
            // Condition always true.
            b.setInsertionPoint(cmpiOp);
            auto cst = b.create<arith::ConstantOp>(
                cmpiOp.getLoc(), b.getI1Type(),
                IntegerAttr::get(b.getI1Type(), true));
            cmpiOp.getResult().replaceAllUsesWith(cst);
          }
          // range <= cst -> !(range > cst)
          if (smin.sgt(cstRhs)) {
            // Condition always false.
            b.setInsertionPoint(cmpiOp);
            auto cst = b.create<arith::ConstantOp>(
                cmpiOp.getLoc(), b.getI1Type(),
                IntegerAttr::get(b.getI1Type(), false));
            cmpiOp.getResult().replaceAllUsesWith(cst);
          }
        }
        if (pred == arith::CmpIPredicate::sgt) {
          const APInt smax = cstRange.smax();
          const APInt smin = cstRange.smin();
          if (smin.sgt(cstRhs)) {
            // Condition always true.
            b.setInsertionPoint(cmpiOp);
            auto cst = b.create<arith::ConstantOp>(
                cmpiOp.getLoc(), b.getI1Type(),
                IntegerAttr::get(b.getI1Type(), true));
            cmpiOp.getResult().replaceAllUsesWith(cst);
          }
          // range > cst -> !(range <= cst)
          if (smax.sle(cstRhs)) {
            // Condition always false.
            b.setInsertionPoint(cmpiOp);
            auto cst = b.create<arith::ConstantOp>(
                cmpiOp.getLoc(), b.getI1Type(),
                IntegerAttr::get(b.getI1Type(), false));
            cmpiOp.getResult().replaceAllUsesWith(cst);
          }
        }
        if (pred == arith::CmpIPredicate::sge) {
          const APInt smax = cstRange.smax();
          const APInt smin = cstRange.smin();
          if (smin.sge(cstRhs)) {
            // Condition always true.
            b.setInsertionPoint(cmpiOp);
            auto cst = b.create<arith::ConstantOp>(
                cmpiOp.getLoc(), b.getI1Type(),
                IntegerAttr::get(b.getI1Type(), true));
            cmpiOp.getResult().replaceAllUsesWith(cst);
          }
          // range >= cst -> !(range < cst)
          if (smax.slt(cstRhs)) {
            // Condition always false.
            b.setInsertionPoint(cmpiOp);
            auto cst = b.create<arith::ConstantOp>(
                cmpiOp.getLoc(), b.getI1Type(),
                IntegerAttr::get(b.getI1Type(), false));
            cmpiOp.getResult().replaceAllUsesWith(cst);
          }
        }
      }
      if (auto inferOp = dyn_cast<InferIntRangeInterface>(op)) {
        if (inferOp->getNumResults() != 1)
          return;
        auto *lattice =
            solver.lookupState<IntegerValueRangeLattice>(inferOp->getResult(0));
        if (!lattice || lattice->getValue().isUninitialized())
          return;
        IntegerValueRange range = lattice->getValue();
        if (range.isUninitialized())
          return;
        std::optional<APInt> maybeRange = range.getValue().getConstantValue();
        if (maybeRange.has_value()) {
          b.setInsertionPoint(inferOp);
          auto cst = b.create<arith::ConstantOp>(
              inferOp.getLoc(), inferOp->getResult(0).getType(),
              IntegerAttr::get(inferOp->getResult(0).getType(),
                               maybeRange.value()));
          inferOp->getResult(0).replaceAllUsesWith(cst);
        }
      }
    });

    {
      RewritePatternSet patterns(&getContext());
      addSingleIter(patterns, &getContext());
      if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }
  }
};
} // namespace

void mlir::enzyme::addSingleIter(RewritePatternSet &patterns,
                                 MLIRContext *ctx) {
  patterns
      .add<RemoveAffineParallelSingleIter, ExtUIOfIndexUI, TruncIOfIndexUI,
           ShrUIOfIndexUI, DivUIOfIndexUI, DivMul, AddIOfIndexUI, SubIOfIndexUI,
           MulIOfIndexUI, ShLIOfIndexUI, AddIOfDoubleIndex, ToRem>(ctx);
}
