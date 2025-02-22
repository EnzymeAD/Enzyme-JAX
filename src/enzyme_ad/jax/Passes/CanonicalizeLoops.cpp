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
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
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

      auto lb = lbounds[loff].dyn_cast<AffineConstantExpr>();
      if (!lb)
        continue;
      auto ub = ubounds[uoff].dyn_cast<AffineConstantExpr>();
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
  std::pair<APInt, APInt>
  getBoundsFromAffineParallel(affine::AffineParallelOp loop, size_t idx) {
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
    auto lb = lbounds[loff].dyn_cast<AffineConstantExpr>();
    auto ub = ubounds[uoff].dyn_cast<AffineConstantExpr>();

    if (lb && ub) {
      // Create APInt values with 64 bit.
      return {APInt(/*numBits=*/64, lb.getValue(), /*isSigned=*/true),
              APInt(/*numBits=*/64, ub.getValue() - 1, /*isSigned=*/true)};
    }
    // Return sentinel values if bounds cannot be determined
    return {APInt::getSignedMinValue(64), APInt::getSignedMaxValue(64)};
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
      auto [min, max] = getBoundsFromAffineParallel(loop, 0);
      IntegerValueRangeLattice *ivEntry = getLatticeElement(iv);
      auto ivRange = ConstantIntRanges::fromSigned(min, max);
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

      auto ub = ubounds[uoff].dyn_cast<AffineConstantExpr>();
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

class AddIOfIndexUI final : public OpRewritePattern<arith::AddIOp> {
public:
  using OpRewritePattern<arith::AddIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddIOp ext,
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
    auto idxshr =
        rewriter.create<arith::AddIOp>(ext.getLoc(), operand.getOperand(), rhs);
    rewriter.replaceOpWithNewOp<arith::IndexCastUIOp>(ext, ext.getType(),
                                                      idxshr);
    return success();
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

} // end namespace

struct CanonicalizeLoopsPass
    : public enzyme::impl::CanonicalizeLoopsPassBase<CanonicalizeLoopsPass> {
  void runOnOperation() override {

    // Step 0: Canonicalize loops when possible.
    {
      RewritePatternSet patterns(&getContext());
      patterns.add<RemoveAffineParallelSingleIter, SwitchToIf>(&getContext());

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
        APInt cstRhs = cst.getValue().cast<IntegerAttr>().getValue();
        if (pred == arith::CmpIPredicate::ne) {
          std::optional<APInt> constantRangeValue =
              range.getValue().getConstantValue();
          if (!constantRangeValue.has_value())
            return;
          if (constantRangeValue->eq(cstRhs)) {
            b.setInsertionPoint(cmpiOp);
            auto cst = b.create<arith::ConstantOp>(
                cmpiOp.getLoc(), b.getI1Type(),
                IntegerAttr::get(b.getI1Type(), false));
            cmpiOp.getResult().replaceAllUsesWith(cst);
          }
        }
        if (pred == arith::CmpIPredicate::ult) {
          const APInt umax = cstRange.umax();
          const APInt umin = cstRange.umin();
          if (umax.ult(cstRhs) && umin.ult(cstRhs)) {
            // Condition always true.
            b.setInsertionPoint(cmpiOp);
            auto cst = b.create<arith::ConstantOp>(
                cmpiOp.getLoc(), b.getI1Type(),
                IntegerAttr::get(b.getI1Type(), true));
            cmpiOp.getResult().replaceAllUsesWith(cst);
          }
          if (!umax.ult(cstRhs) && !umin.ult(cstRhs)) {
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
      patterns
          .add<ExtUIOfIndexUI, ShrUIOfIndexUI, DivUIOfIndexUI, AddIOfIndexUI>(
              &getContext());

      if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }
  }
};
} // namespace
