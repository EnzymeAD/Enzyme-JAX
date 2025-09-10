//===- PrintPass.cpp - Print the MLIR module                     ------------ //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to print the MLIR module
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"

#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Passes/SelectPatterns.h"

#include "src/enzyme_ad/jax/Utils.h"
#include "llvm/ADT/MapVector.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_SCFCANONICALIZEFOR
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir;
using namespace mlir::scf;
using namespace mlir::func;
using namespace mlir::arith;

namespace {
struct CanonicalizeFor
    : public enzyme::impl::SCFCanonicalizeForBase<CanonicalizeFor> {
  void runOnOperation() override;
};
} // namespace

// %f = scf.for %c = true, %a = ...
//    %r = if %c {
//      %d = cond()
//      %q = scf.if not %d {
//        yield %a
//      } else {
//        %a2 = addi %a, ...
//        yield %a2
//      }
//      yield %d, %q
//    } else {
//      yield %false, %a
//    }
//    yield %r#0, %r#1
// }
// no use of %f#1
//
// becomes
//
// %f = scf.for %c = true, %a = ...
//    %r = if %c {
//      %d = cond()
//      %a2 = addi %a, ...
//      yield %d, %a2
//    } else {
//      yield %false, %a
//    }
//    yield %r#0, %r#1
// }
// no use of %f#1
//
// and finally
//
// %f = scf.for %c = true, %a = ...
//    %a2 = addi %a, ...
//    %r = if %c {
//      %d = cond()
//      yield %d, %a2
//    } else {
//      yield %false, %a
//    }
//    yield %r#0, %a2
// }
// no use of %f#1
struct ForBreakAddUpgrade : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const final {
    if (!forOp.getInits().size())
      return failure();

    Block &block = forOp.getRegion().front();
    // First check there is an outermost if
    auto outerIfOp = dyn_cast<scf::IfOp>(*block.begin());
    if (!outerIfOp)
      return failure();

    auto condition = outerIfOp.getCondition();
    //  and that the outermost if's condition is an iter arg of the for
    auto condArg = dyn_cast<BlockArgument>(condition);
    if (!condArg)
      return failure();
    if (condArg.getOwner()->getParentOp() != forOp)
      return failure();
    // which starts as true
    if (!matchPattern(forOp.getInits()[condArg.getArgNumber() - 1], m_One()))
      return failure();
    // and is false unless coming from inside the if
    auto forYieldOp = cast<scf::YieldOp>(block.getTerminator());
    auto opres =
        dyn_cast<OpResult>(forYieldOp.getOperand(condArg.getArgNumber() - 1));
    if (!opres)
      return failure();
    if (opres.getOwner() != outerIfOp)
      return failure();

    if (!matchPattern(outerIfOp.elseYield().getOperand(opres.getResultNumber()),
                      m_Zero()))
      return failure();

    bool changed = false;
    for (auto it :
         llvm::zip(forOp.getRegionIterArgs(), forYieldOp.getOperands(),
                   forOp.getResults(), forOp.getInits())) {
      auto regionArg = std::get<0>(it);
      Value forYieldOperand = std::get<1>(it);
      Value res = std::get<2>(it);
      Value iterOp = std::get<3>(it);

      if (opres.getResultNumber() == regionArg.getArgNumber() - 1)
        continue;

      auto opres2 = dyn_cast<OpResult>(forYieldOperand);
      if (!opres2)
        continue;
      if (opres2.getOwner() != outerIfOp)
        continue;
      auto topOp = outerIfOp.thenYield().getOperand(opres2.getResultNumber());

      auto trueYield =
          outerIfOp.thenYield().getOperand(opres.getResultNumber());
      bool negated = false;
      while (auto neg = trueYield.getDefiningOp<XOrIOp>())
        if (matchPattern(neg.getOperand(1), m_One())) {
          trueYield = neg.getOperand(0);
          negated = !negated;
        }

      if (auto innerIfOp = topOp.getDefiningOp<scf::IfOp>()) {
        Value ifcond = innerIfOp.getCondition();
        while (auto neg = ifcond.getDefiningOp<XOrIOp>())
          if (matchPattern(neg.getOperand(1), m_One())) {
            ifcond = neg.getOperand(0);
            negated = !negated;
          }

        if (ifcond == trueYield) {
          // If never used, can always pick the "continue" value
          if (res.use_empty()) {
            auto idx = cast<OpResult>(topOp).getResultNumber();
            Value val =
                (negated ? innerIfOp.elseYield() : innerIfOp.thenYield())
                    .getOperand(idx);
            Region *reg = (negated ? &innerIfOp.getElseRegion()
                                   : &innerIfOp.getThenRegion());
            if (reg->isAncestor(val.getParentRegion())) {
              if (auto addi = val.getDefiningOp<arith::AddIOp>()) {
                if (reg->isAncestor(addi.getOperand(0).getParentRegion()) ||
                    reg->isAncestor(addi.getOperand(1).getParentRegion()))
                  continue;
                rewriter.setInsertionPoint(innerIfOp);
                val = rewriter.replaceOpWithNewOp<arith::AddIOp>(
                    addi, addi.getOperand(0), addi.getOperand(1));
              } else
                continue;
            }

            rewriter.setInsertionPoint(innerIfOp);
            auto cloned = rewriter.clone(*innerIfOp);
            SmallVector<Value> results(cloned->getResults());
            results[idx] = val;
            rewriter.replaceOp(innerIfOp, results);
            changed = true;
            continue;
          }
        }
      }

      if (auto innerSelOp = topOp.getDefiningOp<arith::SelectOp>()) {
        bool negated = false;
        Value ifcond = innerSelOp.getCondition();
        while (auto neg = ifcond.getDefiningOp<XOrIOp>())
          if (matchPattern(neg.getOperand(1), m_One())) {
            ifcond = neg.getOperand(0);
            negated = !negated;
          }
        if (ifcond == trueYield) {
          // If never used, can always pick the "continue" value
          if (res.use_empty()) {
            Value val = (negated ? innerSelOp.getFalseValue()
                                 : innerSelOp.getTrueValue());
            Value results[] = {val};
            rewriter.replaceOp(innerSelOp, results);
            changed = true;
            continue;
          }
          bool seenSelf = false;
          bool seenIllegal = false;
          for (auto &u : regionArg.getUses()) {
            if (u.getOwner() == innerSelOp &&
                regionArg != innerSelOp.getCondition()) {
              seenSelf = true;
              continue;
            }
            if (u.getOwner() != outerIfOp.elseYield()) {
              seenIllegal = true;
              break;
            }
            if (u.getOperandNumber() != opres2.getResultNumber()) {
              seenIllegal = true;
              break;
            }
          }
          // if this is only used by itself and yielding out of the for, remove
          // the induction of the region arg and just simply set it to be the
          // incoming iteration arg.
          if (seenSelf && !seenIllegal) {
            rewriter.setInsertionPoint(innerSelOp);
            rewriter.replaceOpWithNewOp<arith::SelectOp>(
                innerSelOp, innerSelOp.getCondition(),
                negated ? iterOp : innerSelOp.getTrueValue(),
                negated ? innerSelOp.getFalseValue() : iterOp);
            changed = true;
            continue;
          }
        }
      }

      // Only do final hoisting on a variable which can be loop induction
      // replaced.
      //  Otherwise additional work is added outside the break
      if (res.use_empty()) {
        if (auto add = topOp.getDefiningOp<arith::AddIOp>()) {
          if (forOp.getRegion().isAncestor(add.getOperand(1).getParentRegion()))
            continue;

          if (add.getOperand(0) != regionArg)
            continue;
          rewriter.setInsertionPoint(outerIfOp);
          SmallVector<Value> results(forYieldOp->getOperands());
          results[regionArg.getArgNumber() - 1] =
              rewriter.replaceOpWithNewOp<arith::AddIOp>(add, add.getOperand(0),
                                                         add.getOperand(1));
          rewriter.setInsertionPoint(forYieldOp);
          rewriter.replaceOpWithNewOp<scf::YieldOp>(forYieldOp, results);
          return success();
        }
      }
    }
    return success(changed);
  }
};

// Helper function to cast value to target type
static Value castValue(PatternRewriter &rewriter, Value value, Value target,
                       Location loc) {
  Type valueType = value.getType();
  Type targetType = target.getType();
  if (valueType == targetType)
    return value;

  if (targetType.isIndex() || value.getType().isIndex()) {
    return rewriter.create<IndexCastOp>(loc, targetType, value);
  } else if (targetType.getIntOrFloatBitWidth() >
             valueType.getIntOrFloatBitWidth()) {
    return rewriter.create<arith::ExtSIOp>(loc, targetType, value);
  } else {
    return rewriter.create<arith::TruncIOp>(loc, targetType, value);
  }
};

struct ForOpInductionReplacement : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const final {
    // Defer until after the step is a constant, if possible
    if (auto icast = forOp.getStep().getDefiningOp<IndexCastOp>())
      if (matchPattern(icast.getIn(), m_Constant()))
        return failure();
    bool canonicalize = false;
    Block &block = forOp.getRegion().front();
    auto yieldOp = cast<scf::YieldOp>(block.getTerminator());

    for (auto [outiter, iterarg, res, yld] :
         llvm::zip(forOp.getInits(),          // iter from outside
                   forOp.getRegionIterArgs(), // iter inside region
                   forOp.getResults(),        // op results
                   yieldOp.getOperands()      // iter yield
                   )) {

      AddIOp addOp = yld.getDefiningOp<AddIOp>();
      if (!addOp)
        continue;

      if (addOp.getOperand(0) != iterarg)
        continue;

      if (!addOp.getOperand(1).getParentRegion()->isAncestor(
              forOp->getParentRegion()))
        continue;

      bool sameValue = addOp.getOperand(1) == forOp.getStep();

      APInt rattr;
      APInt sattr;
      if (matchPattern(addOp.getOperand(1), m_ConstantInt(&rattr)))
        if (matchPattern(forOp.getStep(), m_ConstantInt(&sattr))) {
          size_t maxWidth = (rattr.getBitWidth() > sattr.getBitWidth())
                                ? rattr.getBitWidth()
                                : sattr.getBitWidth();
          sameValue |= rattr.zext(maxWidth) == sattr.zext(maxWidth);
        }

      if (!iterarg.use_empty()) {
        Value init = outiter;
        rewriter.setInsertionPointToStart(&forOp.getRegion().front());
        Value replacement = rewriter.create<SubIOp>(
            forOp.getLoc(), forOp.getInductionVar(), forOp.getLowerBound());

        if (!sameValue)
          replacement = rewriter.create<DivUIOp>(forOp.getLoc(), replacement,
                                                 forOp.getStep());

        if (!sameValue) {
          Value step = addOp.getOperand(1);

          if (step.getType() != replacement.getType()) {
            if (replacement.getType().isIndex() || step.getType().isIndex())
              step = rewriter.create<IndexCastOp>(forOp.getLoc(),
                                                  replacement.getType(), step);
            else if (replacement.getType().getIntOrFloatBitWidth() >
                     step.getType().getIntOrFloatBitWidth())
              step = rewriter.create<arith::ExtSIOp>(
                  forOp.getLoc(), replacement.getType(), step);
            else
              step = rewriter.create<arith::TruncIOp>(
                  forOp.getLoc(), replacement.getType(), step);
          }

          replacement =
              rewriter.create<MulIOp>(forOp.getLoc(), replacement, step);
        }

        init = castValue(rewriter, init, replacement, forOp.getLoc());

        replacement =
            rewriter.create<AddIOp>(forOp.getLoc(), init, replacement);

        replacement = castValue(rewriter, replacement, iterarg, forOp.getLoc());

        rewriter.modifyOpInPlace(
            forOp, [&] { iterarg.replaceAllUsesWith(replacement); });
        canonicalize = true;
      }

      if (!res.use_empty()) {
        Value init = outiter;
        rewriter.setInsertionPoint(forOp);
        Value replacement = rewriter.create<SubIOp>(
            forOp.getLoc(), forOp.getUpperBound(), forOp.getLowerBound());

        if (!sameValue)
          replacement = rewriter.create<DivUIOp>(forOp.getLoc(), replacement,
                                                 forOp.getStep());

        if (!sameValue) {
          Value step = addOp.getOperand(1);

          step = castValue(rewriter, step, replacement, forOp.getLoc());

          replacement =
              rewriter.create<MulIOp>(forOp.getLoc(), replacement, step);
        }

        init = castValue(rewriter, init, replacement, forOp.getLoc());

        replacement =
            rewriter.create<AddIOp>(forOp.getLoc(), init, replacement);

        replacement = castValue(rewriter, replacement, iterarg, forOp.getLoc());

        rewriter.modifyOpInPlace(forOp,
                                 [&] { res.replaceAllUsesWith(replacement); });
        canonicalize = true;
      }
    }

    return success(canonicalize);
  }
};

struct RemoveInductionVarRelated : public OpRewritePattern<ForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: generalize this into some affine expr of inductionVar instead of
    // just inductionVar + step
    // Check if any yielded value is inductionVar + step
    auto yieldOp = cast<scf::YieldOp>(op.getBody()->getTerminator());
    for (auto [idx, yieldVal] : llvm::enumerate(yieldOp.getOperands())) {
      auto addOp = yieldVal.getDefiningOp<arith::AddIOp>();
      if (!addOp)
        continue;

      Value inductionVar = op.getInductionVar();
      Value step = op.getStep();

      // Check if this is inductionVar + step.
      if ((addOp.getLhs() == inductionVar && addOp.getRhs() == step) ||
          (addOp.getRhs() == inductionVar && addOp.getLhs() == step)) {
        return rewriteLoop(op, idx, rewriter);
      }
    }
    return failure();
  }

  LogicalResult rewriteLoop(scf::ForOp forOp, unsigned inductionVarResultIdx,
                            PatternRewriter &rewriter) const {
    Value inductionVar = forOp.getInductionVar();
    Value step = forOp.getStep();
    Value lb = forOp.getLowerBound();
    Value initVal = forOp.getInitArgs()[inductionVarResultIdx];
    BlockArgument iterArg = forOp.getRegionIterArg(inductionVarResultIdx);
    Location loc = forOp.getLoc();

    // Step 1: Replace uses of the iter_arg inside the loop by
    // select(arg == lb, init, arg - step)
    for (OpOperand &use : llvm::make_early_inc_range(iterArg.getUses())) {
      rewriter.setInsertionPoint(use.getOwner());
      Value cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                 inductionVar, lb);
      Value sub = rewriter.create<arith::SubIOp>(loc, inductionVar, step);
      Value newVal = rewriter.create<arith::SelectOp>(loc, cmp, initVal, sub);
      rewriter.replaceAllUsesWith(use.get(), newVal);
    }

    // Step 2: Create new for loop without the induction var tracking
    SmallVector<Value> newIterArgs;
    for (auto [idx, iterArg] : llvm::enumerate(forOp.getInitArgs())) {
      if (idx != inductionVarResultIdx)
        newIterArgs.push_back(iterArg);
    }

    rewriter.setInsertionPointAfter(forOp);
    auto newForOp = rewriter.create<scf::ForOp>(loc, lb, forOp.getUpperBound(),
                                                step, newIterArgs);
    newForOp->setAttrs(forOp->getAttrs());

    // Move body blocks to the new for op (excluding the iter_arg we removed)
    Block *newBody = newForOp.getBody();
    Block *oldBody = forOp.getBody();

    // Map old block args to new ones, skipping our removed arg
    IRMapping mapping;
    for (unsigned i = 0, j = 0; i < oldBody->getNumArguments(); ++i) {
      if (i == inductionVarResultIdx + 1)
        continue; // +1 for induction var
      mapping.map(oldBody->getArgument(i), newBody->getArgument(j++));
    }

    // Move all operations, remapping any operands that might refer to old block
    // arguments.
    Block::iterator it = newIterArgs.empty()
                             ? newBody->getTerminator()->getIterator()
                             : newBody->end();
    for (auto &op : oldBody->getOperations()) {
      if (isa<scf::YieldOp>(op))
        continue;
      Operation *newOp = rewriter.clone(op, mapping);
      rewriter.moveOpBefore(newOp, newBody, it);
    }

    // Update the yield op to not yield the induction var
    auto oldYieldOp = cast<scf::YieldOp>(oldBody->getTerminator());
    SmallVector<Value> newYieldVals;
    for (auto [idx, val] : llvm::enumerate(oldYieldOp.getOperands())) {
      if (idx != inductionVarResultIdx)
        newYieldVals.push_back(mapping.lookup(val));
    }
    rewriter.setInsertionPointToEnd(newBody);
    if (!newYieldVals.empty())
      rewriter.create<scf::YieldOp>(oldYieldOp.getLoc(), newYieldVals);
    rewriter.eraseOp(oldYieldOp);

    // Step 3: Replace old loop uses.

    // Replace induction var result with
    // select(ub <= lb, init, last_value)
    // with last_value = lb + step * floor((ub - lb - 1) / step) + step
    if (!forOp.getResult(inductionVarResultIdx).use_empty()) {
      rewriter.setInsertionPointAfter(newForOp);
      Value ub = forOp.getUpperBound();
      auto computeLastValue = [&]() -> Value {
        Value diff = rewriter.create<arith::SubIOp>(loc, ub, lb);
        Value one = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIntegerAttr(lb.getType(), 1));
        Value diffMinus1 = rewriter.create<arith::SubIOp>(loc, diff, one);
        Value iterations =
            rewriter.create<arith::DivUIOp>(loc, diffMinus1, step);
        Value stepTimesIters =
            rewriter.create<arith::MulIOp>(loc, step, iterations);
        Value lastValue =
            rewriter.create<arith::AddIOp>(loc, step, stepTimesIters);
        return rewriter.create<arith::AddIOp>(loc, lb, lastValue);
      };
      Value lastValue = computeLastValue();
      // TODO: handle negative step
      Value cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ule,
                                                 ub, lb);

      Value newVal =
          rewriter.create<arith::SelectOp>(loc, cmp, initVal, lastValue);
      rewriter.replaceAllUsesWith(forOp.getResult(inductionVarResultIdx),
                                  newVal);
    }

    // Replace other results
    for (unsigned i = 0; i < forOp.getNumResults(); ++i) {
      if (i != inductionVarResultIdx) {
        unsigned newResultIdx = i > inductionVarResultIdx ? i - 1 : i;
        rewriter.replaceAllUsesWith(forOp.getResult(i),
                                    newForOp.getResult(newResultIdx));
      }
    }

    // Finally erase the old loop.
    rewriter.eraseOp(forOp);

    return success();
  }
};

struct RemoveUnusedForResults : public OpRewritePattern<ForOp> {
  using OpRewritePattern<ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForOp op,
                                PatternRewriter &rewriter) const override {

    bool changed = false;
    for (auto [iter, iterarg, yld, res] : llvm::zip(
             op.getInitArgs(), op.getRegionIterArgs(),
             op.getBody()->getTerminator()->getOperands(), op.getResults())) {
      bool replacable = iter == yld;
      Value replacement = yld;
      if (iter.getDefiningOp<ub::PoisonOp>()) {
        if (auto yldop = yld.getDefiningOp()) {
          if (!op->isAncestor(yldop))
            replacable = true;
        } else if (auto blk = dyn_cast<BlockArgument>(yld)) {
          if (!op->isAncestor(blk.getOwner()->getParentOp()))
            replacable = true;
        }
      }
      if (yld == iterarg) {
        replacable = true;
        replacement = iter;
        if (!iterarg.use_empty()) {
          rewriter.modifyOpInPlace(
              op, [&] { iterarg.replaceAllUsesWith(replacement); });
          changed = true;
        }
      }
      if (!res.use_empty() && replacable) {
        rewriter.modifyOpInPlace(op,
                                 [&] { res.replaceAllUsesWith(replacement); });
        changed = true;
      }
    }
    return success(changed);
  }
};

/// Remove unused iterator operands.
// TODO: IRMapping for indvar.
struct RemoveUnusedArgs : public OpRewritePattern<ForOp> {
  using OpRewritePattern<ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForOp op,
                                PatternRewriter &rewriter) const override {

    SmallVector<Value, 2> usedBlockArgs;
    SmallVector<OpResult, 2> usedResults;
    SmallVector<Value, 2> usedOperands;

    unsigned i = 0;
    // if the block argument or the result at the
    // same index position have uses do not eliminate.
    for (auto blockArg : op.getRegionIterArgs()) {
      if ((!blockArg.use_empty()) || (!op.getResult(i).use_empty())) {
        usedOperands.push_back(op.getOperand(op.getNumControlOperands() + i));
        usedResults.push_back(op->getOpResult(i));
        usedBlockArgs.push_back(blockArg);
      }
      i++;
    }

    // no work to do.
    if (usedOperands.size() == op.getInits().size())
      return failure();

    auto newForOp =
        rewriter.create<ForOp>(op.getLoc(), op.getLowerBound(),
                               op.getUpperBound(), op.getStep(), usedOperands);
    newForOp->setAttrs(op->getAttrs());

    if (!newForOp.getBody()->empty())
      rewriter.eraseOp(newForOp.getBody()->getTerminator());

    newForOp.getBody()->getOperations().splice(
        newForOp.getBody()->getOperations().begin(),
        op.getBody()->getOperations());

    rewriter.modifyOpInPlace(op, [&] {
      op.getInductionVar().replaceAllUsesWith(newForOp.getInductionVar());
      for (auto pair : llvm::zip(usedBlockArgs, newForOp.getRegionIterArgs())) {
        std::get<0>(pair).replaceAllUsesWith(std::get<1>(pair));
      }
    });

    // adjust return.
    auto yieldOp = cast<scf::YieldOp>(newForOp.getBody()->getTerminator());
    SmallVector<Value, 2> usedYieldOperands{};
    llvm::transform(usedResults, std::back_inserter(usedYieldOperands),
                    [&](OpResult result) {
                      return yieldOp.getOperand(result.getResultNumber());
                    });
    rewriter.setInsertionPoint(yieldOp);
    rewriter.replaceOpWithNewOp<YieldOp>(yieldOp, usedYieldOperands);

    // Replace the operation's results with the new ones.
    SmallVector<Value, 4> repResults(op.getNumResults());
    for (auto en : llvm::enumerate(usedResults))
      repResults[cast<OpResult>(en.value()).getResultNumber()] =
          newForOp.getResult(en.index());

    rewriter.replaceOp(op, repResults);
    return success();
  }
};

struct ReplaceRedundantArgs : public OpRewritePattern<ForOp> {
  using OpRewritePattern<ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForOp op,
                                PatternRewriter &rewriter) const override {
    auto yieldOp = cast<scf::YieldOp>(op.getBody()->getTerminator());
    bool replaced = false;
    unsigned i = 0;
    for (auto blockArg : op.getRegionIterArgs()) {
      for (unsigned j = 0; j < i; j++) {
        if (op.getOperand(op.getNumControlOperands() + i) ==
                op.getOperand(op.getNumControlOperands() + j) &&
            yieldOp.getOperand(i) == yieldOp.getOperand(j)) {

          rewriter.modifyOpInPlace(op, [&] {
            op.getResult(i).replaceAllUsesWith(op.getResult(j));
            blockArg.replaceAllUsesWith(op.getRegionIterArgs()[j]);
          });
          replaced = true;
          goto skip;
        }
      }
    skip:
      i++;
    }

    return success(replaced);
  }
};

/*
+struct RemoveNotIf : public OpRewritePattern<IfOp> {
+  using OpRewritePattern<IfOp>::OpRewritePattern;
+
+  LogicalResult matchAndRewrite(IfOp op,
+                                PatternRewriter &rewriter) const override {
+    // Replace the operation if only a subset of its results have uses.
+    if (op.getNumResults() == 0)
+      return failure();
+
+    auto trueYield =
cast<scf::YieldOp>(op.thenRegion().back().getTerminator());
+    auto falseYield =
+        cast<scf::YieldOp>(op.thenRegion().back().getTerminator());
+
+    rewriter.setInsertionPoint(op->getBlock(),
+                               op.getOperation()->getIterator());
+    bool changed = false;
+    for (auto tup :
+         llvm::zip(trueYield.results(), falseYield.results(), op.results())) {
+      if (!std::get<0>(tup).getType().isInteger(1))
+        continue;
+      if (auto top = std::get<0>(tup).getDefiningOp<ConstantOp>()) {
+        if (auto fop = std::get<1>(tup).getDefiningOp<ConstantOp>()) {
+          if (top.getValue().cast<IntegerAttr>().getValue() == 0 &&
+              fop.getValue().cast<IntegerAttr>().getValue() == 1) {
+
+            for (OpOperand &use :
+                 llvm::make_early_inc_range(std::get<2>(tup).getUses())) {
+              changed = true;
+              rewriter.modifyOpInPlace(use.getOwner(), [&]() {
+                use.set(rewriter.create<XOrOp>(op.getLoc(), op.condition()));
+              });
+            }
+          }
+          if (top.getValue().cast<IntegerAttr>().getValue() == 1 &&
+              fop.getValue().cast<IntegerAttr>().getValue() == 0) {
+            for (OpOperand &use :
+                 llvm::make_early_inc_range(std::get<2>(tup).getUses())) {
+              changed = true;
+              rewriter.modifyOpInPlace(use.getOwner(),
+                                         [&]() { use.set(op.condition()); });
+            }
+          }
+        }
+      }
+    }
+    return changed ? success() : failure();
+  }
+};
+struct RemoveBoolean : public OpRewritePattern<IfOp> {
+  using OpRewritePattern<IfOp>::OpRewritePattern;
+
+  LogicalResult matchAndRewrite(IfOp op,
+                                PatternRewriter &rewriter) const override {
+    bool changed = false;
+
+    if (llvm::all_of(op.results(), [](Value v) {
+          return isa<IntegerType>(v.getType()) &&
+                 v.getType().cast<IntegerType>().getWidth() == 1;
+        })) {
+      if (op.thenRegion().getBlocks().size() == 1 &&
+          op.elseRegion().getBlocks().size() == 1) {
+        while (isa<CmpIOp>(op.thenRegion().front().front())) {
+          op.thenRegion().front().front().moveBefore(op);
+          changed = true;
+        }
+        while (isa<CmpIOp>(op.elseRegion().front().front())) {
+          op.elseRegion().front().front().moveBefore(op);
+          changed = true;
+        }
+        if (op.thenRegion().front().getOperations().size() == 1 &&
+            op.elseRegion().front().getOperations().size() == 1) {
+          auto yop1 =
+              cast<scf::YieldOp>(op.thenRegion().front().getTerminator());
+          auto yop2 =
+              cast<scf::YieldOp>(op.elseRegion().front().getTerminator());
+          size_t idx = 0;
+
+          auto c1 = (mlir::Value)rewriter.create<ConstantOp>(
+              op.getLoc(), op.condition().getType(),
+              rewriter.getIntegerAttr(op.condition().getType(), 1));
+          auto notcond = (mlir::Value)rewriter.create<mlir::XOrOp>(
+              op.getLoc(), op.condition(), c1);
+
+          std::vector<Value> replacements;
+          for (auto res : op.results()) {
+            auto rep = rewriter.create<OrOp>(
+                op.getLoc(),
+                rewriter.create<AndOp>(op.getLoc(), op.condition(),
+                                       yop1.results()[idx]),
+                rewriter.create<AndOp>(op.getLoc(), notcond,
+                                       yop2.results()[idx]));
+            replacements.push_back(rep);
+            idx++;
+          }
+          rewriter.replaceOp(op, replacements);
+          // op.erase();
+          return success();
+        }
+      }
+    }
+
+    if (op.thenRegion().getBlocks().size() == 1 &&
+        op.elseRegion().getBlocks().size() == 1 &&
+        op.thenRegion().front().getOperations().size() == 1 &&
+        op.elseRegion().front().getOperations().size() == 1) {
+      auto yop1 = cast<scf::YieldOp>(op.thenRegion().front().getTerminator());
+      auto yop2 = cast<scf::YieldOp>(op.elseRegion().front().getTerminator());
+      size_t idx = 0;
+
+      std::vector<Value> replacements;
+      for (auto res : op.results()) {
+        auto rep =
+            rewriter.create<SelectOp>(op.getLoc(), op.condition(),
+                                      yop1.results()[idx],
yop2.results()[idx]);
+        replacements.push_back(rep);
+        idx++;
+      }
+      rewriter.replaceOp(op, replacements);
+      return success();
+    }
+    return changed ? success() : failure();
+  }
+};
*/

bool isTopLevelArgValue(Value value, Region *region) {
  if (auto arg = dyn_cast<BlockArgument>(value))
    return arg.getParentRegion() == region;
  return false;
}

bool isBlockArg(Value value) {
  if (auto arg = dyn_cast<BlockArgument>(value))
    return true;
  return false;
}

bool dominateWhile(Value value, WhileOp loop) {
  if (Operation *op = value.getDefiningOp()) {
    DominanceInfo dom(loop);
    return dom.properlyDominates(op, loop);
  } else if (auto arg = dyn_cast<BlockArgument>(value)) {
    return arg.getOwner()->getParentOp()->isProperAncestor(loop);
  } else {
    assert("????");
    return false;
  }
}

bool canMoveOpOutsideWhile(Operation *op, WhileOp loop) {
  DominanceInfo dom(loop);
  for (auto operand : op->getOperands()) {
    if (!dom.properlyDominates(operand, loop))
      return false;
  }
  return true;
}

class truncProp final : public OpRewritePattern<TruncIOp> {
public:
  using OpRewritePattern<TruncIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TruncIOp op,
                                PatternRewriter &rewriter) const override {
    auto ifOp = op.getIn().getDefiningOp<scf::IfOp>();
    if (!ifOp)
      return failure();

    auto idx = cast<OpResult>(op.getIn()).getResultNumber();
    bool change = false;
    for (auto v :
         {ifOp.thenYield().getOperand(idx), ifOp.elseYield().getOperand(idx)}) {
      change |= v.getDefiningOp<ConstantIntOp>() ||
                v.getDefiningOp<mlir::LLVM::UndefOp>();
      if (auto extOp = v.getDefiningOp<ExtUIOp>())
        if (auto it = dyn_cast<IntegerType>(extOp.getIn().getType()))
          change |= it.getWidth() == 1;
      if (auto extOp = v.getDefiningOp<ExtSIOp>())
        if (auto it = dyn_cast<IntegerType>(extOp.getIn().getType()))
          change |= it.getWidth() == 1;
    }
    if (!change) {
      return failure();
    }

    // Avoid creating redundant results
    if (!op.getOperand().hasOneUse())
      return failure();

    SmallVector<Type> resultTypes;
    llvm::append_range(resultTypes, ifOp.getResultTypes());
    resultTypes.push_back(op.getType());

    rewriter.setInsertionPoint(ifOp);
    auto nop = rewriter.create<scf::IfOp>(
        ifOp.getLoc(), resultTypes, ifOp.getCondition(), /*hasElse*/ true);
    rewriter.eraseBlock(nop.thenBlock());
    rewriter.eraseBlock(nop.elseBlock());

    rewriter.inlineRegionBefore(ifOp.getThenRegion(), nop.getThenRegion(),
                                nop.getThenRegion().begin());
    rewriter.inlineRegionBefore(ifOp.getElseRegion(), nop.getElseRegion(),
                                nop.getElseRegion().begin());

    SmallVector<Value> thenYields;
    llvm::append_range(thenYields, nop.thenYield().getOperands());
    rewriter.setInsertionPoint(nop.thenYield());
    thenYields.push_back(
        rewriter.create<TruncIOp>(op.getLoc(), op.getType(), thenYields[idx]));
    rewriter.replaceOpWithNewOp<scf::YieldOp>(nop.thenYield(), thenYields);

    SmallVector<Value> elseYields;
    llvm::append_range(elseYields, nop.elseYield().getOperands());
    rewriter.setInsertionPoint(nop.elseYield());
    elseYields.push_back(
        rewriter.create<TruncIOp>(op.getLoc(), op.getType(), elseYields[idx]));
    rewriter.replaceOpWithNewOp<scf::YieldOp>(nop.elseYield(), elseYields);
    rewriter.replaceOp(ifOp, nop.getResults().take_front(ifOp.getNumResults()));
    rewriter.replaceOp(op, nop.getResults().take_back(1));
    return success();
  }
};

struct WhileToForHelper {
  WhileOp loop;
  CmpIOp cmpIOp;
  bool cmpNegated;

  WhileToForHelper(WhileOp loop, CmpIOp cmpIOp, bool cmpNegated = false)
      : loop(loop), cmpIOp(cmpIOp), cmpNegated(cmpNegated) {}

  Value step;
  Value lb;
  bool lb_addOne;
  bool lb_addStep;
  bool lb_subStep;
  Value ub;
  bool ub_addOne;
  bool ub_addStep;
  bool ub_cloneMove;
  bool negativeStep;
  AddIOp addIOp;
  BlockArgument indVar;

  // Index within the compare (e.g. cmpArg0 ?= cmpArg1 ) of the iter arg.
  int compareIndex;

  // Comparing against the updated value instead of the iter arg.
  bool comparingUpdated;

  bool checkPredicate(PatternRewriter &rewriter) {
    auto cmpEnd = cmpIOp->getOperand(1 - compareIndex);
    auto cmpStart = loop.getInits()[indVar.getArgNumber()];

    if (!dominateWhile(cmpEnd, loop)) {
      auto *op = cmpIOp.getRhs().getDefiningOp();
      if (!op || !canMoveOpOutsideWhile(op, loop) ||
          (op->getNumResults() != 1)) {
        rewriter.notifyMatchFailure(loop, "Non-dominating rhs");
        return false;
      }
      ub_cloneMove = true;
    }

    auto pred = cmpIOp.getPredicate();
    if (cmpNegated) {
      pred = arith::invertPredicate(pred);
    }
    if (compareIndex == 1) {
      pred = swapPredicate(pred);
    }

    int stepInt = 0;
    assert(step);
    {
      APInt stepConstInt;
      if (matchPattern(step, m_ConstantInt(&stepConstInt))) {
        stepInt = stepConstInt.getSExtValue();
      }
    }

    switch (pred) {
    case CmpIPredicate::slt:
    case CmpIPredicate::ult: {
      lb = cmpStart;
      ub = cmpEnd;
      if (comparingUpdated) {
        lb_addStep = true;
      }

      if (stepInt < 0) {
        rewriter.notifyMatchFailure(
            loop, "Cmp less than with negative step unhandled");
        return false;
      } else if (stepInt == 0) {
        llvm::errs() << " Warning: unknown step size with less comparison\n";
      }

      break;
    }
    case CmpIPredicate::ule:
    case CmpIPredicate::sle: {
      lb = cmpStart;
      ub = cmpEnd;
      if (comparingUpdated) {
        lb_addStep = true;
      }
      ub_addOne = true;

      if (stepInt < 0) {
        rewriter.notifyMatchFailure(
            loop, "Cmp less than with negative step unhandled");
        return false;
      } else if (stepInt == 0) {
        llvm::errs() << " Warning: unknown step size with less comparison\n";
      }

      break;
    }
    case CmpIPredicate::uge:
    case CmpIPredicate::sge: {
      ub = cmpStart;
      lb = cmpEnd;
      if (comparingUpdated) {
        ub_addStep = true;
      }

      if (stepInt > 0) {
        rewriter.notifyMatchFailure(
            loop, "Cmp less than with positive step unhandled");
        return false;
      } else if (stepInt == 0) {
        llvm::errs() << " Warning: unknown step size with greater comparison\n";
      }

      break;
    }

    case CmpIPredicate::ugt:
    case CmpIPredicate::sgt: {
      ub = cmpStart;
      lb = cmpEnd;
      lb_addOne = true;
      if (comparingUpdated) {
        ub_addStep = true;
      }

      if (stepInt > 0) {
        rewriter.notifyMatchFailure(
            loop, "Cmp less than with positive step unhandled");
        return false;
      } else if (stepInt == 0) {
        llvm::errs() << " Warning: unknown step size with greater comparison\n";
      }

      break;
    }

    case CmpIPredicate::ne: {
      // Transform arith.cmpi NE to SLT /SGT
      // 1. Check to see if step size is negative or positive to decide
      // between SLT and SGT
      // 2. Check to see if linearly scaling IV from init with step size can
      // be equal to upperbound
      // 3. If yes and step size is negative, then we need to transform the
      // condition to SGT
      // 4. If yes and step size is positive, then we need to transform the
      // condition to SLT
      APInt lbConstInt, ubConstInt, stepConstInt;
      int lbInt, ubInt, stepInt;
      bool constantBounds = false;
      assert(step);
      if (matchPattern(step, m_ConstantInt(&stepConstInt))) {
        stepInt = stepConstInt.getSExtValue();
        if (matchPattern(cmpStart, m_ConstantInt(&lbConstInt)) &&
            matchPattern(cmpEnd, m_ConstantInt(&ubConstInt))) {
          lbInt = lbConstInt.getSExtValue();
          ubInt = ubConstInt.getSExtValue();
          constantBounds = true;
        }
      } else {
        rewriter.notifyMatchFailure(loop,
                                    "Predicate ne with non-constant step");
        return false;
      }

      if (stepInt == 1 || stepInt == -1 || (ubInt - lbInt) % stepInt == 0) {

        if ((stepInt < 0) && (!constantBounds || (ubInt < lbInt))) {
          if (!constantBounds) {
            llvm::errs() << " found loop with negative increment and "
                            "non-constant bounds, assuming no wrap around\n";
          }
          lb = cmpEnd;
          ub = cmpStart;

          // add one only if we compare with the updated iv
          lb_subStep = comparingUpdated;
          ub_addOne = false;
        } else if ((stepInt > 0) && (!constantBounds || (lbInt < ubInt))) {
          if (!constantBounds) {
            llvm::errs() << " found loop with positive increment and "
                            "non-constant bounds, assuming no wrap around\n";
          }
          lb = cmpStart;
          ub = cmpEnd;

          // inclusive range if the cmpiop compares with the indVar, not the
          // updated value
          lb_addStep = comparingUpdated;
        } else {
          rewriter.notifyMatchFailure(loop,
                                      "Predicate ne with unhandled bounds");
          return false;
        }
      } else {
        rewriter.notifyMatchFailure(loop,
                                    "Predicate ne with non-divisible bounds");
        return false; // If upperbound - lowerbound is not divisible by step
                      // size, then we cannot transform the condition
      }
      break;
    }
    case CmpIPredicate::eq: {
      rewriter.notifyMatchFailure(loop, "Predicate eq predicate unhandled");
      return false;
    }
    }

    return lb && ub;
  }

  void checkNegativeStep() {
    negativeStep = false;
    if (auto cop = step.getDefiningOp<ConstantIntOp>()) {
      if (cop.value() < 0) {
        negativeStep = true;
      }
    } else if (auto cop = step.getDefiningOp<ConstantIndexOp>()) {
      if (cop.value() < 0)
        negativeStep = true;
    }

    if (!negativeStep)
      lb = loop.getOperand(indVar.getArgNumber());
    else {
      ub = loop.getOperand(indVar.getArgNumber());
      ub_addOne = true;
    }
  }

  void initVariables() {
    step = nullptr;
    lb = nullptr;
    lb_addOne = false;
    lb_addStep = false;
    lb_subStep = false;
    ub = nullptr;
    ub_addOne = false;
    ub_addStep = false;
    ub_cloneMove = false;
    negativeStep = false;
  }

  // return true if legal
  bool considerStep(BlockArgument beforeArg, Value steppingVal, bool inBefore,
                    Value lookThrough) {

    assert(steppingVal);
    bool negateLookThrough = false;
    if (lookThrough) {
      while (auto neg = lookThrough.getDefiningOp<XOrIOp>())
        if (matchPattern(neg.getOperand(1), m_One())) {
          lookThrough = neg.getOperand(0);
          negateLookThrough = !negateLookThrough;
        }

      if (auto ifOp = steppingVal.getDefiningOp<IfOp>()) {
        Value condition = ifOp.getCondition();
        while (auto neg = condition.getDefiningOp<XOrIOp>())
          if (matchPattern(neg.getOperand(1), m_One())) {
            condition = neg.getOperand(0);
            negateLookThrough = !negateLookThrough;
          }
        if (ifOp.getCondition() == lookThrough && beforeArg) {
          for (auto r : llvm::enumerate(ifOp.getResults())) {
            if (r.value() == loop.getAfter()
                                 .front()
                                 .getTerminator()
                                 ->getOperands()[beforeArg.getArgNumber()]) {
              steppingVal =
                  (negateLookThrough ? ifOp.elseYield() : ifOp.thenYield())
                      .getOperand(r.index());
              break;
            }
          }
        }
      } else if (auto selOp = steppingVal.getDefiningOp<SelectOp>()) {
        Value condition = selOp.getCondition();
        while (auto neg = condition.getDefiningOp<XOrIOp>())
          if (matchPattern(neg.getOperand(1), m_One())) {
            condition = neg.getOperand(0);
            negateLookThrough = !negateLookThrough;
          }
        if (selOp.getCondition() == lookThrough)
          steppingVal = (negateLookThrough ? selOp.getFalseValue()
                                           : selOp.getTrueValue());
      }
    }

    assert(steppingVal);
    if (auto add = steppingVal.getDefiningOp<AddIOp>()) {
      for (int j = 0; j < 2; j++) {
        BlockArgument arg = beforeArg;
        if (inBefore) {
          if (beforeArg) {
            if (add->getOperand(j) != beforeArg) {
              continue;
            }
          } else {
            auto ba2 = dyn_cast<BlockArgument>(add->getOperand(j));
            if (!ba2)
              continue;
            if (ba2.getOwner() != &loop.getBefore().front())
              continue;
            arg = ba2;
          }
        } else {
          assert(beforeArg);
          auto ba2 = dyn_cast<BlockArgument>(add->getOperand(j));
          if (!ba2)
            continue;

          if (ba2.getOwner() != &loop.getAfter().front())
            continue;

          auto beforeYielded =
              loop.getConditionOp().getArgs()[ba2.getArgNumber()];
          if (beforeYielded != beforeArg) {
            continue;
          }
        }

        assert(arg);
        indVar = arg;
        step = add->getOperand(1 - j);
        addIOp = add;
        return true;
      }
    }
    return false;
  }

  bool computeLegality(PatternRewriter &rewriter, bool sizeCheck,
                       Value lookThrough = nullptr, bool doWhile = false) {

    initVariables();

    auto condOp = loop.getConditionOp();
    auto doYield = cast<scf::YieldOp>(loop.getAfter().front().getTerminator());

    Type extType = nullptr;
    comparingUpdated = false;

    for (int i = 0; i < 2; i++) {
      Value opVal = cmpIOp->getOperand(i);
      compareIndex = i;

      if (auto ext = opVal.getDefiningOp<ExtSIOp>()) {
        opVal = ext.getIn();
        extType = ext.getType();
      }

      if (auto arg = dyn_cast<BlockArgument>(opVal)) {
        if (arg.getOwner() != &loop.getBefore().front()) {
          continue;
        }

        auto pval = doYield.getOperand(arg.getArgNumber());

        if (auto afterArg = dyn_cast<BlockArgument>(pval)) {

          if (afterArg.getOwner() != &loop.getAfter().front()) {
            continue;
          }

          comparingUpdated = false;
          assert(condOp.getArgs()[afterArg.getArgNumber()]);
          if (considerStep(arg, condOp.getArgs()[afterArg.getArgNumber()],
                           /*inBefore*/ true, lookThrough)) {
            goto endDetect;
          }
        } else if (auto ifOp = pval.getDefiningOp<scf::IfOp>()) {
          auto opResult = dyn_cast<OpResult>(pval);

          unsigned resultNum = opResult.getResultNumber();
          Value steppingVal;
          auto thenYield =
              cast<scf::YieldOp>(ifOp.getThenRegion().front().getTerminator());
          auto elseYield =
              cast<scf::YieldOp>(ifOp.getElseRegion().front().getTerminator());
          if (thenYield.getOperand(resultNum).getDefiningOp<arith::AddIOp>())
            steppingVal = thenYield.getOperand(resultNum);
          else if (elseYield.getOperand(resultNum)
                       .getDefiningOp<arith::AddIOp>())
            steppingVal = elseYield.getOperand(resultNum);
          else
            goto endDetect;

          comparingUpdated = false;
          if (considerStep(arg, steppingVal, /*inBefore*/ false, lookThrough)) {
            goto endDetect;
          }
        } else {
          comparingUpdated = false;
          if (considerStep(arg, pval, /*inBefore*/ false, lookThrough)) {
            goto endDetect;
          }
        }
      } else {
        comparingUpdated = true;
        assert(opVal);
        if (considerStep(nullptr, opVal, /*inBefore*/ true, lookThrough)) {
          goto endDetect;
        }
      }
    }

  endDetect:;

    if (!indVar) {
      rewriter.notifyMatchFailure(loop, "Failed to find iv");
      return false;
    }
    assert(step);
    assert(addIOp);

    // Cannot transform for if step is not loop-invariant
    if (auto *op = step.getDefiningOp()) {
      if (loop->isAncestor(op)) {
        rewriter.notifyMatchFailure(loop, "Step is not loop invariant");
        return false;
      }
    }

    // Before region contains more than just the comparison
    if (!doWhile) {
      size_t size = loop.getBefore().front().getOperations().size();
      if (extType)
        size--;
      if (!sizeCheck)
        size--;
      if (size != 2) {
        rewriter.notifyMatchFailure(
            loop, "Before region contains more than just comparison");
        return false;
      }
    }

    checkNegativeStep();

    return checkPredicate(rewriter);
  }

  void prepareFor(PatternRewriter &rewriter) {

    assert(lb);
    assert(ub);
    assert(step);

    if (lb_addOne) {
      Value one =
          rewriter.create<ConstantIntOp>(loop.getLoc(), lb.getType(), 1);
      lb = rewriter.create<AddIOp>(loop.getLoc(), lb, one);
    }

    if (lb_addStep) {
      lb = rewriter.create<AddIOp>(loop.getLoc(), lb, step);
    }
    if (lb_subStep) {
      lb = rewriter.create<SubIOp>(loop.getLoc(), lb, step);
    }

    if (ub_cloneMove) {
      auto *op = ub.getDefiningOp();
      assert(op);
      auto *newOp = rewriter.clone(*op);
      rewriter.replaceOp(op, newOp->getResults());
      ub = newOp->getResult(0);
    }
    if (ub_addOne) {
      Value one =
          rewriter.create<ConstantIntOp>(loop.getLoc(), ub.getType(), 1);
      ub = rewriter.create<AddIOp>(loop.getLoc(), ub, one);
    }
    if (ub_addStep) {
      ub = rewriter.create<AddIOp>(loop.getLoc(), ub, step);
    }
    bool modifyTypeToIndex = true;
    if ((step.getType() == lb.getType()) && (ub.getType() == lb.getType())) {
      modifyTypeToIndex = false;
    }

    if (negativeStep) {
      auto zero = rewriter.create<arith::ConstantOp>(
          step.getLoc(), step.getType(), rewriter.getZeroAttr(step.getType()));
      step = rewriter.create<SubIOp>(step.getLoc(), zero, step);
    }

    // Only cast if the types of step, ub and lb are different
    if (modifyTypeToIndex) {
      ub = rewriter.create<IndexCastOp>(loop.getLoc(),
                                        IndexType::get(loop.getContext()), ub);
      lb = rewriter.create<IndexCastOp>(loop.getLoc(),
                                        IndexType::get(loop.getContext()), lb);
      step = rewriter.create<IndexCastOp>(
          loop.getLoc(), IndexType::get(loop.getContext()), step);
    }
  }
};

struct MoveWhileToFor : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp loop,
                                PatternRewriter &rewriter) const override {
    auto condOp = loop.getConditionOp();
    SmallVector<Value, 2> results = {condOp.getArgs()};
    WhileToForHelper helper(loop,
                            condOp.getCondition().getDefiningOp<CmpIOp>());
    Value lookThrough = nullptr;
    if (helper.cmpIOp) {
      if (!helper.computeLegality(rewriter, /*sizeCheck*/ true, lookThrough,
                                  /*doWhile*/ true)) {
        return failure();
      }
      assert(helper.lb);
      assert(helper.ub);
    } else if (auto andOp = condOp.getCondition().getDefiningOp<AndIOp>()) {
      bool legal = false;

      for (int i = 0; i < 2; i++) {
        helper.cmpIOp = andOp->getOperand(i).getDefiningOp<CmpIOp>();
        if (!helper.cmpIOp)
          continue;
        lookThrough = andOp->getOperand(1 - i);

        if (!helper.computeLegality(rewriter, /*sizeCheck*/ true, lookThrough,
                                    /*doWhile*/ true)) {
          continue;
        }
        legal = true;
        assert(helper.lb);
        assert(helper.ub);
      }
      if (!legal) {
        return rewriter.notifyMatchFailure(loop,
                                           "No legal and comparison found");
      }
    } else {
      return rewriter.notifyMatchFailure(loop, "No comparison found");
    }
    assert(helper.lb);
    assert(helper.ub);
    helper.prepareFor(rewriter);

    // input of the for goes the input of the scf::while plus the output taken
    // from the conditionOp.
    SmallVector<Value, 8> forArgs = llvm::to_vector(loop.getInits());

    for (Value arg : condOp.getArgs()) {
      Value newArg = nullptr;
      if (auto blockArg = dyn_cast<BlockArgument>(arg)) {
        if (blockArg.getOwner() == &condOp->getParentRegion()->front()) {
          newArg = loop.getOperand(blockArg.getArgNumber());
        }
      } else if (auto selectOp = arg.getDefiningOp<arith::SelectOp>()) {

        if (auto trueBlockArg =
                dyn_cast<BlockArgument>(selectOp.getTrueValue())) {
          newArg = loop.getOperand(trueBlockArg.getArgNumber());
        } else if (auto falseBlockArg =
                       dyn_cast<BlockArgument>(selectOp.getFalseValue())) {
          newArg = loop.getOperand(falseBlockArg.getArgNumber());
        }
      }
      if (!newArg)
        newArg = rewriter.create<ub::PoisonOp>(arg.getLoc(), arg.getType());
      forArgs.push_back(newArg);
    }

    // And finally, optionally, the extra condition from the and
    if (lookThrough) {
      forArgs.push_back(rewriter.create<arith::ConstantIntOp>(
          loop.getLoc(), rewriter.getI1Type(), true));
    }

    bool doWhile = false;
    for (auto &blk : loop.getBefore()) {
      for (auto &op : blk) {
        if (!mlir::isPure(&op)) {
          doWhile = true;
          break;
        }
      }
    }

    bool mutableAfter = false;
    if (doWhile) {
      for (auto &blk : loop.getAfter()) {
        for (auto &op : blk) {
          if (!mlir::isPure(&op)) {
            mutableAfter = true;
            break;
          }
        }
      }
    }

    auto oldYield = cast<scf::YieldOp>(loop.getAfter().front().getTerminator());

    auto ub = helper.ub;

    if (doWhile) {
      // we add step to force the extra iteration at the end, but also we need
      // to max with lb + step to make sure one iteration runs at all. max(lb +
      // step, ub + step) -> max(ub, lb) + step
      ub = rewriter.create<arith::MaxSIOp>(loop.getLoc(), ub, helper.lb);

      ub = rewriter.create<arith::AddIOp>(loop.getLoc(), ub, helper.step);
    }

    auto forloop = rewriter.create<scf::ForOp>(loop.getLoc(), helper.lb, ub,
                                               helper.step, forArgs);
    forloop->setAttrs(loop->getAttrs());

    if (!forloop.getBody()->empty())
      rewriter.eraseOp(forloop.getBody()->getTerminator());

    Value nextLookThrough = nullptr;

    rewriter.modifyOpInPlace(loop, [&] {
      for (auto pair : llvm::zip(loop.getBefore().getArguments(),
                                 forloop.getRegionIterArgs())) {
        std::get<0>(pair).replaceAllUsesWith(std::get<1>(pair));
        if (lookThrough == std::get<0>(pair))
          lookThrough = std::get<1>(pair);
      }
    });

    loop.getBefore().front().eraseArguments([](BlockArgument) { return true; });

    scf::IfOp ifOp;
    if (lookThrough) {
      rewriter.setInsertionPointToEnd(forloop.getBody());
      SmallVector<Type> resTys = llvm::to_vector(loop.getResultTypes());
      resTys.push_back(rewriter.getI1Type());

      ifOp = rewriter.create<scf::IfOp>(
          forloop.getLoc(), resTys, forloop.getRegionIterArgs().back(), true);

      rewriter.eraseBlock(&ifOp->getRegion(0).front());
      ifOp->getRegion(0).getBlocks().splice(
          ifOp->getRegion(0).getBlocks().begin(),
          loop->getRegion(0).getBlocks());

      rewriter.setInsertionPointToEnd(&ifOp.getThenRegion().front());
      auto toYield = llvm::to_vector(condOp.getArgs());

      toYield.push_back(lookThrough);
      rewriter.create<scf::YieldOp>(loop.getLoc(), toYield);

      rewriter.setInsertionPointToEnd(&ifOp.getElseRegion().front());

      toYield.clear();
      for (size_t i = 0; i < loop.getNumResults(); i++) {
        toYield.push_back(
            forloop.getRegionIterArgs()[i + loop.getInits().size()]);
      }

      toYield.push_back(rewriter.create<arith::ConstantIntOp>(
          loop.getLoc(), rewriter.getI1Type(), false));

      rewriter.create<scf::YieldOp>(loop.getLoc(), toYield);

      rewriter.modifyOpInPlace(loop, [&] {
        for (auto pair :
             llvm::zip(loop.getAfter().getArguments(), ifOp->getResults())) {
          std::get<0>(pair).replaceAllUsesWith(std::get<1>(pair));
        }
      });

      nextLookThrough = ifOp->getResults().back();
    } else {
      forloop.getBody()->getOperations().splice(
          forloop.getBody()->getOperations().begin(),
          loop.getBefore().front().getOperations());

      rewriter.modifyOpInPlace(loop, [&] {
        for (auto pair :
             llvm::zip(loop.getAfter().getArguments(), condOp.getArgs())) {
          std::get<0>(pair).replaceAllUsesWith(std::get<1>(pair));
        }
      });
    }

    loop.getAfter().front().eraseArguments([](BlockArgument) { return true; });

    SmallVector<Value> newYields(forArgs.size());

    for (size_t i = 0; i < loop.getResults().size(); i++) {
      newYields[loop.getInits().size() + i] =
          lookThrough ? ifOp->getResults()[i] : condOp.getArgs()[i];
    }

    if (lookThrough || (doWhile && loop.getInits().size() && mutableAfter)) {
      rewriter.setInsertionPointToEnd(forloop.getBody());
      SmallVector<Type> initTypes;
      for (auto T : loop.getInits())
        initTypes.push_back(T.getType());
      Value cond = rewriter.create<arith::CmpIOp>(
          forloop.getLoc(),
          helper.negativeStep ? arith::CmpIPredicate::sgt
                              : arith::CmpIPredicate::slt,
          forloop.getInductionVar(), helper.ub);
      if (lookThrough) {
        cond = rewriter.create<AndIOp>(loop.getLoc(), cond, nextLookThrough);
      }
      auto ifOp =
          rewriter.create<scf::IfOp>(forloop.getLoc(), initTypes, cond, true);

      rewriter.eraseBlock(&ifOp->getRegion(0).front());
      ifOp->getRegion(0).getBlocks().splice(
          ifOp->getRegion(0).getBlocks().begin(),
          loop->getRegion(1).getBlocks());

      rewriter.setInsertionPointToEnd(&ifOp.getThenRegion().front());
      rewriter.create<scf::YieldOp>(loop.getLoc(), oldYield.getResults());

      rewriter.setInsertionPointToEnd(&ifOp.getElseRegion().front());
      SmallVector<Value> nonArgs;
      for (auto A : loop.getInits()) {
        nonArgs.push_back(
            rewriter.create<ub::PoisonOp>(loop.getLoc(), A.getType()));
      }
      rewriter.create<scf::YieldOp>(loop.getLoc(), nonArgs);

      for (size_t i = 0; i < loop.getInits().size(); i++) {
        newYields[i] = ifOp->getResults()[i];
      }
      if (lookThrough) {
        newYields[newYields.size() - 1] = nextLookThrough;
      }
    } else {
      forloop.getBody()->getOperations().splice(
          forloop.getBody()->getOperations().end(),
          loop.getAfter().front().getOperations());
      for (size_t i = 0; i < loop.getInits().size(); i++) {
        newYields[i] = oldYield.getResults()[i];
      }
    }

    rewriter.setInsertionPointToEnd(forloop.getBody());
    rewriter.create<scf::YieldOp>(loop.getLoc(), newYields);

    rewriter.eraseOp(condOp);
    rewriter.eraseOp(oldYield);

    rewriter.replaceOp(loop,
                       forloop.getResults().slice(loop.getInits().size(),
                                                  loop.getResults().size()));
    return success();
  }
};

// If and and with something is preventing creating a for
// move the and into the after body guarded by an if
struct RotateWhileAnd : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp loop,
                                PatternRewriter &rewriter) const override {
    auto condOp = loop.getConditionOp();

    if (auto andIOp = condOp.getCondition().getDefiningOp<AndIOp>()) {
      for (int i = 0; i < 2; i++) {
        WhileToForHelper helper(loop,
                                andIOp->getOperand(i).getDefiningOp<CmpIOp>());
        if (!helper.cmpIOp)
          continue;

        YieldOp oldYield =
            cast<YieldOp>(loop.getAfter().front().getTerminator());

        Value extraCmp = andIOp->getOperand(1 - i);
        Value lookThrough = nullptr;
        if (auto BA = dyn_cast<BlockArgument>(extraCmp)) {
          lookThrough = oldYield.getOperand(BA.getArgNumber());
        }
        if (helper.computeLegality(rewriter, /*sizeCheck*/ false, lookThrough,
                                   /*doWhile*/ true)) {
          return rewriter.notifyMatchFailure(loop, "Hand legal while");
        }
      }
    }

    SmallVector<std::pair<Value, bool>> todo;

    SmallVector<Value> andValues;
    SmallVector<Value> negatedValues;

    todo.emplace_back(condOp.getCondition(), false);
    while (todo.size()) {
      auto &&[operand, negated] = todo.pop_back_val();
      if (!negated) {
        if (auto andop = operand.getDefiningOp<AndIOp>()) {
          todo.emplace_back(andop.getLhs(), negated);
          todo.emplace_back(andop.getRhs(), negated);
          continue;
        }
      } else {
        if (auto orop = operand.getDefiningOp<OrIOp>()) {
          todo.emplace_back(orop.getLhs(), negated);
          todo.emplace_back(orop.getRhs(), negated);
          continue;
        }
      }

      if (auto neg = operand.getDefiningOp<arith::XOrIOp>()) {
        if (matchPattern(neg.getRhs(), m_One())) {
          todo.emplace_back(neg.getLhs(), !negated);
          continue;
        }
      }

      auto cmpIOp = operand.getDefiningOp<CmpIOp>();
      if (!cmpIOp) {
        if (negated) {
          negatedValues.push_back(operand);
        } else {
          andValues.push_back(operand);
        }
        continue;
      }

      WhileToForHelper helper(loop, cmpIOp, negated);

      if (!helper.computeLegality(rewriter, /*sizeCheck*/ false,
                                  /*lookThrough*/ nullptr, /*doWhile*/ true)) {
        if (negated) {
          negatedValues.push_back(operand);
        } else {
          andValues.push_back(operand);
        }
        continue;
      }

      if (negatedValues.size() == 0 && andValues.size() == 0 && !negated) {
        return rewriter.notifyMatchFailure(loop, "No rewriting required");
      }

      rewriter.setInsertionPoint(condOp);
      Value current = nullptr;
      for (auto aval : andValues) {
        if (current == nullptr) {
          current = aval;
        } else {
          current = rewriter.create<AndIOp>(loop.getLoc(), current, aval);
        }
      }
      for (auto aval : negatedValues) {
        aval = rewriter.create<XOrIOp>(loop.getLoc(), aval,
                                       rewriter.create<arith::ConstantIntOp>(
                                           aval.getLoc(), aval.getType(), 1));
        if (current == nullptr) {
          current = aval;
        } else {
          current = rewriter.create<AndIOp>(loop.getLoc(), current, aval);
        }
      }
      for (auto &&[aval, negated] : todo) {
        if (negated)
          aval = rewriter.create<XOrIOp>(loop.getLoc(), aval,
                                         rewriter.create<arith::ConstantIntOp>(
                                             aval.getLoc(), aval.getType(), 1));
        if (current == nullptr) {
          current = aval;
        } else {
          current = rewriter.create<AndIOp>(loop.getLoc(), current, aval);
        }
      }

      Value cmpIOp2 = cmpIOp;
      if (negated) {
        cmpIOp2 = rewriter.create<CmpIOp>(
            loop.getLoc(), invertPredicate(cmpIOp.getPredicate()),
            cmpIOp.getLhs(), cmpIOp.getRhs());
      }
      if (current)
        current = rewriter.create<AndIOp>(loop.getLoc(), current, cmpIOp2);
      else
        current = cmpIOp2;

      rewriter.modifyOpInPlace(
          condOp, [&]() { condOp.getConditionMutable().assign(current); });
      return success();
    }

    return failure();
  }
};

struct MoveWhileDown : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    auto term = cast<scf::ConditionOp>(op.getBefore().front().getTerminator());
    if (auto ifOp = term.getCondition().getDefiningOp<scf::IfOp>()) {
      if (ifOp.getNumResults() != term.getArgs().size() + 1)
        return failure();
      if (ifOp.getResult(0) != term.getCondition())
        return failure();
      for (size_t i = 1; i < ifOp.getNumResults(); ++i) {
        if (ifOp.getResult(i) != term.getArgs()[i - 1])
          return failure();
      }
      auto yield1 =
          cast<scf::YieldOp>(ifOp.getThenRegion().front().getTerminator());
      auto yield2 =
          cast<scf::YieldOp>(ifOp.getElseRegion().front().getTerminator());
      if (auto cop = yield1.getOperand(0).getDefiningOp<ConstantIntOp>()) {
        if (cop.value() == 0)
          return failure();
      } else
        return failure();
      if (auto cop = yield2.getOperand(0).getDefiningOp<ConstantIntOp>()) {
        if (cop.value() != 0)
          return failure();
      } else
        return failure();
      if (ifOp.getElseRegion().front().getOperations().size() != 1)
        return failure();
      op.getAfter().front().getOperations().splice(
          op.getAfter().front().begin(),
          ifOp.getThenRegion().front().getOperations());
      rewriter.modifyOpInPlace(term, [&] {
        term.getConditionMutable().assign(ifOp.getCondition());
      });
      SmallVector<Value, 2> args;
      for (size_t i = 1; i < yield2.getNumOperands(); ++i) {
        args.push_back(yield2.getOperand(i));
      }
      rewriter.modifyOpInPlace(term,
                               [&] { term.getArgsMutable().assign(args); });
      rewriter.eraseOp(yield2);
      rewriter.eraseOp(ifOp);

      for (size_t i = 0; i < op.getAfter().front().getNumArguments(); ++i) {
        op.getAfter().front().getArgument(i).replaceAllUsesWith(
            yield1.getOperand(i + 1));
      }
      rewriter.eraseOp(yield1);
      // TODO move operands from begin to after
      SmallVector<Value> todo(op.getBefore().front().getArguments().begin(),
                              op.getBefore().front().getArguments().end());
      for (auto &op : op.getBefore().front()) {
        for (auto res : op.getResults()) {
          todo.push_back(res);
        }
      }

      rewriter.modifyOpInPlace(op, [&] {
        for (auto val : todo) {
          auto na =
              op.getAfter().front().addArgument(val.getType(), op->getLoc());
          val.replaceUsesWithIf(na, [&](OpOperand &u) -> bool {
            return op.getAfter().isAncestor(u.getOwner()->getParentRegion());
          });
          args.push_back(val);
        }
      });

      rewriter.modifyOpInPlace(term,
                               [&] { term.getArgsMutable().assign(args); });

      SmallVector<Type, 4> tys;
      for (auto a : args)
        tys.push_back(a.getType());

      auto op2 = rewriter.create<WhileOp>(op.getLoc(), tys, op.getInits(),
                                          op->getAttrs());
      op2.getBefore().takeBody(op.getBefore());
      op2.getAfter().takeBody(op.getAfter());
      SmallVector<Value, 4> replacements;
      for (auto a : op2.getResults()) {
        if (replacements.size() == op.getResults().size())
          break;
        replacements.push_back(a);
      }
      rewriter.replaceOp(op, replacements);
      return success();
    }
    return failure();
  }
};

// Given code of the structure
// scf.while ()
//    ...
//    %z = if (%c) {
//       %i1 = ..
//       ..
//    } else {
//    }
//    condition (%c) %z#0 ..
//  } loop {
//    ...
//  }
// Move the body of the if into the lower loo

struct MoveWhileDown2 : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  /// Populates `crossing` with values (op results) that are defined in the
  /// same block as `op` and above it, and used by at least one op in the same
  /// block below `op`. Uses may be in nested regions.
  static void findValuesUsedBelow(IfOp op, llvm::SetVector<Value> &crossing) {
    for (Operation *it = op->getPrevNode(); it != nullptr;
         it = it->getPrevNode()) {
      for (Value value : it->getResults()) {
        for (Operation *user : value.getUsers()) {
          // ignore use of condition
          if (user == op)
            continue;

          if (op->isAncestor(user)) {
            crossing.insert(value);
            break;
          }
        }
      }
    }

    for (Value value : op->getBlock()->getArguments()) {
      for (Operation *user : value.getUsers()) {
        // ignore use of condition
        if (user == op)
          continue;

        if (op->isAncestor(user)) {
          crossing.insert(value);
          break;
        }
      }
    }
    // No need to process block arguments, they are assumed to be induction
    // variables and will be replicated.
  }

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    auto term = cast<scf::ConditionOp>(op.getBefore().front().getTerminator());
    if (auto ifOp = dyn_cast_or_null<scf::IfOp>(term->getPrevNode())) {
      if (ifOp.getCondition() != term.getCondition())
        return failure();

      SmallVector<std::pair<BlockArgument, Value>, 2> m;

      // The return results of the while which are used
      SmallVector<Value, 2> prevResults;
      // The corresponding value in the before which
      // is to be returned
      SmallVector<Value, 2> condArgs;

      SmallVector<std::pair<size_t, Value>, 2> afterYieldRewrites;
      auto afterYield = cast<YieldOp>(op.getAfter().front().back());
      for (auto pair :
           llvm::zip(op.getResults(), term.getArgs(), op.getAfterArguments())) {
        if (std::get<1>(pair).getDefiningOp() == ifOp) {

          Value thenYielded, elseYielded;
          for (auto p :
               llvm::zip(ifOp.thenYield().getResults(), ifOp.getResults(),
                         ifOp.elseYield().getResults())) {
            if (std::get<1>(pair) == std::get<1>(p)) {
              thenYielded = std::get<0>(p);
              elseYielded = std::get<2>(p);
              break;
            }
          }
          assert(thenYielded);
          assert(elseYielded);

          // If one of the if results is returned, only handle the case
          // where the value yielded is a block argument
          // %out-i:pair<0> = scf.while (... i:%blockArg=... ) {
          //   %z:j = scf.if (%c) {
          //      ...
          //   } else {
          //      yield ... j:%blockArg
          //   }
          //   condition %c ... i:pair<1>=%z:j
          // } loop ( ... i:) {
          //    yield   i:pair<2>
          // }
          if (!std::get<0>(pair).use_empty()) {
            if (auto blockArg = dyn_cast<BlockArgument>(elseYielded))
              if (blockArg.getOwner() == &op.getBefore().front()) {
                if (afterYield.getResults()[blockArg.getArgNumber()] ==
                        std::get<2>(pair) &&
                    op.getResults()[blockArg.getArgNumber()] ==
                        std::get<0>(pair)) {
                  prevResults.push_back(std::get<0>(pair));
                  condArgs.push_back(blockArg);
                  afterYieldRewrites.emplace_back(blockArg.getArgNumber(),
                                                  thenYielded);
                  continue;
                }
              }
            return failure();
          }
          // If the value yielded from then then is defined in the while
          // before but not being moved down with the if, don't change
          // anything.
          if (!ifOp.getThenRegion().isAncestor(thenYielded.getParentRegion()) &&
              op.getBefore().isAncestor(thenYielded.getParentRegion())) {
            prevResults.push_back(std::get<0>(pair));
            condArgs.push_back(thenYielded);
          } else {
            // Otherwise, mark the corresponding after argument to be replaced
            // with the value yielded in the if statement.
            m.emplace_back(std::get<2>(pair), thenYielded);
          }
        } else {
          assert(prevResults.size() == condArgs.size());
          prevResults.push_back(std::get<0>(pair));
          condArgs.push_back(std::get<1>(pair));
        }
      }

      SmallVector<Value> yieldArgs = afterYield.getResults();
      for (auto pair : afterYieldRewrites) {
        yieldArgs[pair.first] = pair.second;
      }

      rewriter.modifyOpInPlace(afterYield, [&] {
        afterYield.getResultsMutable().assign(yieldArgs);
      });
      Block *afterB = &op.getAfter().front();

      {
        llvm::SetVector<Value> sv;
        findValuesUsedBelow(ifOp, sv);

        for (auto v : sv) {
          condArgs.push_back(v);
          auto arg = afterB->addArgument(v.getType(), ifOp->getLoc());
          for (OpOperand &use : llvm::make_early_inc_range(v.getUses())) {
            if (ifOp->isAncestor(use.getOwner()) ||
                use.getOwner() == afterYield)
              rewriter.modifyOpInPlace(use.getOwner(), [&]() { use.set(arg); });
          }
        }
      }

      rewriter.setInsertionPoint(term);
      rewriter.replaceOpWithNewOp<ConditionOp>(term, term.getCondition(),
                                               condArgs);

      BitVector toErase(afterB->getNumArguments());
      for (int i = m.size() - 1; i >= 0; i--) {
        assert(m[i].first.getType() == m[i].second.getType());
        m[i].first.replaceAllUsesWith(m[i].second);
        toErase[m[i].first.getArgNumber()] = true;
      }
      afterB->eraseArguments(toErase);

      rewriter.eraseOp(ifOp.thenYield());
      Block *thenB = ifOp.thenBlock();
      afterB->getOperations().splice(afterB->getOperations().begin(),
                                     thenB->getOperations());

      rewriter.eraseOp(ifOp);

      SmallVector<Type, 4> resultTypes;
      for (auto v : condArgs) {
        resultTypes.push_back(v.getType());
      }

      rewriter.setInsertionPoint(op);
      auto nop = rewriter.create<WhileOp>(op.getLoc(), resultTypes,
                                          op.getInits(), op->getAttrs());
      nop.getBefore().takeBody(op.getBefore());
      nop.getAfter().takeBody(op.getAfter());

      rewriter.modifyOpInPlace(op, [&] {
        for (auto pair : llvm::enumerate(prevResults)) {
          pair.value().replaceAllUsesWith(nop.getResult(pair.index()));
        }
      });

      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

struct MoveWhileInvariantIfResult : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<BlockArgument, 2> origAfterArgs(op.getAfterArguments().begin(),
                                                op.getAfterArguments().end());
    bool changed = false;
    scf::ConditionOp term =
        cast<scf::ConditionOp>(op.getBefore().front().getTerminator());
    assert(origAfterArgs.size() == op.getResults().size());
    assert(origAfterArgs.size() == term.getArgs().size());

    for (auto pair :
         llvm::zip(op.getResults(), term.getArgs(), origAfterArgs)) {
      if (!std::get<0>(pair).use_empty()) {
        if (auto ifOp = std::get<1>(pair).getDefiningOp<scf::IfOp>()) {
          if (ifOp.getCondition() == term.getCondition()) {
            auto idx = cast<OpResult>(std::get<1>(pair)).getResultNumber();
            Value returnWith = ifOp.elseYield().getResults()[idx];
            if (!op.getBefore().isAncestor(returnWith.getParentRegion())) {
              rewriter.modifyOpInPlace(op, [&] {
                std::get<0>(pair).replaceAllUsesWith(returnWith);
              });
              changed = true;
            }
          }
        } else if (auto selOp =
                       std::get<1>(pair).getDefiningOp<arith::SelectOp>()) {
          if (selOp.getCondition() == term.getCondition()) {
            Value returnWith = selOp.getFalseValue();
            if (!op.getBefore().isAncestor(returnWith.getParentRegion())) {
              rewriter.modifyOpInPlace(op, [&] {
                std::get<0>(pair).replaceAllUsesWith(returnWith);
              });
              changed = true;
            }
          }
        }
      }
    }

    return success(changed);
  }
};

struct WhileLogicalNegation : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    bool changed = false;
    scf::ConditionOp term =
        cast<scf::ConditionOp>(op.getBefore().front().getTerminator());

    SmallPtrSet<Value, 1> condOps;
    SmallVector<Value> todo = {term.getCondition()};
    while (todo.size()) {
      Value val = todo.back();
      todo.pop_back();
      condOps.insert(val);
      if (auto ao = val.getDefiningOp<AndIOp>()) {
        todo.push_back(ao.getLhs());
        todo.push_back(ao.getRhs());
      }
    }

    for (auto pair :
         llvm::zip(op.getResults(), term.getArgs(), op.getAfterArguments())) {
      auto termArg = std::get<1>(pair);
      bool afterValue;
      if (condOps.count(termArg)) {
        afterValue = true;
      } else {
        bool found = false;
        if (auto termCmp = termArg.getDefiningOp<arith::CmpIOp>()) {
          for (auto cond : condOps) {
            if (auto condCmp = cond.getDefiningOp<CmpIOp>()) {
              if (termCmp.getLhs() == condCmp.getLhs() &&
                  termCmp.getRhs() == condCmp.getRhs()) {
                // TODO generalize to logical negation of
                if (condCmp.getPredicate() == CmpIPredicate::slt &&
                    termCmp.getPredicate() == CmpIPredicate::sge) {
                  found = true;
                  afterValue = false;
                  break;
                }
              }
            }
          }
        }
        if (!found)
          continue;
      }

      if (!std::get<0>(pair).use_empty()) {
        rewriter.modifyOpInPlace(op, [&] {
          rewriter.setInsertionPoint(op);
          auto truev =
              rewriter.create<ConstantIntOp>(op.getLoc(), !afterValue, 1);
          std::get<0>(pair).replaceAllUsesWith(truev);
        });
        changed = true;
      }
      if (!std::get<2>(pair).use_empty()) {
        rewriter.modifyOpInPlace(op, [&] {
          rewriter.setInsertionPointToStart(&op.getAfter().front());
          auto truev =
              rewriter.create<ConstantIntOp>(op.getLoc(), afterValue, 1);
          std::get<2>(pair).replaceAllUsesWith(truev);
        });
        changed = true;
      }
    }

    return success(changed);
  }
};

/// TODO move the addi down and repalce below with a subi
struct WhileCmpOffset : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<BlockArgument, 2> origAfterArgs(op.getAfterArguments().begin(),
                                                op.getAfterArguments().end());
    scf::ConditionOp term =
        cast<scf::ConditionOp>(op.getBefore().front().getTerminator());
    assert(origAfterArgs.size() == op.getResults().size());
    assert(origAfterArgs.size() == term.getArgs().size());

    if (auto condCmp = term.getCondition().getDefiningOp<CmpIOp>()) {
      if (auto addI = condCmp.getLhs().getDefiningOp<AddIOp>()) {
        if (addI.getOperand(1).getDefiningOp() &&
            !op.getBefore().isAncestor(
                addI.getOperand(1).getDefiningOp()->getParentRegion()))
          if (auto blockArg = dyn_cast<BlockArgument>(addI.getOperand(0))) {
            if (blockArg.getOwner() == &op.getBefore().front()) {
              auto rng = llvm::make_early_inc_range(blockArg.getUses());

              {
                rewriter.setInsertionPoint(op);
                SmallVector<Value> oldInits = op.getInits();
                oldInits[blockArg.getArgNumber()] = rewriter.create<AddIOp>(
                    addI.getLoc(), oldInits[blockArg.getArgNumber()],
                    addI.getOperand(1));
                op.getInitsMutable().assign(oldInits);
                rewriter.modifyOpInPlace(
                    addI, [&] { addI.replaceAllUsesWith(blockArg); });
              }

              YieldOp afterYield = cast<YieldOp>(op.getAfter().front().back());
              rewriter.setInsertionPoint(afterYield);
              SmallVector<Value> oldYields = afterYield.getResults();
              oldYields[blockArg.getArgNumber()] = rewriter.create<AddIOp>(
                  addI.getLoc(), oldYields[blockArg.getArgNumber()],
                  addI.getOperand(1));
              rewriter.modifyOpInPlace(afterYield, [&] {
                afterYield.getResultsMutable().assign(oldYields);
              });

              rewriter.setInsertionPointToStart(&op.getBefore().front());
              auto sub = rewriter.create<SubIOp>(addI.getLoc(), blockArg,
                                                 addI.getOperand(1));
              for (OpOperand &use : rng) {
                rewriter.modifyOpInPlace(use.getOwner(),
                                         [&]() { use.set(sub); });
              }
              rewriter.eraseOp(addI);
              return success();
            }
          }
      }
    }

    return failure();
  }
};

/// Given a while loop which yields a select whose condition is
/// the same as the condition, remove the select.
struct RemoveWhileSelect : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp loop,
                                PatternRewriter &rewriter) const override {
    scf::ConditionOp term =
        cast<scf::ConditionOp>(loop.getBefore().front().getTerminator());

    SmallVector<BlockArgument, 2> origAfterArgs(
        loop.getAfterArguments().begin(), loop.getAfterArguments().end());
    SmallVector<unsigned> newResults;
    SmallVector<unsigned> newAfter;
    SmallVector<Value> newYields;
    bool changed = false;
    for (auto pair :
         llvm::zip(loop.getResults(), term.getArgs(), origAfterArgs)) {
      auto selOp = std::get<1>(pair).getDefiningOp<arith::SelectOp>();
      if (!selOp || selOp.getCondition() != term.getCondition()) {
        newResults.push_back(newYields.size());
        newAfter.push_back(newYields.size());
        newYields.push_back(std::get<1>(pair));
        continue;
      }
      newResults.push_back(newYields.size());
      newYields.push_back(selOp.getFalseValue());
      newAfter.push_back(newYields.size());
      newYields.push_back(selOp.getTrueValue());
      changed = true;
    }
    if (!changed)
      return failure();

    SmallVector<Type, 4> resultTypes;
    for (auto v : newYields) {
      resultTypes.push_back(v.getType());
    }
    auto nop = rewriter.create<WhileOp>(loop.getLoc(), resultTypes,
                                        loop.getInits(), loop->getAttrs());

    nop.getBefore().takeBody(loop.getBefore());

    auto *after = rewriter.createBlock(&nop.getAfter());
    for (auto y : newYields)
      after->addArgument(y.getType(), loop.getLoc());

    SmallVector<Value> replacedArgs;
    for (auto idx : newAfter)
      replacedArgs.push_back(after->getArgument(idx));
    rewriter.mergeBlocks(&loop.getAfter().front(), after, replacedArgs);

    SmallVector<Value> replacedReturns;
    for (auto idx : newResults)
      replacedReturns.push_back(nop.getResult(idx));
    rewriter.replaceOp(loop, replacedReturns);
    rewriter.setInsertionPoint(term);
    rewriter.replaceOpWithNewOp<ConditionOp>(term, term.getCondition(),
                                             newYields);
    return success();
  }
};

struct MoveWhileDown3 : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    scf::ConditionOp term =
        cast<scf::ConditionOp>(op.getBefore().front().getTerminator());
    SmallVector<unsigned int, 2> toErase;
    SmallVector<Value, 2> newOps;
    SmallVector<Value, 2> condOps;
    SmallVector<BlockArgument, 2> origAfterArgs(op.getAfterArguments().begin(),
                                                op.getAfterArguments().end());
    SmallVector<Value, 2> returns;
    assert(origAfterArgs.size() == op.getResults().size());
    assert(origAfterArgs.size() == term.getArgs().size());
    for (auto pair :
         llvm::zip(op.getResults(), term.getArgs(), origAfterArgs)) {
      if (std::get<0>(pair).use_empty()) {
        if (std::get<2>(pair).use_empty()) {
          toErase.push_back(std::get<2>(pair).getArgNumber());
          continue;
        }
        // TODO generalize to any non memory effecting op
        if (auto idx =
                std::get<1>(pair).getDefiningOp<MemoryEffectOpInterface>()) {
          if (idx.hasNoEffect() &&
              !llvm::is_contained(newOps, std::get<1>(pair))) {
            Operation *cloned = std::get<1>(pair).getDefiningOp();
            if (!std::get<1>(pair).hasOneUse()) {
              cloned = std::get<1>(pair).getDefiningOp()->clone();
              op.getAfter().front().push_front(cloned);
            } else {
              cloned->moveBefore(&op.getAfter().front().front());
            }
            rewriter.modifyOpInPlace(std::get<1>(pair).getDefiningOp(), [&] {
              std::get<2>(pair).replaceAllUsesWith(cloned->getResult(0));
            });
            toErase.push_back(std::get<2>(pair).getArgNumber());
            for (auto &o :
                 llvm::make_early_inc_range(cloned->getOpOperands())) {
              {
                newOps.push_back(o.get());
                o.set(op.getAfter().front().addArgument(o.get().getType(),
                                                        o.get().getLoc()));
              }
            }
            continue;
          }
        }
      }
      condOps.push_back(std::get<1>(pair));
      returns.push_back(std::get<0>(pair));
    }
    if (toErase.size() == 0)
      return failure();

    condOps.append(newOps.begin(), newOps.end());

    BitVector toEraseVec(op.getAfter().front().getNumArguments());
    for (auto argNum : toErase)
      toEraseVec[argNum] = true;
    rewriter.modifyOpInPlace(
        term, [&] { op.getAfter().front().eraseArguments(toEraseVec); });
    rewriter.setInsertionPoint(term);
    rewriter.replaceOpWithNewOp<ConditionOp>(term, term.getCondition(),
                                             condOps);

    rewriter.setInsertionPoint(op);
    SmallVector<Type, 4> resultTypes;
    for (auto v : condOps) {
      resultTypes.push_back(v.getType());
    }
    auto nop = rewriter.create<WhileOp>(op.getLoc(), resultTypes, op.getInits(),
                                        op->getAttrs());

    nop.getBefore().takeBody(op.getBefore());
    nop.getAfter().takeBody(op.getAfter());

    rewriter.modifyOpInPlace(op, [&] {
      for (auto pair : llvm::enumerate(returns)) {
        pair.value().replaceAllUsesWith(nop.getResult(pair.index()));
      }
    });

    assert(resultTypes.size() == nop.getAfter().front().getNumArguments());
    assert(resultTypes.size() == condOps.size());

    rewriter.eraseOp(op);
    return success();
  }
};

#if 0
// Rewritten from LoopInvariantCodeMotion.cpp
struct WhileLICM : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;
  static bool canBeHoisted(Operation *op,
                           function_ref<bool(Value)> definedOutside,
                           bool isSpeculatable, WhileOp whileOp) {
    // TODO consider requirement of isSpeculatable

    // Check that dependencies are defined outside of loop.
    if (!llvm::all_of(op->getOperands(), definedOutside))
      return false;
    // Check whether this op is side-effect free. If we already know that there
    // can be no side-effects because the surrounding op has claimed so, we can
    // (and have to) skip this step.
    if (auto memInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
      if (!memInterface.hasNoEffect()) {
        if (isReadOnly(op) && !isSpeculatable) {

          SmallVector<MemoryEffects::EffectInstance> whileEffects;
          collectEffects(whileOp, whileEffects, /*ignoreBarriers*/ false);

          SmallVector<MemoryEffects::EffectInstance> opEffects;
          collectEffects(op, opEffects, /*ignoreBarriers*/ false);

          bool conflict = false;
          for (auto before : opEffects)
            for (auto after : whileEffects) {
              if (mayAlias(before, after)) {
                // Read, read is okay
                if (isa<MemoryEffects::Read>(before.getEffect()) &&
                    isa<MemoryEffects::Read>(after.getEffect())) {
                  continue;
                }

                // Write, write is not okay because may be different offsets and
                // the later must subsume other conflicts are invalid.
                conflict = true;
                break;
              }
            }
          if (conflict)
            return false;
        } else
          return false;
      }
      // If the operation doesn't have side effects and it doesn't recursively
      // have side effects, it can always be hoisted.
      if (!op->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
        return true;

      // Otherwise, if the operation doesn't provide the memory effect interface
      // and it doesn't have recursive side effects we treat it conservatively
      // as side-effecting.
    } else if (!op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
      return false;
    }

    // Recurse into the regions for this op and check whether the contained ops
    // can be hoisted.
    for (auto &region : op->getRegions()) {
      for (auto &block : region) {
        for (auto &innerOp : block.without_terminator())
          if (!canBeHoisted(&innerOp, definedOutside, isSpeculatable, whileOp))
            return false;
      }
    }
    return true;
  }

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    // We use two collections here as we need to preserve the order for
    // insertion and this is easiest.
    SmallPtrSet<Operation *, 8> willBeMovedSet;
    SmallVector<Operation *, 8> opsToMove;

    // Helper to check whether an operation is loop invariant wrt. SSA
    // properties.
    auto isDefinedOutsideOfBody = [&](Value value) {
      auto *definingOp = value.getDefiningOp();
      if (!definingOp) {
        if (auto ba = dyn_cast<BlockArgument>(value))
          definingOp = ba.getOwner()->getParentOp();
        assert(definingOp);
      }
      if (willBeMovedSet.count(definingOp))
        return true;
      return op != definingOp && !op->isAncestor(definingOp);
    };

    // Do not use walk here, as we do not want to go into nested regions and
    // hoist operations from there. These regions might have semantics unknown
    // to this rewriting. If the nested regions are loops, they will have been
    // processed.
    for (auto &block : op.getBefore()) {
      for (auto &iop : block.without_terminator()) {
        bool legal = canBeHoisted(&iop, isDefinedOutsideOfBody, false, op);
        if (legal) {
          opsToMove.push_back(&iop);
          willBeMovedSet.insert(&iop);
        }
      }
    }

    for (auto &block : op.getAfter()) {
      for (auto &iop : block.without_terminator()) {
        bool legal = canBeHoisted(&iop, isDefinedOutsideOfBody, true, op);
        if (legal) {
          opsToMove.push_back(&iop);
          willBeMovedSet.insert(&iop);
        }
      }
    }

    for (auto *moveOp : opsToMove)
      rewriter.modifyOpInPlace(moveOp, [&] { moveOp->moveBefore(op); });

    return success(opsToMove.size() > 0);
  }
};
#endif

struct RemoveUnusedCondVar : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    auto term = cast<scf::ConditionOp>(op.getBefore().front().getTerminator());
    SmallVector<Value, 4> conds;
    BitVector eraseArgs;
    SmallVector<unsigned, 4> keepArgs;
    SmallVector<Type, 4> tys;
    unsigned i = 0;
    std::map<void *, unsigned> valueOffsets;
    std::map<unsigned, unsigned> resultOffsets;
    SmallVector<Value, 4> resultArgs;
    for (auto pair :
         llvm::zip(term.getArgs(), op.getAfter().front().getArguments(),
                   op.getResults())) {
      auto arg = std::get<0>(pair);
      auto afarg = std::get<1>(pair);
      auto res = std::get<2>(pair);
      if (!op.getBefore().isAncestor(arg.getParentRegion())) {
        res.replaceAllUsesWith(arg);
        afarg.replaceAllUsesWith(arg);
      }
      if (afarg.use_empty() && res.use_empty()) {
        eraseArgs.push_back(true);
      } else if (valueOffsets.find(arg.getAsOpaquePointer()) !=
                 valueOffsets.end()) {
        resultOffsets[i] = valueOffsets[arg.getAsOpaquePointer()];
        afarg.replaceAllUsesWith(
            resultArgs[valueOffsets[arg.getAsOpaquePointer()]]);
        eraseArgs.push_back(true);
      } else {
        valueOffsets[arg.getAsOpaquePointer()] = keepArgs.size();
        resultOffsets[i] = keepArgs.size();
        resultArgs.push_back(afarg);
        conds.push_back(arg);
        keepArgs.push_back((unsigned)i);
        eraseArgs.push_back(false);
        tys.push_back(arg.getType());
      }
      i++;
    }
    assert(i == op.getAfter().front().getArguments().size());

    if (eraseArgs.any()) {

      rewriter.setInsertionPoint(term);
      rewriter.replaceOpWithNewOp<scf::ConditionOp>(term, term.getCondition(),
                                                    conds);

      rewriter.setInsertionPoint(op);
      auto op2 = rewriter.create<WhileOp>(op.getLoc(), tys, op.getInits(),
                                          op->getAttrs());

      op2.getBefore().takeBody(op.getBefore());
      op2.getAfter().takeBody(op.getAfter());
      for (auto pair : resultOffsets) {
        op.getResult(pair.first).replaceAllUsesWith(op2.getResult(pair.second));
      }
      rewriter.eraseOp(op);
      op2.getAfter().front().eraseArguments(eraseArgs);
      return success();
    }
    return failure();
  }
};

struct MoveSideEffectFreeWhile : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    scf::ConditionOp term =
        cast<scf::ConditionOp>(op.getBefore().front().getTerminator());
    SmallVector<Value, 4> conds(term.getArgs().begin(), term.getArgs().end());
    bool changed = false;
    unsigned i = 0;
    for (auto arg : term.getArgs()) {
      if (auto IC = arg.getDefiningOp<IndexCastOp>()) {
        if (arg.hasOneUse() && op.getResult(i).use_empty()) {
          auto rep = op.getAfter().front().addArgument(
              IC->getOperand(0).getType(), IC->getOperand(0).getLoc());
          IC->moveBefore(&op.getAfter().front(), op.getAfter().front().begin());
          conds.push_back(IC.getIn());
          IC.getInMutable().assign(rep);
          op.getAfter().front().getArgument(i).replaceAllUsesWith(
              IC->getResult(0));
          changed = true;
        }
      }
      i++;
    }
    if (changed) {
      SmallVector<Type, 4> tys;
      for (auto arg : conds) {
        tys.push_back(arg.getType());
      }
      auto op2 = rewriter.create<WhileOp>(op.getLoc(), tys, op.getInits(),
                                          op->getAttrs());
      op2.getBefore().takeBody(op.getBefore());
      op2.getAfter().takeBody(op.getAfter());
      unsigned j = 0;
      for (auto a : op.getResults()) {
        a.replaceAllUsesWith(op2.getResult(j));
        j++;
      }
      rewriter.eraseOp(op);
      rewriter.setInsertionPoint(term);
      rewriter.replaceOpWithNewOp<scf::ConditionOp>(term, term.getCondition(),
                                                    conds);
      return success();
    }
    return failure();
  }
};

struct SubToAdd : public OpRewritePattern<SubIOp> {
  using OpRewritePattern<SubIOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(SubIOp op,
                                PatternRewriter &rewriter) const override {
    if (auto cop = op.getOperand(1).getDefiningOp<ConstantIntOp>()) {
      rewriter.replaceOpWithNewOp<AddIOp>(
          op, op.getOperand(0),
          rewriter.create<ConstantIntOp>(cop.getLoc(), cop.getType(),
                                         -cop.value()));
      return success();
    }
    return failure();
  }
};

struct ReturnSq : public OpRewritePattern<ReturnOp> {
  using OpRewritePattern<ReturnOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ReturnOp op,
                                PatternRewriter &rewriter) const override {
    bool changed = false;
    SmallVector<Operation *> toErase;
    for (auto iter = op->getBlock()->rbegin();
         iter != op->getBlock()->rend() && &*iter != op; iter++) {
      changed = true;
      toErase.push_back(&*iter);
    }
    for (auto *op : toErase) {
      rewriter.eraseOp(op);
    }
    return success(changed);
  }
};

// From SCF.cpp
// Pattern to remove unused IfOp results.
struct RemoveUnusedResults : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  void transferBody(Block *source, Block *dest, ArrayRef<OpResult> usedResults,
                    PatternRewriter &rewriter) const {
    // Move all operations to the destination block.
    rewriter.mergeBlocks(source, dest);
    // Replace the yield op by one that returns only the used values.
    auto yieldOp = cast<scf::YieldOp>(dest->getTerminator());
    SmallVector<Value, 4> usedOperands;
    llvm::transform(usedResults, std::back_inserter(usedOperands),
                    [&](OpResult result) {
                      return yieldOp.getOperand(result.getResultNumber());
                    });
    rewriter.modifyOpInPlace(yieldOp,
                             [&]() { yieldOp->setOperands(usedOperands); });
  }

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter &rewriter) const override {
    // Compute the list of used results.
    SmallVector<OpResult, 4> usedResults;
    llvm::copy_if(op.getResults(), std::back_inserter(usedResults),
                  [](OpResult result) { return !result.use_empty(); });

    // Replace the operation if only a subset of its results have uses.
    if (usedResults.size() == op.getNumResults())
      return failure();

    // Compute the result types of the replacement operation.
    SmallVector<Type, 4> newTypes;
    llvm::transform(usedResults, std::back_inserter(newTypes),
                    [](OpResult result) { return result.getType(); });

    // Create a replacement operation with empty then and else regions.
    auto emptyBuilder = [](OpBuilder &, Location) {};
    auto newOp = rewriter.create<IfOp>(op.getLoc(), newTypes, op.getCondition(),
                                       emptyBuilder, emptyBuilder);

    // Move the bodies and replace the terminators (note there is a then and
    // an else region since the operation returns results).
    transferBody(op.getBody(0), newOp.getBody(0), usedResults, rewriter);
    transferBody(op.getBody(1), newOp.getBody(1), usedResults, rewriter);

    // Replace the operation by the new one.
    SmallVector<Value, 4> repResults(op.getNumResults());
    for (const auto &en : llvm::enumerate(usedResults))
      repResults[en.value().getResultNumber()] = newOp.getResult(en.index());
    rewriter.replaceOp(op, repResults);
    return success();
  }
};

struct SelectTruncToTruncSelect : public OpRewritePattern<TruncIOp> {
  using OpRewritePattern<TruncIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TruncIOp op,
                                PatternRewriter &rewriter) const override {
    auto selectOp = op.getIn().getDefiningOp<SelectOp>();
    if (!selectOp)
      return failure();

    // Get select operands and extract position
    auto cond = selectOp.getCondition();
    auto a = selectOp.getTrueValue();
    auto b = selectOp.getFalseValue();

    // Create new extract operations
    auto aTrunc = rewriter.create<TruncIOp>(op.getLoc(), op.getType(), a);
    auto bTrunc = rewriter.create<TruncIOp>(op.getLoc(), op.getType(), b);

    // Create new select with same condition and operands
    auto newSelect = rewriter.create<SelectOp>(selectOp.getLoc(), op.getType(),
                                               cond, aTrunc, bTrunc);

    // Replace old extract with new select
    rewriter.replaceOp(op, newSelect);

    return success();
  }
};

// If and and with something is preventing creating a for
// move the and into the after body guarded by an if
struct WhileShiftToInduction : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp loop,
                                PatternRewriter &rewriter) const override {
    auto condOp = loop.getConditionOp();

    if (!llvm::hasNItems(loop.getBefore().back(), 2))
      return failure();

    auto cmpIOp = condOp.getCondition().getDefiningOp<CmpIOp>();
    if (!cmpIOp) {
      return failure();
    }

    if (cmpIOp.getPredicate() != CmpIPredicate::ugt)
      return failure();

    if (!matchPattern(cmpIOp.getRhs(), m_Zero()))
      return failure();

    auto indVar = dyn_cast<BlockArgument>(cmpIOp.getLhs());
    if (!indVar)
      return failure();

    if (indVar.getOwner() != &loop.getBefore().front())
      return failure();

    auto endYield = cast<YieldOp>(loop.getAfter().back().getTerminator());

    // Check that the block argument is actually an induction var:
    //   Namely, its next value adds to the previous with an invariant step.
    auto shiftOp =
        endYield.getResults()[indVar.getArgNumber()].getDefiningOp<ShRUIOp>();
    if (!shiftOp)
      return failure();

    if (!matchPattern(shiftOp.getRhs(), m_One()))
      return failure();

    auto prevIndVar = dyn_cast<BlockArgument>(shiftOp.getLhs());
    if (!prevIndVar)
      return failure();

    if (prevIndVar.getOwner() != &loop.getAfter().front())
      return failure();

    if (condOp.getOperand(1 + prevIndVar.getArgNumber()) != indVar)
      return failure();

    auto startingV = loop.getInits()[indVar.getArgNumber()];

    Value lz =
        rewriter.create<math::CountLeadingZerosOp>(loop.getLoc(), startingV);
    if (!lz.getType().isIndex())
      lz = rewriter.create<IndexCastOp>(loop.getLoc(), rewriter.getIndexType(),
                                        lz);

    auto len = rewriter.create<SubIOp>(
        loop.getLoc(),
        rewriter.create<ConstantIndexOp>(
            loop.getLoc(), indVar.getType().getIntOrFloatBitWidth()),
        lz);

    SmallVector<Value> newInits(loop.getInits());
    newInits[indVar.getArgNumber()] =
        rewriter.create<ConstantIndexOp>(loop.getLoc(), 0);
    SmallVector<Type> postTys(loop.getResultTypes());
    postTys.push_back(rewriter.getIndexType());

    auto newWhile = rewriter.create<WhileOp>(loop.getLoc(), postTys, newInits,
                                             loop->getAttrs());
    rewriter.createBlock(&newWhile.getBefore());

    IRMapping map;
    Value newIndVar;
    for (auto a : loop.getBefore().front().getArguments()) {
      auto arg = newWhile.getBefore().addArgument(
          a == indVar ? rewriter.getIndexType() : a.getType(), a.getLoc());
      if (a != indVar)
        map.map(a, arg);
      else
        newIndVar = arg;
    }

    rewriter.setInsertionPointToEnd(&newWhile.getBefore().front());
    Value newCmp = rewriter.create<CmpIOp>(cmpIOp.getLoc(), CmpIPredicate::ult,
                                           newIndVar, len);
    map.map(cmpIOp, newCmp);

    Value newIndVarTyped = newIndVar;
    if (newIndVarTyped.getType() != indVar.getType())
      newIndVarTyped = rewriter.create<arith::IndexCastOp>(
          shiftOp.getLoc(), indVar.getType(), newIndVar);
    map.map(indVar, rewriter.create<ShRUIOp>(shiftOp.getLoc(), startingV,
                                             newIndVarTyped));
    SmallVector<Value> remapped;
    for (auto o : condOp.getArgs())
      remapped.push_back(map.lookup(o));
    remapped.push_back(newIndVar);
    rewriter.create<ConditionOp>(condOp.getLoc(), newCmp, remapped);

    newWhile.getAfter().takeBody(loop.getAfter());

    auto newPostInd = newWhile.getAfter().front().addArgument(
        rewriter.getIndexType(), loop.getLoc());
    auto yieldOp =
        cast<scf::YieldOp>(newWhile.getAfter().front().getTerminator());
    SmallVector<Value> yields(yieldOp.getOperands());
    rewriter.setInsertionPointToEnd(&newWhile.getAfter().front());
    yields[indVar.getArgNumber()] = rewriter.create<AddIOp>(
        loop.getLoc(), newPostInd,
        rewriter.create<arith::ConstantIndexOp>(loop.getLoc(), 1));
    rewriter.replaceOpWithNewOp<scf::YieldOp>(yieldOp, yields);

    SmallVector<Value> res(newWhile.getResults());
    res.pop_back();
    rewriter.replaceOp(loop, res);
    return success();
  }
};

// Transforms a select of a boolean to arithmetic operations
//
//  arith.select %arg, %x, %y : i1
//
//  becomes
//
//  and(%arg, %x) or and(!%arg, %y)
struct SelectI1Simplify : public OpRewritePattern<arith::SelectOp> {
  using OpRewritePattern<arith::SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SelectOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getType().isInteger(1))
      return failure();

    Value falseConstant =
        rewriter.create<arith::ConstantIntOp>(op.getLoc(), true, 1);
    Value notCondition = rewriter.create<arith::XOrIOp>(
        op.getLoc(), op.getCondition(), falseConstant);

    Value trueVal = rewriter.create<arith::AndIOp>(
        op.getLoc(), op.getCondition(), op.getTrueValue());
    Value falseVal = rewriter.create<arith::AndIOp>(op.getLoc(), notCondition,
                                                    op.getFalseValue());
    rewriter.replaceOpWithNewOp<arith::OrIOp>(op, trueVal, falseVal);
    return success();
  }
};

// This function checks if given operands can be yielded instead of moved
// outside the if operation
// Checks:
// 1. If operand is a block argument
// 2. If operand is not in the same region as the if operation
// 3. If there is only 1 unique user of op
// 4. If the if and else operands are not of the same operation
// 5. If the operands are not readnone
// In any of these cases we can't propagate the operand outside the if
// operation and are yielded instead
bool isLegalToSinkYieldedValue(Value thenOperand, Value elseOperand,
                               scf::IfOp ifOp) {
  if (thenOperand.getType() != elseOperand.getType())
    return false;

  for (auto operand : {thenOperand, elseOperand}) {
    auto defop = operand.getDefiningOp();
    if (!defop)
      return false;

    if (!ifOp->isAncestor(defop))
      return false;

    if (!isReadNone(operand.getDefiningOp()))
      return false;

    if (operand.getDefiningOp()->getNumRegions())
      return false;
  }

  if (thenOperand.getDefiningOp()->getName() !=
      elseOperand.getDefiningOp()->getName())
    return false;

  if (thenOperand.getDefiningOp()->getAttrDictionary() !=
      elseOperand.getDefiningOp()->getAttrDictionary())
    return false;

  // Get defining operations
  auto thenOp = thenOperand.getDefiningOp();
  auto elseOp = elseOperand.getDefiningOp();

  // Check operand types match
  if (thenOp->getNumOperands() != elseOp->getNumOperands())
    return false;

  for (unsigned i = 0; i < thenOp->getNumOperands(); ++i) {
    if (thenOp->getOperand(i).getType() != elseOp->getOperand(i).getType())
      return false;
  }

  return true;
}

std::pair<Value, size_t> checkOperands(
    scf::IfOp ifOp, Value operandIf, Value operandElse,
    llvm::MapVector<Operation *,
                    std::pair<Value, SmallVector<std::pair<Value, size_t>>>>
        &opsToMoveAfterIf,
    SmallVector<Value> &ifYieldOperands, SmallVector<Value> &elseYieldOperands,
    DenseMap<std::pair<Value, Value>, size_t> &thenOperationsToYieldIndex,
    PatternRewriter &rewriter) {

  if (operandIf == operandElse)
    return std::pair<Value, size_t>(operandIf, 0xdeadbeef);

  std::pair<Value, Value> key = {operandIf, operandElse};
  if (!isLegalToSinkYieldedValue(operandIf, operandElse, ifOp)) {
    if (!thenOperationsToYieldIndex.contains(key)) {
      thenOperationsToYieldIndex[key] = ifYieldOperands.size();
      ifYieldOperands.push_back(operandIf);
      elseYieldOperands.push_back(operandElse);
    }
    return std::pair<Value, size_t>(nullptr, thenOperationsToYieldIndex[key]);
  }

  Operation *opToMove = operandIf.getDefiningOp();

  auto foundAfterIf = opsToMoveAfterIf.find(opToMove);
  if (foundAfterIf != opsToMoveAfterIf.end()) {
    // We don't currently support the same if operand being moved after the if
    // when paired with a different instruction for the else
    if (foundAfterIf->second.first == operandElse)
      return std::pair<Value, size_t>(operandIf, 0xdeadbeef);
    else {
      if (!thenOperationsToYieldIndex.contains(key)) {
        thenOperationsToYieldIndex[key] = ifYieldOperands.size();
        ifYieldOperands.push_back(operandIf);
        elseYieldOperands.push_back(operandElse);
      }
      return std::pair<Value, size_t>(nullptr, thenOperationsToYieldIndex[key]);
    }
  }

  opsToMoveAfterIf.try_emplace(
      opToMove,
      std::make_pair(operandElse, SmallVector<std::pair<Value, size_t>>()));
  SmallVector<std::pair<Value, size_t>> newresults;

  for (auto [index, operands] : llvm::enumerate(
           llvm::zip_equal(operandIf.getDefiningOp()->getOperands(),
                           operandElse.getDefiningOp()->getOperands()))) {
    auto [thenOperand, elseOperand] = operands;
    newresults.push_back(checkOperands(
        ifOp, thenOperand, elseOperand, opsToMoveAfterIf, ifYieldOperands,
        elseYieldOperands, thenOperationsToYieldIndex, rewriter));
  }

  opsToMoveAfterIf[opToMove].second = std::move(newresults);

  return std::pair<Value, size_t>(operandIf, 0xdeadbeef);
}

// Algorithm:
// 1. Extract yield operations from both regions
// 2. Check if yield operations match in if and else region
// 3. If match, recursively check their operands to see if they can be moved
// as well
// 4. Track all ops which can be moved outside the if op
// 5. Track yiels ops which didn't change and the ones which changed.
// 6. Create a new if operation with updated yields
// 7. Updated source operands of operations moved outside to the new yields
// 8. Replace uses of the original if operation with the new one
struct IfYieldMovementPattern : public OpRewritePattern<scf::IfOp> {
  IfYieldMovementPattern(mlir::MLIRContext *context)
      : OpRewritePattern<scf::IfOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(scf::IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    // Ensure both regions exist and have single blocks
    if (ifOp.getThenRegion().empty() || ifOp.getElseRegion().empty())
      return failure();

    // Extract yield operations from both regions
    auto thenYield =
        cast<scf::YieldOp>(ifOp.getThenRegion().front().getTerminator());
    auto elseYield =
        cast<scf::YieldOp>(ifOp.getElseRegion().front().getTerminator());

    // List of replacement values for each of the original if's results
    // There are two kinds of replacements:
    //   1) A new value, which will be moved after the if statement
    //   2) if the value is null, the pair.second denotes the index of the new
    //   if
    //      statement that we should use here.
    SmallVector<std::pair<Value, size_t>> originalYields;

    // Use SetVector to ensure uniqueness while preserving order
    SmallVector<Value> ifYieldOperands, elseYieldOperands;
    llvm::MapVector<Operation *,
                    std::pair<Value, SmallVector<std::pair<Value, size_t>>>>
        opsToMoveAfterIf;

    // A list of operands defined within the if block, which have been
    // promoted to be yielded from the if statement. The size_t argument
    // denotes the index of the new if result which contains the value
    DenseMap<std::pair<Value, Value>, size_t> thenOperationsToYieldIndex;

    bool changed = false;

    for (auto [thenYieldOperand, elseYieldOperand] :
         llvm::zip(thenYield.getOperands(), elseYield.getOperands())) {

      auto yld =
          checkOperands(ifOp, thenYieldOperand, elseYieldOperand,
                        opsToMoveAfterIf, ifYieldOperands, elseYieldOperands,
                        thenOperationsToYieldIndex, rewriter);

      originalYields.emplace_back(yld);
      if (yld.first) {
        changed = true;
      }
    }

    // If no changes to yield operands, return failure
    if (!changed) {
      return failure();
    }

    // Create a new if operation with the same condition
    SmallVector<Type> resultTypes;

    // Cannot do unique, as unique might differ for if-else
    for (auto operand : ifYieldOperands) {
      resultTypes.push_back(operand.getType());
    }

    auto newIfOp = rewriter.create<scf::IfOp>(ifOp.getLoc(), resultTypes,
                                              ifOp.getCondition(),
                                              /*hasElse=*/true);

    // Move operations from the original then block to the new then block

    rewriter.eraseBlock(&newIfOp.getThenRegion().front());
    if (ifOp.getElseRegion().getBlocks().size()) {
      rewriter.eraseBlock(&newIfOp.getElseRegion().front());
    }

    rewriter.inlineRegionBefore(ifOp.getThenRegion(), newIfOp.getThenRegion(),
                                newIfOp.getThenRegion().begin());
    rewriter.inlineRegionBefore(ifOp.getElseRegion(), newIfOp.getElseRegion(),
                                newIfOp.getElseRegion().begin());

    // Create new yield in then block
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(newIfOp.thenBlock());
      rewriter.create<scf::YieldOp>(thenYield.getLoc(), ifYieldOperands);
      rewriter.eraseOp(thenYield);
    }

    // Create new yield in else block
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(newIfOp.elseBlock());
      rewriter.create<scf::YieldOp>(elseYield.getLoc(), elseYieldOperands);
      rewriter.eraseOp(elseYield);
    }

    IRMapping mappingAfterIf;

    rewriter.setInsertionPointAfter(newIfOp);
    for (auto &op : ifOp->getBlock()->getOperations()) {
      if (&op == ifOp)
        break;
      if (opsToMoveAfterIf.find(&op) != opsToMoveAfterIf.end()) {
        SmallVector<Value> operands;
        for (auto &&[valoperand, idxop] : opsToMoveAfterIf[&op].second) {
          if (valoperand)
            operands.push_back(mappingAfterIf.lookupOrDefault(valoperand));
          else
            operands.push_back(newIfOp.getResult(idxop));
        }
        auto *newOp = rewriter.create(op.getLoc(), op.getName().getIdentifier(),
                                      operands, op.getResultTypes(),
                                      op.getAttrs(), op.getSuccessors());

        mappingAfterIf.map(&op, newOp);
        for (auto &&[prev, post] :
             llvm::zip_equal(op.getResults(), newOp->getResults()))
          mappingAfterIf.map(prev, post);
      }
    }
    for (auto &op : newIfOp.thenBlock()->getOperations()) {
      if (opsToMoveAfterIf.find(&op) != opsToMoveAfterIf.end()) {
        SmallVector<Value> operands;
        for (auto &&[valoperand, idxop] : opsToMoveAfterIf[&op].second) {
          if (valoperand)
            operands.push_back(mappingAfterIf.lookupOrDefault(valoperand));
          else
            operands.push_back(newIfOp.getResult(idxop));
        }
        auto *newOp = rewriter.create(op.getLoc(), op.getName().getIdentifier(),
                                      operands, op.getResultTypes(),
                                      op.getAttrs(), op.getSuccessors());

        mappingAfterIf.map(&op, newOp);
        for (auto &&[prev, post] :
             llvm::zip_equal(op.getResults(), newOp->getResults()))
          mappingAfterIf.map(prev, post);
      }
    }

    // Replace uses of the original if operation with the new one
    SmallVector<Value> newResults;
    for (auto [idx, pair] : llvm::enumerate(originalYields)) {
      if (!pair.first) {
        newResults.push_back(newIfOp.getResult(pair.second));
      } else {
        newResults.push_back(mappingAfterIf.lookupOrDefault(pair.first));
      }
    }

    // Erase yield operations of prev if operation
    rewriter.replaceOp(ifOp, newResults);
    return success();
  }
};

bool isOneGreaterThan(Value v1, Value v2) {
  APInt i1, i2;
  if (matchPattern(v1, m_ConstantInt(&i1)) &&
      matchPattern(v2, m_ConstantInt(&i2))) {
    return i1 == (i2 + 1);
  }
  if (auto add = v1.getDefiningOp<arith::AddIOp>()) {
    if (add.getLhs() == v2 && matchPattern(add.getRhs(), m_One()))
      return true;
  }

  return false;
}

struct MaxSimplify : public OpRewritePattern<arith::MaxSIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::MaxSIOp maxOp,
                                PatternRewriter &rewriter) const override {
    Operation *op = maxOp;

    for (Operation *op = maxOp; op; op = op->getParentOp()) {
      auto ifOp = dyn_cast_or_null<scf::IfOp>(op->getParentOp());
      if (!ifOp)
        continue;
      auto cmpOp = ifOp.getCondition().getDefiningOp<arith::CmpIOp>();
      if (!cmpOp)
        continue;
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
          if (maxOp->getOperand(i) == cmpOp->getOperand(j)) {
            auto pred = cmpOp.getPredicate();
            if (j == 1)
              pred = swapPredicate(pred);
            if (pred == arith::CmpIPredicate::sgt &&
                isOneGreaterThan(maxOp->getOperand(1 - i),
                                 cmpOp->getOperand(1 - j))) {
              if (op->getParentRegion() == &ifOp.getThenRegion()) {
                rewriter.replaceOp(maxOp, maxOp->getOperand(i));
                return success();
              } else {
                rewriter.replaceOp(maxOp, maxOp->getOperand(1 - i));
                return success();
              }
            }

            if (pred == arith::CmpIPredicate::sge &&
                maxOp->getOperand(1 - i) == cmpOp->getOperand(1 - j)) {
              if (op->getParentRegion() == &ifOp.getThenRegion()) {
                rewriter.replaceOp(maxOp, maxOp->getOperand(i));
                return success();
              } else {
                rewriter.replaceOp(maxOp, maxOp->getOperand(1 - i));
                return success();
              }
            }
          }
        }
      }
    }
    return failure();
    // Ensure both regions exist and have single blocks
  }
};

struct ForBoundUnSwitch : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp loop,
                                PatternRewriter &rewriter) const override {

    auto ifOp = dyn_cast<scf::IfOp>(loop->getParentOp());
    if (!ifOp)
      return failure();

    auto cmpOp = ifOp.getCondition().getDefiningOp<arith::CmpIOp>();
    if (!cmpOp)
      return failure();

    Value niters = nullptr;
    if (matchPattern(loop.getLowerBound(), m_Zero())) {
      niters = loop.getUpperBound();
    } else if (auto addOp =
                   loop.getUpperBound().getDefiningOp<arith::AddIOp>()) {
      for (int i = 0; i < 2; i++) {
        if (addOp->getOperand(i) == loop.getLowerBound()) {
          niters = addOp->getOperand(1 - i);
          break;
        }
      }
    }

    if (!niters)
      return failure();

    bool legal = false;
    for (int j = 0; j < 2; j++) {
      if (niters == cmpOp->getOperand(j)) {
        auto pred = cmpOp.getPredicate();
        if (j == 1)
          pred = swapPredicate(pred);
        if (pred == arith::CmpIPredicate::sgt &&
            matchPattern(cmpOp->getOperand(1 - j), m_Zero())) {
          legal = true;
          break;
        }
      }
    }
    if (!legal)
      return failure();
    for (auto v : {loop.getLowerBound(), loop.getUpperBound()}) {
      if (auto op = v.getDefiningOp()) {
        if (op->getParentOp() == ifOp) {
          rewriter.modifyOpInPlace(op, [&]() { op->moveBefore(ifOp); });
        }
      }
    }

    if (&ifOp.getThenRegion().front().front() != loop) {
      return failure();
    }
    rewriter.modifyOpInPlace(loop, [&]() { loop->moveBefore(ifOp); });
    return success();
  }
};

struct ForLoopApplyEnzymeAttributes
    : public OpRewritePattern<LLVM::LLVMFuncOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::LLVMFuncOp func,
                                PatternRewriter &rewriter) const override {
    if (!func.getBody().empty())
      return failure();

    auto name = func.getSymName();

    bool isCheckpointingAttr = name.contains("__enzyme_set_checkpointing"),
         isMincutAttr = name.contains("__enzyme_set_mincut");

    if (!isCheckpointingAttr && !isMincutAttr)
      return failure();

    auto modOp = func->getParentOfType<ModuleOp>();
    auto uses = func.getSymbolUses(modOp);

    if (!uses) {
      rewriter.eraseOp(func);
      return success();
    }

    bool anyErased = false;

    for (auto use : *uses) {
      auto user = use.getUser();
      assert(isa<LLVM::CallOp>(user));

      bool enable = matchPattern(user->getOperand(0), m_One());

      Operation *loop = user->getParentOp();
      while (loop && !isa<scf::ForOp, scf::WhileOp>(loop)) {
        loop = loop->getParentOp();
      }

      if (loop && isa<scf::ForOp, scf::WhileOp>(loop)) {
        loop->setAttr(isCheckpointingAttr ? "enzyme_enable_checkpointing"
                                          : "enzyme_enable_mincut",
                      rewriter.getBoolAttr(enable));
      }

      rewriter.eraseOp(user);
      anyErased = true;
    }

    rewriter.eraseOp(func);

    return success();
  }
};

void CanonicalizeFor::runOnOperation() {
  mlir::RewritePatternSet rpl(getOperation()->getContext());
  populateSelectExtractPatterns(rpl);
  rpl.add<IfYieldMovementPattern, truncProp, ForOpInductionReplacement,
          RemoveUnusedForResults, RemoveUnusedArgs, MoveWhileToFor,
          RemoveWhileSelect, SelectTruncToTruncSelect, MaxSimplify,
          ForBoundUnSwitch, SelectI1Simplify, RemoveInductionVarRelated,
          RotateWhileAnd, MoveWhileDown, MoveWhileDown2,

          ReplaceRedundantArgs,

          WhileShiftToInduction,

          ForBreakAddUpgrade, RemoveUnusedResults,

          ForLoopApplyEnzymeAttributes,

          // MoveWhileDown3 Infinite loops on current kernel code, disabling
          // [and should fix] MoveWhileDown3,
          MoveWhileInvariantIfResult, WhileLogicalNegation, SubToAdd,
          WhileCmpOffset, RemoveUnusedCondVar, ReturnSq,
          MoveSideEffectFreeWhile>(getOperation()->getContext());
  //    WhileLICM,
  GreedyRewriteConfig config;
  config.setMaxIterations(247);
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(rpl),
                                          config))) {
    signalPassFailure();
  }
}
