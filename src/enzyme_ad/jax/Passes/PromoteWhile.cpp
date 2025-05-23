#include "mlir/Analysis/CallGraph.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <memory>


namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_PROMOTEWHILE
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

// Below code is from numba-mlir (https://github.com/numba/numba-mlir)
//
// SPDX-FileCopyrightText: 2021 - 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


namespace numba {
mlir::TypedAttr getConstVal(mlir::Operation *op) {
  assert(op);
  if (!op->hasTrait<mlir::OpTrait::ConstantLike>())
    return {};

  return op->getAttr("value").dyn_cast<mlir::TypedAttr>();
}

mlir::TypedAttr getConstVal(mlir::Value op) {
  assert(op);
  if (auto parentOp = op.getDefiningOp())
    return getConstVal(parentOp);

  return {};
}

mlir::TypedAttr getConstAttr(mlir::Type type, double val) {
  assert(type);
  if (type.isa<mlir::FloatType>())
    return mlir::FloatAttr::get(type, val);

  if (type.isa<mlir::IntegerType, mlir::IndexType>())
    return mlir::IntegerAttr::get(type, static_cast<int64_t>(val));

  return {};
}

int64_t getIntAttrValue(mlir::IntegerAttr attr) {
  assert(attr);
  auto attrType = attr.getType();
  if (attrType.isa<mlir::IndexType>())
    return attr.getInt();

  auto type = attrType.cast<mlir::IntegerType>();
  if (type.isSigned()) {
    return attr.getSInt();
  } else if (type.isUnsigned()) {
    return static_cast<int64_t>(attr.getUInt());
  } else {
    assert(type.isSignless());
    return attr.getInt();
  }
}
} // namespace numba

namespace {

static bool hasSideEffects(mlir::Operation *op) {
  assert(op);
  for (auto &region : op->getRegions()) {
    auto visitor = [](mlir::Operation *bodyOp) -> mlir::WalkResult {
      if (mlir::isa<mlir::scf::ReduceOp>(bodyOp) ||
          bodyOp->hasTrait<mlir::OpTrait::HasRecursiveMemoryEffects>() ||
          bodyOp->hasTrait<mlir::OpTrait::IsTerminator>())
        return mlir::WalkResult::advance();

      if (mlir::isa<mlir::CallOpInterface>(bodyOp))
        return mlir::WalkResult::interrupt();

      auto memEffects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(bodyOp);
      if (!memEffects || memEffects.hasEffect<mlir::MemoryEffects::Write>())
        return mlir::WalkResult::interrupt();

      return mlir::WalkResult::advance();
    };
    if (region.walk(visitor).wasInterrupted())
      return true;
  }
  return false;
}

static bool canParallelizeLoop(mlir::Operation *op, bool hasParallelAttr) {
  return hasParallelAttr || !hasSideEffects(op);
}

using CheckFunc = bool (*)(mlir::Operation *, mlir::Value);
using LowerFunc = mlir::Value (*)(mlir::OpBuilder &, mlir::Location,
                                  mlir::Value, mlir::Operation *);
using LowerReductionFunc = void (*)(mlir::OpBuilder &, mlir::Location,
                                    mlir::Value, mlir::Value,
                                    mlir::Operation *);

template <typename Op>
static bool simpleCheck(mlir::Operation *op, mlir::Value /*iterVar*/) {
  return mlir::isa<Op>(op);
}

template <typename Op>
static bool lhsArgCheck(mlir::Operation *op, mlir::Value iterVar) {
  auto casted = mlir::dyn_cast<Op>(op);
  if (!casted)
    return false;

  return casted.getLhs() == iterVar;
}

template <typename Op>
static mlir::Value simpleLower(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::Value val, mlir::Operation *origOp) {
  return val;
}

template <typename Op>
static void simpleBodyLower(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value lhs, mlir::Value rhs,
                            mlir::Operation *origOp) {
  auto casted = mlir::cast<Op>(origOp);
  mlir::IRMapping mapper;
  mapper.map(casted.getLhs(), lhs);
  mapper.map(casted.getRhs(), rhs);
  mlir::Value res = builder.clone(*origOp, mapper)->getResult(0);
  builder.create<mlir::scf::ReduceReturnOp>(loc, res);
}

template <typename SubOp, typename AddOp>
static mlir::Value subLower(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value val, mlir::Operation *origOp) {
  auto type = val.getType();
  auto zeroAttr = numba::getConstAttr(type, 0.0);
  auto zero = builder.create<mlir::arith::ConstantOp>(loc, type, zeroAttr);
  val = builder.create<SubOp>(loc, zero, val);
  return val;
}

template <typename SubOp, typename AddOp>
static void subBodyLower(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Value lhs, mlir::Value rhs,
                         mlir::Operation *origOp) {
  mlir::Value res = builder.create<AddOp>(loc, lhs, rhs);
  builder.create<mlir::scf::ReduceReturnOp>(loc, res);
}

template <typename DivOp, typename MulOp>
static mlir::Value divLower(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value val, mlir::Operation *origOp) {
  auto type = val.getType();
  auto oneAttr = numba::getConstAttr(type, 1.0);
  auto one = builder.create<mlir::arith::ConstantOp>(loc, type, oneAttr);
  val = builder.create<DivOp>(loc, one, val);
  return val;
}

template <typename DivOp, typename MulOp>
static void divBodyLower(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Value lhs, mlir::Value rhs,
                         mlir::Operation *origOp) {
  mlir::Value res = builder.create<MulOp>(loc, lhs, rhs);
  builder.create<mlir::scf::ReduceReturnOp>(loc, res);
}

template <typename Op>
static constexpr std::tuple<CheckFunc, LowerFunc, LowerReductionFunc>
getSimpleHandler() {
  return {&simpleCheck<Op>, &simpleLower<Op>, &simpleBodyLower<Op>};
}

namespace arith = mlir::arith;
static const constexpr std::tuple<CheckFunc, LowerFunc, LowerReductionFunc>
    promoteHandlers[] = {
        // clang-format off
    getSimpleHandler<arith::AddIOp>(),
    getSimpleHandler<arith::AddFOp>(),

    getSimpleHandler<arith::MulIOp>(),
    getSimpleHandler<arith::MulFOp>(),

    getSimpleHandler<arith::MinSIOp>(),
    getSimpleHandler<arith::MinUIOp>(),
    getSimpleHandler<arith::MinimumFOp>(),

    getSimpleHandler<arith::MaxSIOp>(),
    getSimpleHandler<arith::MaxUIOp>(),
    getSimpleHandler<arith::MaximumFOp>(),

    {&lhsArgCheck<arith::SubIOp>, &subLower<arith::SubIOp, arith::AddIOp>, &subBodyLower<arith::SubIOp, arith::AddIOp>},
    {&lhsArgCheck<arith::SubFOp>, &subLower<arith::SubFOp, arith::AddFOp>, &subBodyLower<arith::SubFOp, arith::AddFOp>},

    {&lhsArgCheck<arith::DivFOp>, &divLower<arith::DivFOp, arith::MulFOp>, &divBodyLower<arith::DivFOp, arith::MulFOp>},
        // clang-format on
};

static std::pair<LowerFunc, LowerReductionFunc>
getLowerer(mlir::Operation *op, mlir::Value iterVar) {
  assert(op);
  for (auto &&[checker, lowerer, bodyLowerer] : promoteHandlers)
    if (checker(op, iterVar))
      return {lowerer, bodyLowerer};

  return {nullptr, nullptr};
}

static bool isInsideParallelRegion(mlir::Operation *op) {
  return false;
  // assert(op && "Invalid op");
  // while (true) {
  //   auto region = op->getParentOfType<numba::util::EnvironmentRegionOp>();
  //   if (!region)
  //     return false;

  //   if (mlir::isa<numba::util::ParallelAttr>(region.getEnvironment()))
  //     return true;

  //   op = region;
  // }
}

static bool checkIndexType(mlir::arith::CmpIOp op) {
  auto type = op.getLhs().getType();
  if (mlir::isa<mlir::IndexType>(type))
    return true;

  if (type.isSignlessInteger())
    return true;

  return false;
}
static bool canMoveOpToBefore(mlir::Operation *op) {
  if (op->getNumResults() != 1)
    return false;

  return mlir::isPure(op);
}
static std::optional<llvm::SmallVector<unsigned>>
getArgsMapping(mlir::ValueRange args1, mlir::ValueRange args2) {
  if (args1.size() != args2.size())
    return std::nullopt;

  llvm::SmallVector<unsigned> ret(args1.size());
  for (auto &&[i, arg1] : llvm::enumerate(args1)) {
    auto it = llvm::find(args2, arg1);
    if (it == args2.end())
      return std::nullopt;

    auto j = it - args2.begin();
    ret[j] = static_cast<unsigned>(i);
  }

  return ret;
}
struct WhileOpAlignBeforeArgs
    : public mlir::OpRewritePattern<mlir::scf::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::WhileOp loop,
                  mlir::PatternRewriter &rewriter) const override {
    auto oldBefore = loop.getBeforeBody();
    auto oldTerm =
        mlir::cast<mlir::scf::ConditionOp>(oldBefore->getTerminator());
    mlir::ValueRange beforeArgs = oldBefore->getArguments();
    mlir::ValueRange termArgs = oldTerm.getArgs();
    if (beforeArgs == termArgs)
      return mlir::failure();

    auto mapping = getArgsMapping(beforeArgs, termArgs);
    if (!mapping)
      return mlir::failure();

    {
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(oldTerm);
      rewriter.replaceOpWithNewOp<mlir::scf::ConditionOp>(
          oldTerm, oldTerm.getCondition(), beforeArgs);
    }

    auto oldAfter = loop.getAfterBody();

    llvm::SmallVector<mlir::Type> newResultTypes(beforeArgs.size());
    for (auto &&[i, j] : llvm::enumerate(*mapping))
      newResultTypes[j] = loop.getResult(i).getType();

    auto newLoop = rewriter.create<mlir::scf::WhileOp>(
        loop.getLoc(), newResultTypes, loop.getInits(), nullptr, nullptr);
    auto newBefore = newLoop.getBeforeBody();
    auto newAfter = newLoop.getAfterBody();

    llvm::SmallVector<mlir::Value> newResults(beforeArgs.size());
    llvm::SmallVector<mlir::Value> newAfterArgs(beforeArgs.size());
    for (auto &&[i, j] : llvm::enumerate(*mapping)) {
      newResults[i] = newLoop.getResult(j);
      newAfterArgs[i] = newAfter->getArgument(j);
    }

    rewriter.inlineBlockBefore(oldBefore, newBefore, newBefore->begin(),
                               newBefore->getArguments());
    rewriter.inlineBlockBefore(oldAfter, newAfter, newAfter->begin(),
                               newAfterArgs);

    rewriter.replaceOp(loop, newResults);
    return mlir::success();
  }
};

bool isOppositeImpl(arith::CmpIOp c1, arith::CmpIOp c2) {
  bool sameOperands = c1.getLhs() == c2.getLhs() && c1.getRhs() == c2.getRhs();
  bool sameOperandsSwapped =
      c1.getLhs() == c2.getRhs() && c1.getRhs() == c2.getLhs();
  if (!(sameOperands || sameOperandsSwapped))
    return false;

  if (c1.getPredicate() == arith::CmpIPredicate::eq &&
      c2.getPredicate() == arith::CmpIPredicate::ne)
    return true;

  return false;
}

bool isOpposite(arith::CmpIOp c1, arith::CmpIOp c2) {
  return isOppositeImpl(c1, c2) || isOppositeImpl(c2, c1);
}

/// scf.while () {
///   %cond1 = eq %1 %2
///   %cond2 = ne %1 %2
///   scf.if (%cond1) {A} else {B}
///   scf.condition %cond2
///
/// ->
///
/// scf.while () {
///   %cond1 = eq %1 %2
///   %cond2 = ne %1 %2
///   scf.if (%cond2) {B} else {A}
///   scf.condition %cond2
struct WhileOpPrepIf : public mlir::OpRewritePattern<mlir::scf::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::IfOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto loop = mlir::dyn_cast<mlir::scf::WhileOp>(op->getParentOp());
    if (!loop || op->getBlock() != loop.getBeforeBody())
      return mlir::failure();

    mlir::Block *beforeBody = loop.getBeforeBody();
    auto beforeTerm =
        mlir::cast<mlir::scf::ConditionOp>(beforeBody->getTerminator());
    if (op.getCondition() == beforeTerm.getCondition())
      return mlir::failure();

    auto c1 = mlir::dyn_cast_or_null<arith::CmpIOp>(
        op.getCondition().getDefiningOp());
    auto c2 = mlir::dyn_cast_or_null<arith::CmpIOp>(
        beforeTerm.getCondition().getDefiningOp());

    if (!c1 || !c2)
      return mlir::failure();

    if (!isOpposite(c1, c2))
      return mlir::failure();

    auto newIf = rewriter.create<scf::IfOp>(op.getLoc(), op->getResultTypes(),
                                            beforeTerm.getCondition());
    rewriter.inlineRegionBefore(op.getElseRegion(), newIf.getThenRegion(),
                                newIf.getThenRegion().begin());
    rewriter.inlineRegionBefore(op.getThenRegion(), newIf.getElseRegion(),
                                newIf.getElseRegion().begin());
    rewriter.replaceOp(op, newIf);

    return success();
  }
};

struct WhileOpMoveIfCond : public mlir::OpRewritePattern<mlir::scf::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::IfOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto loop = mlir::dyn_cast<mlir::scf::WhileOp>(op->getParentOp());
    if (!loop || op->getBlock() != loop.getBeforeBody())
      return mlir::failure();

    mlir::Block *beforeBody = loop.getBeforeBody();
    auto beforeTerm =
        mlir::cast<mlir::scf::ConditionOp>(beforeBody->getTerminator());
    if (op.getCondition() != beforeTerm.getCondition())
      return mlir::failure();

    for (auto result : op.getResults())
      for (auto user : result.getUsers())
        if (user != beforeTerm)
          return mlir::failure();

    for (auto &nextOp : llvm::make_range(std::next(op->getIterator()),
                                         beforeTerm->getIterator()))
      if (!mlir::isPure(&nextOp))
        return mlir::failure();

    mlir::DominanceInfo dom;
    llvm::SmallSetVector<mlir::Value, 8> capturedValues;
    for (auto body : {op.thenBlock(), op.elseBlock()}) {
      if (!body)
        continue;

      body->walk([&](mlir::Operation *blockOp) {
        for (auto arg : blockOp->getOperands()) {
          if (dom.properlyDominates(arg, loop) ||
              !dom.properlyDominates(arg, op))
            continue;

          capturedValues.insert(arg);
        }
      });
    }

    auto newResTypes = llvm::to_vector(loop.getResultTypes());
    llvm::append_range(
        newResTypes, mlir::ValueRange(capturedValues.getArrayRef()).getTypes());

    mlir::OpBuilder::InsertionGuard g(rewriter);

    auto newBeforeArgs = llvm::to_vector(beforeTerm.getArgs());
    llvm::append_range(newBeforeArgs, capturedValues);
    rewriter.setInsertionPoint(beforeTerm);
    auto newTerm = rewriter.replaceOpWithNewOp<mlir::scf::ConditionOp>(
        beforeTerm, beforeTerm.getCondition(), newBeforeArgs);

    rewriter.setInsertionPoint(loop);
    auto newLoop = rewriter.create<mlir::scf::WhileOp>(
        loop.getLoc(), newResTypes, loop.getInits(), nullptr, nullptr);

    auto newAfter = newLoop.getAfterBody();
    mlir::ValueRange newAfterArgs = newAfter->getArguments();

    {
      auto replaceChecker = [&](mlir::OpOperand &operand) -> bool {
        auto owner = operand.getOwner();
        return op.getThenRegion().isAncestor(owner->getParentRegion());
      };
      rewriter.replaceUsesWithIf(capturedValues.getArrayRef(),
                                 newAfterArgs.take_back(capturedValues.size()),
                                 replaceChecker);
    }
    if (op.elseBlock()) {
      auto replaceChecker = [&](mlir::OpOperand &operand) -> bool {
        auto owner = operand.getOwner();
        return op.getElseRegion().isAncestor(owner->getParentRegion());
      };
      rewriter.replaceUsesWithIf(
          capturedValues.getArrayRef(),
          newLoop.getResults().take_back(capturedValues.size()),
          replaceChecker);
    }

    auto newBefore = newLoop.getBeforeBody();

    rewriter.inlineBlockBefore(beforeBody, newBefore, newBefore->begin(),
                               newBefore->getArguments());

    auto afterMapping =
        llvm::to_vector(newAfterArgs.drop_back(capturedValues.size()));

    auto thenYield = op.thenYield();
    for (auto &&[res, yieldArg] :
         llvm::zip(op.getResults(), thenYield.getResults())) {
      for (auto &use : res.getUses()) {
        assert(use.getOwner() == newTerm && "Invalid user");
        afterMapping[use.getOperandNumber() - 1] = yieldArg;
      }
    }
    rewriter.eraseOp(thenYield);

    rewriter.inlineBlockBefore(op.thenBlock(), newAfter, newAfter->end());

    auto afterBody = loop.getAfterBody();
    rewriter.inlineBlockBefore(afterBody, newAfter, newAfter->end(),
                               afterMapping);

    afterMapping.clear();
    llvm::append_range(afterMapping,
                       newLoop.getResults().drop_back(capturedValues.size()));

    if (op.elseBlock()) {
      auto elseYield = op.elseYield();
      for (auto &&[res, yieldArg] :
           llvm::zip(op.getResults(), elseYield.getResults())) {
        for (auto &use : res.getUses()) {
          assert(use.getOwner() == newTerm && "Invalid user");
          afterMapping[use.getOperandNumber() - 1] = yieldArg;
        }
      }
      rewriter.eraseOp(elseYield);

      rewriter.inlineBlockBefore(op.elseBlock(), newLoop->getBlock(),
                                 std::next(newLoop->getIterator()));
    }
    rewriter.replaceOp(loop, afterMapping);

    auto termLoc = newTerm.getLoc();
    rewriter.setInsertionPoint(newTerm);
    for (auto res : op.getResults()) {
      mlir::Value newRes =
          rewriter.create<mlir::ub::PoisonOp>(termLoc, res.getType(), nullptr);
      rewriter.replaceAllUsesWith(res, newRes);
    }

    rewriter.eraseOp(op);
    return mlir::success();
  }
};
struct WhileOpLICM : public mlir::OpRewritePattern<mlir::scf::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::WhileOp loop,
                  mlir::PatternRewriter &rewriter) const override {
    bool changed = false;

    mlir::DominanceInfo dom;
    for (mlir::Block *body : {loop.getBeforeBody(), loop.getAfterBody()}) {
      for (mlir::Operation &op :
           llvm::make_early_inc_range(body->without_terminator())) {
        if (!mlir::isPure(&op))
          continue;

        if (llvm::any_of(op.getOperands(), [&](auto &&arg) {
              return !dom.properlyDominates(arg, loop);
            }))
          continue;

        rewriter.modifyOpInPlace(&op, [&]() { op.moveBefore(loop); });
        changed = true;
      }
    }
    return mlir::success(changed);
  }
};
struct MoveOpsFromBefore : public mlir::OpRewritePattern<mlir::scf::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::WhileOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto oldBefore = op.getBeforeBody();
    auto oldAfter = op.getAfterBody();
    auto oldTerm =
        mlir::cast<mlir::scf::ConditionOp>(oldBefore->getTerminator());

    mlir::Operation *opToMove = nullptr;
    size_t idx = 0;
    for (auto &&[i, args] : llvm::enumerate(llvm::zip(
             oldTerm.getArgs(), oldAfter->getArguments(), op.getResults()))) {
      auto &&[arg, afterArg, res] = args;
      if (afterArg.use_empty() && res.use_empty())
        continue;

      auto argOp = arg.getDefiningOp();
      if (argOp && canMoveOpToBefore(argOp)) {
        opToMove = argOp;
        idx = i;
        break;
      }
    }

    if (!opToMove)
      return rewriter.notifyMatchFailure(op, "No ops to move");

    mlir::OpBuilder::InsertionGuard g(rewriter);

    auto newResults = llvm::to_vector(op->getResultTypes());
    llvm::append_range(newResults, opToMove->getOperandTypes());

    auto newTermArgs = llvm::to_vector(oldTerm.getArgs());
    llvm::append_range(newTermArgs, opToMove->getOperands());

    rewriter.setInsertionPoint(oldTerm);
    rewriter.replaceOpWithNewOp<mlir::scf::ConditionOp>(
        oldTerm, oldTerm.getCondition(), newTermArgs);

    rewriter.setInsertionPoint(op);
    auto newLoop = rewriter.create<mlir::scf::WhileOp>(
        op.getLoc(), newResults, op.getInits(), nullptr, nullptr);

    auto newBefore = newLoop.getBeforeBody();
    auto newAfter = newLoop.getAfterBody();

    auto numArgs = opToMove->getNumOperands();
    auto newAfterArgs = newAfter->getArguments();
    rewriter.inlineBlockBefore(oldBefore, newBefore, newBefore->begin(),
                               newBefore->getArguments());
    rewriter.inlineBlockBefore(oldAfter, newAfter, newAfter->begin(),
                               newAfterArgs.drop_back(numArgs));

    mlir::IRMapping mapping;
    mapping.map(opToMove->getOperands(), newAfterArgs.take_back(numArgs));

    rewriter.setInsertionPointToStart(newAfter);
    auto newOp = rewriter.clone(*opToMove, mapping);
    rewriter.replaceAllUsesWith(newAfterArgs[idx], newOp->getResult(0));

    mapping.map(opToMove->getOperands(),
                newLoop.getResults().take_back(numArgs));

    rewriter.setInsertionPointAfter(newLoop);
    newOp = rewriter.clone(*opToMove, mapping);
    rewriter.replaceAllUsesWith(op.getResult(idx), newOp->getResult(0));

    rewriter.replaceOp(op, newLoop.getResults().drop_back(numArgs));
    return mlir::success();
  }
};
struct CanonicalizeLoopMemrefIndex
    : public mlir::OpRewritePattern<mlir::memref::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::LoadOp loadOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto loop = mlir::dyn_cast<mlir::scf::WhileOp>(loadOp->getParentOp());
    if (!loop || loadOp->getBlock() != loop.getBeforeBody())
      return rewriter.notifyMatchFailure(loadOp, "Not inside the loop");

    auto memref = loadOp.getMemref();
    if (!mlir::isa_and_present<mlir::memref::AllocOp, mlir::memref::AllocaOp>(
            memref.getDefiningOp()))
      return rewriter.notifyMatchFailure(loadOp, "Not result of alloc");

    auto isAncestor = [&](mlir::Operation *op) -> bool {
      auto reg = op->getParentRegion();
      return loop.getBefore().isAncestor(reg) ||
             loop.getAfter().isAncestor(reg);
    };

    mlir::memref::StoreOp storeOp;
    for (auto user : memref.getUsers()) {
      if (user == loadOp)
        continue;

      if (mlir::isa<mlir::memref::DeallocOp>(user))
        continue;

      if (mlir::isa<mlir::memref::LoadOp>(user)) {
        if (isAncestor(user)) {
          return rewriter.notifyMatchFailure(
              loadOp, [&](mlir::Diagnostic &diag) {
                diag << "Unsupported nested load: " << *user;
              });
        } else {
          continue;
        }
      }

      if (auto op = mlir::dyn_cast<mlir::memref::StoreOp>(user)) {
        if (op->getBlock() == loop.getBeforeBody()) {
          if (storeOp) {
            return rewriter.notifyMatchFailure(
                loadOp, [&](mlir::Diagnostic &diag) {
                  diag << "Unsupported Multiple stores: " << *storeOp << " and "
                       << *op;
                });
          } else {
            storeOp = op;
            continue;
          }
        } else {
          if (isAncestor(user)) {
            return rewriter.notifyMatchFailure(
                loadOp, [&](mlir::Diagnostic &diag) {
                  diag << "Unsupported nested store: " << *user;
                });
          } else {
            continue;
          }
        }
      }

      return rewriter.notifyMatchFailure(loadOp, [&](mlir::Diagnostic &diag) {
        diag << "Unsupported user: " << *user;
      });
    }

    if (!storeOp || storeOp.getIndices() != loadOp.getIndices())
      return rewriter.notifyMatchFailure(loadOp, "invalid store op");

    mlir::DominanceInfo dom;
    if (!dom.properlyDominates(loadOp.getOperation(), storeOp.getOperation()))
      return rewriter.notifyMatchFailure(loadOp,
                                         "Store op doesn't dominate load");

    auto indices = storeOp.getIndices();
    for (auto idx : indices) {
      if (!dom.properlyDominates(idx, loop))
        return rewriter.notifyMatchFailure(loadOp, [&](mlir::Diagnostic &diag) {
          diag << "Index doesnt dominate the loop: " << idx;
        });
    }

    mlir::OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(loop);
    auto loc = loop.getLoc();
    mlir::Value init =
        rewriter.create<mlir::memref::LoadOp>(loc, memref, indices);

    auto newInits = llvm::to_vector(loop.getInits());
    newInits.emplace_back(init);

    auto newResults = llvm::to_vector(loop->getResultTypes());
    newResults.emplace_back(init.getType());
    auto newLoop = rewriter.create<mlir::scf::WhileOp>(
        loc, newResults, newInits, nullptr, nullptr);

    auto oldBefore = loop.getBeforeBody();
    auto oldAfter = loop.getAfterBody();
    auto newBefore = newLoop.getBeforeBody();
    auto newAfter = newLoop.getAfterBody();

    rewriter.inlineBlockBefore(oldBefore, newBefore, newBefore->begin(),
                               newBefore->getArguments().drop_back());
    rewriter.inlineBlockBefore(oldAfter, newAfter, newAfter->begin(),
                               newAfter->getArguments().drop_back());

    auto beforeTerm =
        mlir::cast<mlir::scf::ConditionOp>(newBefore->getTerminator());
    rewriter.setInsertionPoint(beforeTerm);
    auto newCondArgs = llvm::to_vector(beforeTerm.getArgs());
    newCondArgs.emplace_back(storeOp.getValueToStore());
    rewriter.replaceOpWithNewOp<mlir::scf::ConditionOp>(
        beforeTerm, beforeTerm.getCondition(), newCondArgs);

    rewriter.eraseOp(storeOp);
    rewriter.replaceOp(loadOp, newBefore->getArguments().back());

    auto afterTerm = mlir::cast<mlir::scf::YieldOp>(newAfter->getTerminator());
    rewriter.setInsertionPoint(afterTerm);
    auto newYieldArgs = llvm::to_vector(afterTerm.getResults());
    newYieldArgs.emplace_back(newAfter->getArguments().back());
    rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(afterTerm, newYieldArgs);

    rewriter.setInsertionPointAfter(newLoop);
    rewriter.create<mlir::memref::StoreOp>(loc, newLoop.getResults().back(),
                                           memref, indices);
    rewriter.replaceOp(loop, newLoop.getResults().drop_back());
    return mlir::success();
  }
};
struct PromoteWhileOp : public mlir::OpRewritePattern<mlir::scf::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::WhileOp loop,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Block *beforeBody = loop.getBeforeBody();
    if (!llvm::hasSingleElement(beforeBody->without_terminator()))
      return rewriter.notifyMatchFailure(loop, "Loop body must have single op");

    auto cmp = mlir::dyn_cast<mlir::arith::CmpIOp>(beforeBody->front());
    if (!cmp)
      return rewriter.notifyMatchFailure(loop,
                                         "Loop body must have single cmp op");

    auto beforeTerm =
        mlir::cast<mlir::scf::ConditionOp>(beforeBody->getTerminator());
    if (!llvm::hasSingleElement(cmp->getUses()) &&
        beforeTerm.getCondition() == cmp.getResult())
      return rewriter.notifyMatchFailure(loop, [&](mlir::Diagnostic &diag) {
        diag << "Expected single condiditon use: " << *cmp;
      });

    if (mlir::ValueRange(beforeBody->getArguments()) != beforeTerm.getArgs())
      return rewriter.notifyMatchFailure(loop, "Invalid args order");

    using Pred = mlir::arith::CmpIPredicate;
    auto predicate = cmp.getPredicate();
    if (predicate != Pred::slt && predicate != Pred::sgt &&
        predicate != Pred::ult && predicate != Pred::ugt &&
        predicate != Pred::ne)
      return rewriter.notifyMatchFailure(loop, [&](mlir::Diagnostic &diag) {
        diag << "Expected 'lt', 'gt', or 'ne' predicate: " << *cmp;
      });

    if (!checkIndexType(cmp))
      return rewriter.notifyMatchFailure(loop, [&](mlir::Diagnostic &diag) {
        diag << "Expected index like type: " << *cmp;
      });

    mlir::BlockArgument iterVar;
    mlir::Value end;
    mlir::DominanceInfo dom;
    for (auto &[gt, lt] : {std::make_pair(Pred::sgt, Pred::slt),
                           std::make_pair(Pred::ugt, Pred::ult)}) {
      for (bool reverse : {false, true}) {
        auto expectedPred = reverse ? gt : lt;
        if (cmp.getPredicate() != expectedPred)
          continue;

        auto arg1 = reverse ? cmp.getRhs() : cmp.getLhs();
        auto arg2 = reverse ? cmp.getLhs() : cmp.getRhs();
        if (!mlir::isa<mlir::BlockArgument>(arg1))
          continue;

        if (!dom.properlyDominates(arg2, loop))
          continue;

        iterVar = mlir::cast<mlir::BlockArgument>(arg1);
        end = arg2;
      }
      if (iterVar)
        break;
    }

    if (!iterVar && cmp.getPredicate() == Pred::ne) {
      auto lhs = cmp.getLhs();
      auto rhs = cmp.getRhs();
      auto blhs = mlir::dyn_cast<mlir::BlockArgument>(lhs);
      auto brhs = mlir::dyn_cast<mlir::BlockArgument>(rhs);

      if (blhs && blhs.getOwner() == beforeBody &&
          dom.properlyDominates(rhs, loop)) {
        iterVar = blhs;
        end = rhs;
      } else if (brhs && brhs.getOwner() == beforeBody &&
                 dom.properlyDominates(lhs, loop)) {
        iterVar = brhs;
        end = lhs;
      }
    }

    if (!iterVar)
      return rewriter.notifyMatchFailure(loop, [&](mlir::Diagnostic &diag) {
        diag << "Unrecognized cmp form: " << *cmp;
      });

    if (!llvm::hasNItems(iterVar.getUses(), 2))
      return rewriter.notifyMatchFailure(loop, [&](mlir::Diagnostic &diag) {
        diag << "Unrecognized iter var: " << iterVar;
      });

    mlir::Block *afterBody = loop.getAfterBody();
    auto afterTerm = mlir::cast<mlir::scf::YieldOp>(afterBody->getTerminator());
    auto argNumber = iterVar.getArgNumber();
    auto afterTermIterArg = afterTerm.getResults()[argNumber];

    auto iterVarAfter = afterBody->getArgument(argNumber);

    mlir::Value step;
    for (auto &use : iterVarAfter.getUses()) {
      auto owner = mlir::dyn_cast<mlir::arith::AddIOp>(use.getOwner());
      if (!owner)
        continue;

      auto other =
          (iterVarAfter == owner.getLhs() ? owner.getRhs() : owner.getLhs());
      if (!dom.properlyDominates(other, loop))
        continue;

      if (afterTermIterArg != owner.getResult())
        continue;

      step = other;
      break;
    }

    if (!step)
      return rewriter.notifyMatchFailure(loop,
                                         "Didn't found suitable 'add' op");

    auto begin = loop.getInits()[argNumber];

    auto loc = loop.getLoc();
    auto indexType = rewriter.getIndexType();
    assert(begin.getType() == end.getType() &&
           begin.getType() == step.getType());

    llvm::SmallVector<mlir::Value> mapping;
    mapping.reserve(loop.getInits().size());
    for (auto &&[i, init] : llvm::enumerate(loop.getInits())) {
      if (i == argNumber)
        continue;

      mapping.emplace_back(init);
    }

    auto emptyBuidler = [](mlir::OpBuilder &, mlir::Location, mlir::Value,
                           mlir::ValueRange) {};
    auto newLoop = rewriter.create<mlir::scf::ForOp>(loc, begin, end, step,
                                                     mapping, emptyBuidler);

    mlir::Block *newBody = newLoop.getBody();

    mlir::OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(newBody);
    mlir::Value newIterVar = newBody->getArgument(0);
    if (newIterVar.getType() != iterVar.getType())
      newIterVar = rewriter.create<mlir::arith::IndexCastOp>(
          loc, iterVar.getType(), newIterVar);

    mapping.clear();
    auto newArgs = newBody->getArguments();
    for (auto i : llvm::seq<size_t>(0, newArgs.size())) {
      if (i < argNumber) {
        mapping.emplace_back(newArgs[i + 1]);
      } else if (i == argNumber) {
        mapping.emplace_back(newArgs.front());
      } else {
        mapping.emplace_back(newArgs[i]);
      }
    }

    rewriter.inlineBlockBefore(loop.getAfterBody(), newBody, newBody->end(),
                               mapping);

    auto term = mlir::cast<mlir::scf::YieldOp>(newBody->getTerminator());

    mapping.clear();
    for (auto &&[i, arg] : llvm::enumerate(term.getResults())) {
      if (i == argNumber)
        continue;

      mapping.emplace_back(arg);
    }

    rewriter.setInsertionPoint(term);
    rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(term, mapping);

    rewriter.setInsertionPointAfter(newLoop);
    mlir::Value one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    mlir::Value stepDec = rewriter.create<mlir::arith::SubIOp>(loc, step, one);
    mlir::Value len = rewriter.create<mlir::arith::SubIOp>(loc, end, begin);
    len = rewriter.create<mlir::arith::AddIOp>(loc, len, stepDec);
    len = rewriter.create<mlir::arith::DivSIOp>(loc, len, step);
    len = rewriter.create<mlir::arith::SubIOp>(loc, len, one);
    mlir::Value res = rewriter.create<mlir::arith::MulIOp>(loc, len, step);
    res = rewriter.create<mlir::arith::AddIOp>(loc, begin, res);
    if (res.getType() != iterVar.getType())
      res = rewriter.create<mlir::arith::IndexCastOp>(loc, iterVar.getType(),
                                                      res);

    mapping.clear();
    llvm::append_range(mapping, newLoop.getResults());
    mapping.insert(mapping.begin() + argNumber, res);
    rewriter.replaceOp(loop, mapping);
    return mlir::success();
  }
};

struct PromoteToParallel : public mlir::OpRewritePattern<mlir::scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::ForOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op.getLowerBound().getType().isIndex())
      return mlir::failure();

    if (!canParallelizeLoop(op, isInsideParallelRegion(op)))
      return mlir::failure();

    mlir::Block *loopBody = op.getBody();
    auto term = mlir::cast<mlir::scf::YieldOp>(loopBody->getTerminator());
    auto iterVars = op.getRegionIterArgs();
    assert(iterVars.size() == term.getResults().size());

    using ReductionDesc = std::tuple<mlir::Operation *, LowerFunc,
                                     LowerReductionFunc, mlir::Value>;
    llvm::SmallVector<ReductionDesc> reductionOps;
    llvm::SmallDenseSet<mlir::Operation *> reductionOpsSet;
    for (auto &&[iterVar, result] : llvm::zip(iterVars, term.getResults())) {
      auto reductionOp = result.getDefiningOp();
      if (!reductionOp || reductionOp->getNumResults() != 1 ||
          reductionOp->getNumOperands() != 2 ||
          !llvm::hasSingleElement(reductionOp->getUses()))
        return mlir::failure();

      mlir::Value reductionArg;
      if (reductionOp->getOperand(0) == iterVar) {
        reductionArg = reductionOp->getOperand(1);
      } else if (reductionOp->getOperand(1) == iterVar) {
        reductionArg = reductionOp->getOperand(0);
      } else {
        return mlir::failure();
      }

      auto &&[lowerer, bodyLowerer] = getLowerer(reductionOp, iterVar);
      if (!lowerer)
        return mlir::failure();

      reductionOps.emplace_back(reductionOp, lowerer, bodyLowerer,
                                reductionArg);
      reductionOpsSet.insert(reductionOp);
    }

    for (auto iterVar : iterVars)
      for (auto user : iterVar.getUsers())
        if (reductionOpsSet.count(user) == 0)
          return mlir::failure();

    auto bodyBuilder = [&](mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::ValueRange iterVals, mlir::ValueRange) {
      assert(1 == iterVals.size());
      mlir::IRMapping mapping;
      mapping.map(op.getInductionVar(), iterVals.front());
      for (auto &oldOp : loopBody->without_terminator())
        if (0 == reductionOpsSet.count(&oldOp))
          builder.clone(oldOp, mapping);

      llvm::SmallVector<mlir::Value> redArgs;
      for (auto &&[reductionOp, lowerer, bodyLowerer, reductionArg] :
           reductionOps) {
        auto arg = mapping.lookupOrDefault(reductionArg);
        auto res = lowerer(builder, loc, arg, reductionOp);
        ;
        redArgs.emplace_back(res);
      }

      auto reduceOp = builder.create<mlir::scf::ReduceOp>(loc, redArgs);

      mlir::OpBuilder::InsertionGuard g(builder);
      for (auto &&[region, it] :
           llvm::zip(reduceOp.getReductions(), reductionOps)) {
        auto &&[reductionOp, lowerer, bodyLowerer, reductionArg] = it;
        mlir::Block &body = region.front();
        assert(body.getNumArguments() == 2);
        builder.setInsertionPointToStart(&body);
        bodyLowerer(builder, loc, body.getArgument(0), body.getArgument(1),
                    reductionOp);
      }
    };

    rewriter.replaceOpWithNewOp<mlir::scf::ParallelOp>(
        op, op.getLowerBound(), op.getUpperBound(), op.getStep(),
        op.getInitArgs(), bodyBuilder);

    return mlir::success();
  }
};

struct MergeNestedForIntoParallel
    : public mlir::OpRewritePattern<mlir::scf::ParallelOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::ParallelOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto parent = mlir::dyn_cast<mlir::scf::ForOp>(op->getParentOp());
    if (!parent)
      return mlir::failure();

    if (!parent.getLowerBound().getType().isIndex())
      return mlir::failure();

    auto block = parent.getBody();
    if (!llvm::hasSingleElement(block->without_terminator()))
      return mlir::failure();

    if (parent.getInitArgs().size() != op.getInitVals().size())
      return mlir::failure();

    auto yield = mlir::cast<mlir::scf::YieldOp>(block->getTerminator());
    assert(yield.getNumOperands() == op.getNumResults());
    for (auto &&[arg, initVal, result, yieldOp] :
         llvm::zip(block->getArguments().drop_front(), op.getInitVals(),
                   op.getResults(), yield.getOperands())) {
      if (!arg.hasOneUse() || arg != initVal || result != yieldOp)
        return mlir::failure();
    }
    auto checkVals = [&](auto vals) {
      for (auto val : vals)
        if (val.getParentBlock() == block)
          return true;

      return false;
    };
    if (checkVals(op.getLowerBound()) || checkVals(op.getUpperBound()) ||
        checkVals(op.getStep()))
      return mlir::failure();

    if (!canParallelizeLoop(op, isInsideParallelRegion(op)))
      return mlir::failure();

    auto makeValueList = [](auto op, auto ops) {
      llvm::SmallVector<mlir::Value> ret;
      ret.reserve(ops.size() + 1);
      ret.emplace_back(op);
      ret.append(ops.begin(), ops.end());
      return ret;
    };

    auto lowerBounds =
        makeValueList(parent.getLowerBound(), op.getLowerBound());
    auto upperBounds =
        makeValueList(parent.getUpperBound(), op.getUpperBound());
    auto steps = makeValueList(parent.getStep(), op.getStep());

    auto oldBody = op.getBody();
    auto bodyBuilder = [&](mlir::OpBuilder &builder, mlir::Location /*loc*/,
                           mlir::ValueRange iter_vals, mlir::ValueRange temp) {
      assert(iter_vals.size() == lowerBounds.size());
      assert(temp.empty());
      mlir::IRMapping mapping;
      assert((oldBody->getNumArguments() + 1) == iter_vals.size());
      mapping.map(block->getArgument(0), iter_vals.front());
      mapping.map(oldBody->getArguments(), iter_vals.drop_front());
      for (auto &op : *oldBody)
        builder.clone(op, mapping);
    };

    rewriter.setInsertionPoint(parent);
    rewriter.replaceOpWithNewOp<mlir::scf::ParallelOp>(
        parent, lowerBounds, upperBounds, steps, parent.getInitArgs(),
        bodyBuilder);

    return mlir::success();
  }
};

void populateLoopOptsPatterns(mlir::RewritePatternSet &patterns) {
  patterns.insert<CanonicalizeLoopMemrefIndex, MoveOpsFromBefore, WhileOpLICM,
                  WhileOpPrepIf,
                  /*WhileOpExpandTuple,*/ WhileOpMoveIfCond,
                  WhileOpAlignBeforeArgs>(patterns.getContext());
}
void populatePromoteWhilePatterns(mlir::RewritePatternSet &patterns) {
  patterns.insert<PromoteWhileOp>(patterns.getContext());
}

struct PromoteWhilePass
    : public enzyme::impl::PromoteWhileBase<PromoteWhilePass> {
  using Base::Base;

  void runOnOperation() override {
    auto context = &getContext();

    mlir::RewritePatternSet patterns(context);
    populatePromoteWhilePatterns(patterns);
    populateLoopOptsPatterns(patterns);
    mlir::scf::WhileOp::getCanonicalizationPatterns(patterns, context);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                        std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

