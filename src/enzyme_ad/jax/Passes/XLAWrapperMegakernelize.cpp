//===- XLAWrapperMegakernelize.cpp - Fuse raised XLA wrappers ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "Enzyme/MLIR/Analysis/DataFlowAliasAnalysis.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"

#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/BitVector.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_XLAWRAPPERMEGAKERNELIZEPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;

namespace {

/// Return an identity suitable for matching the buffers passed to raised XLA
/// wrappers. Restrict this to casts which preserve the exact viewed storage;
/// stripping subviews would incorrectly identify different offsets as the same
/// XLA argument.
static Value getBufferIdentity(Value value) {
  while (Operation *definingOp = value.getDefiningOp()) {
    if (auto pointerToMemref =
            dyn_cast<enzymexla::Pointer2MemrefOp>(definingOp)) {
      value = pointerToMemref.getSource();
      continue;
    }
    if (auto cast = dyn_cast<memref::CastOp>(definingOp)) {
      value = cast.getSource();
      continue;
    }
    break;
  }
  return value;
}

static bool isRaisedWrapperFunction(func::FuncOp function,
                                    enzymexla::XLAWrapperOp wrapper) {
  if (function.isExternal() || !function.getBody().hasOneBlock())
    return false;

  FunctionType type = function.getFunctionType();
  if (type.getNumInputs() != wrapper.getInputs().size() ||
      type.getNumResults() != wrapper.getInputs().size() ||
      type.getInputs() != type.getResults())
    return false;

  Block &body = function.getBody().front();
  auto returnOp = dyn_cast<func::ReturnOp>(body.getTerminator());
  return returnOp && returnOp.getNumOperands() == type.getNumResults();
}

/// Fuse two wrappers separated only by memory-effect-free bookkeeping. The
/// initial rule deliberately requires alias analysis to prove that both
/// wrappers cover exactly the same unique, pairwise-disjoint buffers. This
/// handles ping-pong kernels while avoiding assumptions about partially
/// overlapping, duplicate, or otherwise unknown XLA arguments.
class FuseSequentialXLAWrappers final
    : public OpRewritePattern<enzymexla::XLAWrapperOp> {
public:
  FuseSequentialXLAWrappers(MLIRContext *context, DataFlowSolver &aliasSolver)
      : OpRewritePattern(context), aliasSolver(aliasSolver) {}

  LogicalResult matchAndRewrite(enzymexla::XLAWrapperOp first,
                                PatternRewriter &rewriter) const override {
    Operation *next = first->getNextNode();
    while (next && isMemoryEffectFree(next))
      next = next->getNextNode();

    auto second = dyn_cast_or_null<enzymexla::XLAWrapperOp>(next);
    if (!second || first.getInputs().empty() ||
        first.getInputs().size() != second.getInputs().size())
      return failure();

    // Argument/result metadata has no composition rule yet.
    if (first.getArgAttrsAttr() || first.getResAttrsAttr() ||
        second.getArgAttrsAttr() || second.getResAttrsAttr())
      return failure();

    auto firstFunction = dyn_cast_or_null<func::FuncOp>(
        SymbolTable::lookupNearestSymbolFrom(first, first.getFnAttr()));
    auto secondFunction = dyn_cast_or_null<func::FuncOp>(
        SymbolTable::lookupNearestSymbolFrom(second, second.getFnAttr()));
    if (!firstFunction || !secondFunction ||
        !isRaisedWrapperFunction(firstFunction, first) ||
        !isRaisedWrapperFunction(secondFunction, second))
      return failure();

    SmallVector<Value> firstInputs(first.getInputs().begin(),
                                   first.getInputs().end());
    SmallVector<Value> firstIdentities;
    for (Value input : firstInputs)
      firstIdentities.push_back(getBufferIdentity(input));

    SmallVector<Value> secondIdentities;
    for (Value input : second.getInputs())
      secondIdentities.push_back(getBufferIdentity(input));

    auto arePairwiseNoAlias = [&](ArrayRef<Value> identities) {
      for (unsigned i = 0; i < identities.size(); ++i) {
        for (unsigned j = i + 1; j < identities.size(); ++j) {
          std::optional<AliasResult> result =
              queryAlias(identities[i], identities[j]);
          if (!result || !result->isNo())
            return false;
        }
      }
      return true;
    };
    if (!arePairwiseNoAlias(firstIdentities) ||
        !arePairwiseNoAlias(secondIdentities))
      return failure();

    SmallVector<unsigned> secondToFirst;
    llvm::BitVector matchedFirstBuffers(firstIdentities.size());
    for (auto [secondIndex, identity] : llvm::enumerate(secondIdentities)) {
      std::optional<unsigned> mappedIndex;
      for (auto [firstIndex, firstIdentity] :
           llvm::enumerate(firstIdentities)) {
        std::optional<AliasResult> result = queryAlias(identity, firstIdentity);
        if (!result)
          return failure();
        if (result->isMust()) {
          if (mappedIndex)
            return failure();
          mappedIndex = firstIndex;
        } else if (!result->isNo()) {
          // MayAlias and PartialAlias cannot safely identify which tensor
          // state must be forwarded to the second wrapper.
          return failure();
        }
      }

      if (!mappedIndex || matchedFirstBuffers.test(*mappedIndex))
        return failure();

      matchedFirstBuffers.set(*mappedIndex);
      if (firstInputs[*mappedIndex].getType() !=
              second.getInputs()[secondIndex].getType() ||
          firstFunction.getArgumentTypes()[*mappedIndex] !=
              secondFunction.getArgumentTypes()[secondIndex])
        return failure();
      secondToFirst.push_back(*mappedIndex);
    }

    ModuleOp module = first->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    std::string name;
    do {
      name = "rxla$megakernel_" + std::to_string(nextMegakernelId++);
    } while (SymbolTable::lookupSymbolIn(module, name));

    Location location =
        FusedLoc::get(first.getContext(), {first.getLoc(), second.getLoc()});

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(module.getBody());
    auto fusedFunction = func::FuncOp::create(rewriter, location, name,
                                              firstFunction.getFunctionType());
    fusedFunction.setPrivate();
    Block *fusedBody = fusedFunction.addEntryBlock();
    rewriter.setInsertionPointToEnd(fusedBody);

    SmallVector<Value> bufferState(fusedBody->getArguments().begin(),
                                   fusedBody->getArguments().end());

    auto cloneBody = [&](func::FuncOp function,
                         ArrayRef<Value> arguments) -> SmallVector<Value> {
      IRMapping mapping;
      for (auto [argument, replacement] :
           llvm::zip_equal(function.getArguments(), arguments))
        mapping.map(argument, replacement);

      Block &body = function.getBody().front();
      for (Operation &operation : body.without_terminator())
        rewriter.clone(operation, mapping);

      auto returnOp = cast<func::ReturnOp>(body.getTerminator());
      SmallVector<Value> results;
      for (Value result : returnOp.getOperands())
        results.push_back(mapping.lookupOrDefault(result));
      return results;
    };

    bufferState = cloneBody(firstFunction, bufferState);

    SmallVector<Value> secondArguments;
    for (unsigned firstIndex : secondToFirst)
      secondArguments.push_back(bufferState[firstIndex]);
    SmallVector<Value> secondResults =
        cloneBody(secondFunction, secondArguments);
    for (auto [secondIndex, firstIndex] : llvm::enumerate(secondToFirst))
      bufferState[firstIndex] = secondResults[secondIndex];

    func::ReturnOp::create(rewriter, location, bufferState);

    rewriter.setInsertionPoint(second);
    enzymexla::XLAWrapperOp::create(
        rewriter, location, SymbolRefAttr::get(fusedFunction), firstInputs,
        /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr);
    rewriter.eraseOp(second);
    rewriter.eraseOp(first);
    return success();
  }

private:
  std::optional<AliasResult> queryAlias(Value lhs, Value rhs) const {
    auto *lhsState = aliasSolver.lookupState<enzyme::AliasClassLattice>(lhs);
    auto *rhsState = aliasSolver.lookupState<enzyme::AliasClassLattice>(rhs);
    if (!lhsState || !rhsState || lhsState->isUndefined() ||
        rhsState->isUndefined())
      return std::nullopt;
    return lhsState->alias(*rhsState);
  }

  DataFlowSolver &aliasSolver;
  mutable unsigned nextMegakernelId = 0;
};

struct XLAWrapperMegakernelizePass
    : public enzyme::impl::XLAWrapperMegakernelizePassBase<
          XLAWrapperMegakernelizePass> {
  using XLAWrapperMegakernelizePassBase::XLAWrapperMegakernelizePassBase;

  void runOnOperation() override {
    DataFlowSolver aliasSolver(DataFlowConfig().setInterprocedural(false));
    dataflow::loadBaselineAnalyses(aliasSolver);
    aliasSolver.load<enzyme::AliasAnalysis>(&getContext());
    aliasSolver.load<enzyme::PointsToPointerAnalysis>();
    if (failed(aliasSolver.initializeAndRun(getOperation()))) {
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<FuseSequentialXLAWrappers>(&getContext(), aliasSolver);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
