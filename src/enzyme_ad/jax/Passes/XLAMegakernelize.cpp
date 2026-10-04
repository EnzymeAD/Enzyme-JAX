//===- XLAMegakernelize.cpp - Fuse raised XLA wrappers -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/enzyme_ad/jax/Passes/Passes.h"

#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_XLAMEGAKERNELIZEPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;

namespace {

/// Remove casts that preserve the buffer address. Keep subviews because they
/// can change the offset.
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

static bool isRaisedWrapperFunction(FunctionOpInterface function,
                                    enzymexla::XLAWrapperOp wrapper) {
  if (function.isExternal() || !function.getFunctionBody().hasOneBlock())
    return false;

  if (function.getNumArguments() != wrapper.getInputs().size() ||
      function.getNumResults() != wrapper.getInputs().size() ||
      function.getArgumentTypes() != function.getResultTypes())
    return false;

  Block &body = function.getFunctionBody().front();
  Operation *returnOp = body.getTerminator();
  return returnOp->hasTrait<OpTrait::ReturnLike>() &&
         returnOp->getNumOperands() == function.getNumResults();
}

static SmallVector<Value> cloneRaisedFunctionBody(PatternRewriter &rewriter,
                                                  FunctionOpInterface function,
                                                  ArrayRef<Value> arguments) {
  IRMapping mapping;
  for (auto [argument, replacement] :
       llvm::zip_equal(function.getArguments(), arguments))
    mapping.map(argument, replacement);

  Block &body = function.getFunctionBody().front();
  for (Operation &operation : body.without_terminator())
    rewriter.clone(operation, mapping);

  SmallVector<Value> results;
  for (Value result : body.getTerminator()->getOperands())
    results.push_back(mapping.lookupOrDefault(result));
  return results;
}

static std::string getUniqueMegakernelName(ModuleOp module,
                                           unsigned &nextMegakernelId) {
  std::string name;
  do {
    name = "rxla$megakernel_" + std::to_string(nextMegakernelId++);
  } while (SymbolTable::lookupSymbolIn(module, name));
  return name;
}

static FunctionOpInterface
createRaisedFunction(PatternRewriter &rewriter, FunctionOpInterface source,
                     Location location, StringRef name, Type functionType) {
  auto function = cast<FunctionOpInterface>(
      rewriter.cloneWithoutRegions(*source.getOperation()));
  function->setLoc(location);
  SymbolTable::setSymbolName(function, name);
  SymbolTable::setSymbolVisibility(function, SymbolTable::Visibility::Private);
  function.setType(functionType);
  function.addEntryBlock();
  return function;
}

static void createRaisedReturn(PatternRewriter &rewriter,
                               FunctionOpInterface source, ValueRange results) {
  Operation *returnOp = rewriter.cloneWithoutRegions(
      *source.getFunctionBody().front().getTerminator());
  returnOp->setOperands(results);
}

static bool hasMetadata(ArrayAttr attributes) {
  return attributes && llvm::any_of(attributes, [](Attribute attribute) {
           return !cast<DictionaryAttr>(attribute).empty();
         });
}

/// Fuse two wrappers separated only by memory-effect-free bookkeeping. The
/// fused wrapper shares tensor state for equal buffer identities and retains
/// separate state for distinct inputs.
class FuseSequentialXLAWrappers final
    : public OpRewritePattern<enzymexla::XLAWrapperOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(enzymexla::XLAWrapperOp first,
                                PatternRewriter &rewriter) const override {
    Operation *next = first->getNextNode();
    while (next && isMemoryEffectFree(next))
      next = next->getNextNode();

    auto second = dyn_cast_or_null<enzymexla::XLAWrapperOp>(next);
    if (!second || first.getInputs().empty() || second.getInputs().empty())
      return failure();

    // Do not combine metadata or specialized scalar inputs.
    if (first.getArgAttrsAttr() || first.getResAttrsAttr() ||
        second.getArgAttrsAttr() || second.getResAttrsAttr() ||
        first.getNumSpecialized() || second.getNumSpecialized())
      return failure();

    auto firstFunction = dyn_cast_or_null<FunctionOpInterface>(
        SymbolTable::lookupNearestSymbolFrom(first, first.getFnAttr()));
    auto secondFunction = dyn_cast_or_null<FunctionOpInterface>(
        SymbolTable::lookupNearestSymbolFrom(second, second.getFnAttr()));
    if (!firstFunction || !secondFunction ||
        !isRaisedWrapperFunction(firstFunction, first) ||
        !isRaisedWrapperFunction(secondFunction, second))
      return failure();

    ModuleOp module = first->getParentOfType<ModuleOp>();
    // Keep symbol references in their original scope. Do not change the
    // positions of arguments or results that have metadata.
    if (!module || firstFunction->getParentOp() != module.getOperation() ||
        secondFunction->getParentOp() != module.getOperation() ||
        hasMetadata(firstFunction.getAllArgAttrs()) ||
        hasMetadata(firstFunction.getAllResultAttrs()) ||
        hasMetadata(secondFunction.getAllArgAttrs()) ||
        hasMetadata(secondFunction.getAllResultAttrs()))
      return failure();

    SmallVector<Value> fusedInputs(first.getInputs().begin(),
                                   first.getInputs().end());
    SmallVector<Type> fusedTypes(firstFunction.getArgumentTypes());
    SmallVector<Value> fusedIdentities;
    for (Value input : fusedInputs)
      fusedIdentities.push_back(getBufferIdentity(input));

    SmallVector<Value> secondIdentities;
    for (Value input : second.getInputs())
      secondIdentities.push_back(getBufferIdentity(input));

    // Do not fuse duplicate inputs whose outputs can disagree. Wrapper
    // canonicalization can remove duplicates when their outputs agree.
    DenseSet<Value> distinct;
    for (Value identity : fusedIdentities)
      if (!distinct.insert(identity).second)
        return failure();
    distinct.clear();
    for (Value identity : secondIdentities)
      if (!distinct.insert(identity).second)
        return failure();

    SmallVector<unsigned> secondToFused;
    for (auto [secondIndex, identity] : llvm::enumerate(secondIdentities)) {
      std::optional<unsigned> mappedIndex;
      for (auto [fusedIndex, fusedIdentity] :
           llvm::enumerate(fusedIdentities)) {
        if (identity == fusedIdentity) {
          mappedIndex = fusedIndex;
          break;
        }
      }

      if (!mappedIndex) {
        // Add a separate state slot for this input.
        mappedIndex = fusedInputs.size();
        fusedInputs.push_back(second.getInputs()[secondIndex]);
        fusedTypes.push_back(secondFunction.getArgumentTypes()[secondIndex]);
        fusedIdentities.push_back(identity);
      } else if (fusedInputs[*mappedIndex].getType() !=
                     second.getInputs()[secondIndex].getType() ||
                 fusedTypes[*mappedIndex] !=
                     secondFunction.getArgumentTypes()[secondIndex]) {
        return failure();
      }
      secondToFused.push_back(*mappedIndex);
    }

    // Reuse the first wrapper's views for shared buffers. New inputs must
    // already exist at the first call; do not move their definitions.
    DominanceInfo dominance;
    for (Value input : fusedInputs)
      if (!dominance.properlyDominates(input, first))
        return failure();
    Type fusedType = firstFunction.cloneTypeWith(fusedTypes, fusedTypes);
    if (!fusedType)
      return failure();
    std::string name = getUniqueMegakernelName(module, nextMegakernelId);

    Location location =
        FusedLoc::get(first.getContext(), {first.getLoc(), second.getLoc()});

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(module.getBody());
    auto fusedFunction = createRaisedFunction(rewriter, firstFunction, location,
                                              name, fusedType);
    // The first function's effect summary does not describe the combined body.
    fusedFunction->removeAttr("enzymexla.memory_effects");
    fusedFunction->removeAttr("memory_effects");
    Block *fusedBody = &fusedFunction.getFunctionBody().front();
    rewriter.setInsertionPointToEnd(fusedBody);

    SmallVector<Value> bufferState(fusedBody->getArguments().begin(),
                                   fusedBody->getArguments().end());

    SmallVector<Value> firstResults = cloneRaisedFunctionBody(
        rewriter, firstFunction,
        ArrayRef<Value>(bufferState).take_front(first.getInputs().size()));
    for (auto [firstIndex, result] : llvm::enumerate(firstResults))
      bufferState[firstIndex] = result;

    SmallVector<Value> secondArguments;
    for (unsigned fusedIndex : secondToFused)
      secondArguments.push_back(bufferState[fusedIndex]);
    SmallVector<Value> secondResults =
        cloneRaisedFunctionBody(rewriter, secondFunction, secondArguments);
    for (auto [secondIndex, fusedIndex] : llvm::enumerate(secondToFused))
      bufferState[fusedIndex] = secondResults[secondIndex];

    createRaisedReturn(rewriter, firstFunction, bufferState);

    rewriter.setInsertionPoint(first);
    enzymexla::XLAWrapperOp::create(
        rewriter, location, SymbolRefAttr::get(fusedFunction), fusedInputs,
        /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr);
    rewriter.eraseOp(second);
    rewriter.eraseOp(first);
    return success();
  }

private:
  mutable unsigned nextMegakernelId = 0;
};

struct XLAMegakernelizePass
    : public enzyme::impl::XLAMegakernelizePassBase<XLAMegakernelizePass> {
  using XLAMegakernelizePassBase::XLAMegakernelizePassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseSequentialXLAWrappers>(&getContext());
    enzymexla::XLAWrapperOp::getCanonicalizationPatterns(patterns,
                                                         &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace
