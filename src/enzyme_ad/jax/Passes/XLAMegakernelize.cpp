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

#include "Enzyme/MLIR/Interfaces/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

static bool isRaisedWrapperFunction(FunctionOpInterface function,
                                    enzymexla::XLAWrapperOp wrapper) {
  if (function.isExternal() || !function.getFunctionBody().hasOneBlock())
    return false;

  if (wrapper.getNumSpecialized() > wrapper.getInputs().size())
    return false;
  unsigned numBuffers =
      wrapper.getInputs().size() - wrapper.getNumSpecialized();
  if (function.getNumArguments() != wrapper.getInputs().size() ||
      function.getNumResults() != numBuffers ||
      function.getArgumentTypes().take_front(numBuffers) !=
          function.getResultTypes())
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

static bool canReuseFunction(FunctionOpInterface function,
                             enzymexla::XLAWrapperOp first,
                             enzymexla::XLAWrapperOp second, ModuleOp module) {
  if (SymbolTable::getSymbolVisibility(function) !=
          SymbolTable::Visibility::Private ||
      function->isAncestor(first))
    return false;
  auto uses = SymbolTable::getSymbolUses(function.getOperation(), module);
  if (!uses)
    return false;
  for (const auto &use : *uses)
    if (use.getUser() != first && use.getUser() != second)
      return false;
  return true;
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
    if (!second)
      return failure();

    // Do not combine argument or result metadata.
    if (first.getArgAttrsAttr() || first.getResAttrsAttr() ||
        second.getArgAttrsAttr() || second.getResAttrsAttr())
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

    unsigned firstNumSpecialized = first.getNumSpecialized();
    unsigned secondNumSpecialized = second.getNumSpecialized();
    auto firstBuffers = first.getInputs().drop_back(firstNumSpecialized);
    auto secondBuffers = second.getInputs().drop_back(secondNumSpecialized);
    SmallVector<Value> fusedInputs(firstBuffers.begin(), firstBuffers.end());
    SmallVector<Type> fusedTypes(firstFunction.getResultTypes());
    SmallVector<Value> fusedIdentities;
    for (Value input : fusedInputs)
      fusedIdentities.push_back(
          enzyme::oputils::getBaseObject(input, /*offsetAllowed=*/false));

    SmallVector<Value> secondIdentities;
    for (Value input : secondBuffers)
      secondIdentities.push_back(
          enzyme::oputils::getBaseObject(input, /*offsetAllowed=*/false));

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
        fusedInputs.push_back(secondBuffers[secondIndex]);
        fusedTypes.push_back(secondFunction.getArgumentTypes()[secondIndex]);
        fusedIdentities.push_back(identity);
      } else if (fusedInputs[*mappedIndex].getType() !=
                     secondBuffers[secondIndex].getType() ||
                 fusedTypes[*mappedIndex] !=
                     secondFunction.getArgumentTypes()[secondIndex]) {
        return failure();
      }
      secondToFused.push_back(*mappedIndex);
    }

    // Keep all specialized inputs after the buffer union, in call order.
    unsigned numBuffers = fusedInputs.size();
    SmallVector<Type> fusedResults(fusedTypes);
    llvm::append_range(fusedInputs,
                       first.getInputs().take_back(firstNumSpecialized));
    llvm::append_range(fusedInputs,
                       second.getInputs().take_back(secondNumSpecialized));
    llvm::append_range(fusedTypes, firstFunction.getArgumentTypes().take_back(
                                       firstNumSpecialized));
    llvm::append_range(fusedTypes, secondFunction.getArgumentTypes().take_back(
                                       secondNumSpecialized));

    // Reuse the first wrapper's views for shared buffers. New inputs must
    // already exist at the first call; do not move their definitions.
    for (Value input : fusedInputs)
      if (Operation *definition = input.getDefiningOp())
        if (definition->getBlock() == first->getBlock() &&
            !definition->isBeforeInBlock(first))
          return failure();
    Type fusedType = firstFunction.cloneTypeWith(fusedTypes, fusedResults);
    if (!fusedType)
      return failure();

    Location location =
        FusedLoc::get(first.getContext(), {first.getLoc(), second.getLoc()});

    OpBuilder::InsertionGuard guard(rewriter);
    FunctionOpInterface fusedFunction = firstFunction;
    Block *originalBody = nullptr;
    Block *fusedBody;
    if (canReuseFunction(firstFunction, first, second, module)) {
      // Keep the source block intact until both calls have been copied.
      originalBody = &firstFunction.getFunctionBody().front();
      fusedBody = rewriter.createBlock(
          &firstFunction.getFunctionBody(),
          firstFunction.getFunctionBody().end(), fusedTypes,
          SmallVector<Location>(fusedTypes.size(), location));
    } else {
      rewriter.setInsertionPointToEnd(module.getBody());
      std::string name = getUniqueMegakernelName(module, nextMegakernelId);
      fusedFunction = createRaisedFunction(rewriter, firstFunction, location,
                                           name, fusedType);
      fusedBody = &fusedFunction.getFunctionBody().front();
    }
    rewriter.setInsertionPointToEnd(fusedBody);

    SmallVector<Value> bufferState(
        fusedBody->getArguments().take_front(numBuffers));

    SmallVector<Value> firstArguments(
        ArrayRef<Value>(bufferState).take_front(firstBuffers.size()));
    llvm::append_range(firstArguments, fusedBody->getArguments().slice(
                                           numBuffers, firstNumSpecialized));
    SmallVector<Value> firstResults =
        cloneRaisedFunctionBody(rewriter, firstFunction, firstArguments);
    for (auto [firstIndex, result] : llvm::enumerate(firstResults))
      bufferState[firstIndex] = result;

    SmallVector<Value> secondArguments;
    for (unsigned fusedIndex : secondToFused)
      secondArguments.push_back(bufferState[fusedIndex]);
    llvm::append_range(secondArguments, fusedBody->getArguments().take_back(
                                            secondNumSpecialized));
    SmallVector<Value> secondResults =
        cloneRaisedFunctionBody(rewriter, secondFunction, secondArguments);
    for (auto [secondIndex, fusedIndex] : llvm::enumerate(secondToFused))
      bufferState[fusedIndex] = secondResults[secondIndex];

    createRaisedReturn(rewriter, firstFunction, bufferState);

    if (originalBody)
      rewriter.eraseBlock(originalBody);
    rewriter.modifyOpInPlace(fusedFunction, [&] {
      fusedFunction.setType(fusedType);
      fusedFunction->setLoc(location);
      // The first function's effect summary does not describe the combined
      // body.
      fusedFunction->removeAttr("enzymexla.memory_effects");
      fusedFunction->removeAttr("memory_effects");
    });

    rewriter.setInsertionPoint(first);
    auto fusedWrapper = enzymexla::XLAWrapperOp::create(
        rewriter, location, SymbolRefAttr::get(fusedFunction), fusedInputs,
        /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr);
    if (firstNumSpecialized || secondNumSpecialized)
      fusedWrapper.setNumSpecialized(firstNumSpecialized +
                                     secondNumSpecialized);
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
