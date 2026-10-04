//===- XLAMegakernelize.cpp - Fuse raised XLA wrappers -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"

#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"

#include "Enzyme/MLIR/Interfaces/Utils.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "llvm/ADT/APInt.h"

#include <type_traits>

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

static ArrayAttr prependBoundAttrs(ArrayAttr attrs, unsigned numBounds) {
  if (!attrs)
    return {};
  SmallVector<Attribute> newAttrs(numBounds,
                                  DictionaryAttr::get(attrs.getContext()));
  llvm::append_range(newAttrs, attrs);
  return ArrayAttr::get(attrs.getContext(), newAttrs);
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

/// Move a host loop with one XLA wrapper into a stablehlo.while.
/// Pass the updated tensors directly from one iteration to the next.
template <typename LoopOp>
class LiftXLAWrapperLoop final : public OpRewritePattern<LoopOp> {
public:
  using OpRewritePattern<LoopOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LoopOp loop,
                                PatternRewriter &rewriter) const override {
    if (loop.getNumResults() != 0)
      return failure();

    Value induction = loop.getInductionVar();
    Region &region = loop.getRegion();
    if (!induction.use_empty() || !region.hasOneBlock())
      return failure();

    Block *body = &region.front();
    if (body->getNumArguments() != 1 ||
        !body->getTerminator()->getOperands().empty())
      return failure();

    enzymexla::XLAWrapperOp wrapper;
    for (Operation &operation : body->without_terminator()) {
      if (auto candidate = dyn_cast<enzymexla::XLAWrapperOp>(&operation)) {
        if (wrapper)
          return failure();
        wrapper = candidate;
        continue;
      }

      // Move only pure operations without regions out of the loop.
      // These operations must be safe when the loop has zero iterations.
      // The unused induction variable makes their inputs loop-invariant.
      if (operation.getNumRegions() != 0 || !isPure(&operation))
        return failure();
    }
    if (!wrapper)
      return failure();

    auto function = dyn_cast_or_null<FunctionOpInterface>(
        SymbolTable::lookupNearestSymbolFrom(wrapper, wrapper.getFnAttr()));
    if (!function || !isRaisedWrapperFunction(function, wrapper))
      return failure();

    Type hostBoundType = induction.getType();
    Type scalarType = hostBoundType;
    if (isa<IndexType>(hostBoundType)) {
      scalarType = rewriter.getI64Type();
    } else {
      auto integerType = dyn_cast<IntegerType>(hostBoundType);
      if (!integerType || !integerType.isSignless())
        return failure();
    }

    ModuleOp module = loop->template getParentOfType<ModuleOp>();
    // Keep symbol references in the same scope.
    if (!module || function->getParentOp() != module.getOperation())
      return failure();

    Block *allocaBlock = enzyme::getAllocaBlock(loop);
    if (!allocaBlock)
      return failure();

    // Expand only the bounds of this loop. Use the maximum lower bound
    // and the minimum upper bound when an affine map has several results.
    SmallVector<Value, 3> boundValues;
    bool unsignedCmp = false;
    if constexpr (std::is_same_v<LoopOp, affine::AffineForOp>) {
      boundValues = {lowerAffineLowerBound(loop, rewriter),
                     lowerAffineUpperBound(loop, rewriter),
                     arith::ConstantIndexOp::create(rewriter, loop.getLoc(),
                                                    loop.getStepAsInt())};
    } else {
      boundValues = {loop.getLowerBound(), loop.getUpperBound(),
                     loop.getStep()};
      unsignedCmp = loop.getUnsignedCmp();
    }
    SmallVector<std::optional<APInt>, 3> constantBounds(3);
    SmallVector<unsigned, 3> dynamicBoundIndices;
    unsigned scalarBitWidth = cast<IntegerType>(scalarType).getWidth();
    for (auto [index, bound] : llvm::enumerate(boundValues)) {
      APInt constant;
      if (matchPattern(bound, m_ConstantInt(&constant))) {
        constantBounds[index] = constant.sextOrTrunc(scalarBitWidth);
        continue;
      }
      dynamicBoundIndices.push_back(index);
    }

    // Round the copy size up to a whole number of bytes.
    uint64_t scalarByteWidth = (scalarBitWidth + 7) / 8;

    Location location =
        FusedLoc::get(loop->getContext(), {loop->getLoc(), wrapper.getLoc()});
    auto scalarTensorType = RankedTensorType::get({}, scalarType);

    SmallVector<Type> megakernelTypes(dynamicBoundIndices.size(),
                                      scalarTensorType);
    megakernelTypes.append(function.getArgumentTypes().begin(),
                           function.getArgumentTypes().end());
    SmallVector<Type> megakernelResults(dynamicBoundIndices.size(),
                                        scalarTensorType);
    llvm::append_range(megakernelResults, function.getResultTypes());
    Type megakernelType =
        function.cloneTypeWith(megakernelTypes, megakernelResults);

    // Add empty attributes for the new bound slots. Keep attributes on the
    // original inputs and results, including the trailing specialized inputs.
    ArrayAttr functionArgAttrs = prependBoundAttrs(function.getAllArgAttrs(),
                                                   dynamicBoundIndices.size());
    ArrayAttr functionResAttrs = prependBoundAttrs(function.getAllResultAttrs(),
                                                   dynamicBoundIndices.size());

    FunctionOpInterface megakernel = function;
    Block *originalBody = nullptr;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      Block *entry;
      if (canReuseFunction(function, wrapper, {}, module)) {
        // Keep the original body until its operations and return are copied.
        originalBody = &function.getFunctionBody().front();
        entry = rewriter.createBlock(
            &function.getFunctionBody(), function.getFunctionBody().end(),
            megakernelTypes,
            SmallVector<Location>(megakernelTypes.size(), location));
      } else {
        rewriter.setInsertionPointToEnd(module.getBody());
        std::string name = getUniqueMegakernelName(module, nextMegakernelId);
        megakernel = createRaisedFunction(rewriter, function, location, name,
                                          megakernelType);
        entry = &megakernel.front();
      }
      rewriter.setInsertionPointToEnd(entry);

      SmallVector<Value, 3> hloBounds(3);
      unsigned nextDynamicBound = 0;
      for (unsigned index = 0; index < boundValues.size(); ++index) {
        if (constantBounds[index]) {
          Attribute scalarAttr =
              IntegerAttr::get(scalarType, *constantBounds[index]);
          auto tensorAttr = SplatElementsAttr::get(
              scalarTensorType, ArrayRef<Attribute>{scalarAttr});
          hloBounds[index] = stablehlo::ConstantOp::create(
              rewriter, location, scalarTensorType, tensorAttr);
          continue;
        }
        hloBounds[index] = entry->getArgument(nextDynamicBound++);
      }

      // Only the induction variable and updated buffers change each iteration.
      // Capture the limit, step, and specialized scalars from the entry block.
      SmallVector<Value> initialState{hloBounds[0]};
      llvm::append_range(initialState,
                         entry->getArguments().slice(dynamicBoundIndices.size(),
                                                     function.getNumResults()));

      SmallVector<Type> stateTypes{scalarTensorType};
      llvm::append_range(stateTypes, function.getResultTypes());
      auto whileOp = stablehlo::WhileOp::create(rewriter, location, stateTypes,
                                                initialState);

      Block *condition = rewriter.createBlock(&whileOp.getCond());
      for (Type type : stateTypes)
        condition->addArgument(type, location);
      rewriter.setInsertionPointToStart(condition);
      stablehlo::ComparisonType comparisonType =
          unsignedCmp ? stablehlo::ComparisonType::UNSIGNED
                      : stablehlo::ComparisonType::SIGNED;
      Value keepGoing = stablehlo::CompareOp::create(
          rewriter, location, condition->getArgument(0), hloBounds[1],
          stablehlo::ComparisonDirection::LT, comparisonType);
      stablehlo::ReturnOp::create(rewriter, location, keepGoing);

      Block *whileBody = rewriter.createBlock(&whileOp.getBody());
      for (Type type : stateTypes)
        whileBody->addArgument(type, location);
      rewriter.setInsertionPointToStart(whileBody);

      Value nextInduction = stablehlo::AddOp::create(
          rewriter, location, whileBody->getArgument(0), hloBounds[2]);
      SmallVector<Value> yielded{nextInduction};
      SmallVector<Value> functionArguments(
          whileBody->getArguments().drop_front());
      llvm::append_range(functionArguments, entry->getArguments().take_back(
                                                wrapper.getNumSpecialized()));
      llvm::append_range(yielded, cloneRaisedFunctionBody(rewriter, function,
                                                          functionArguments));
      stablehlo::ReturnOp::create(rewriter, location, yielded);

      rewriter.setInsertionPointAfter(whileOp);
      // Return the bound buffers unchanged to preserve the wrapper signature.
      SmallVector<Value> results(
          entry->getArguments().take_front(dynamicBoundIndices.size()));
      llvm::append_range(results, whileOp.getResults().drop_front());
      createRaisedReturn(rewriter, function, results);
    }

    if (originalBody) {
      rewriter.eraseBlock(originalBody);
      rewriter.modifyOpInPlace(megakernel, [&] {
        megakernel.setType(megakernelType);
        megakernel->setLoc(location);
      });
    }

    rewriter.modifyOpInPlace(megakernel, [&] {
      if (functionArgAttrs)
        megakernel.setAllArgAttrs(functionArgAttrs);
      if (functionResAttrs)
        megakernel.setAllResultAttrs(functionResAttrs);
    });

    rewriter.setInsertionPoint(loop.getOperation());
    SmallVector<Value> newWrapperInputs;
    auto hostMemrefType = MemRefType::get({}, scalarType);
    auto deviceMemrefType =
        MemRefType::get({}, scalarType, MemRefLayoutAttrInterface{},
                        rewriter.getI64IntegerAttr(1));

    // Allocate each bound once when the module starts. Copy its current
    // value before each call. The module frees the storage at shutdown.
    // Calls to this loop must not overlap: they share the bound storage.
    SymbolTable symbols(module);
    std::string name = SymbolTable::getSymbolName(megakernel).getValue().str();
    Value copySize;
    if (!dynamicBoundIndices.empty())
      copySize = arith::ConstantIndexOp::create(
          rewriter, location, static_cast<int64_t>(scalarByteWidth));
    for (unsigned index : dynamicBoundIndices) {
      Value bound = boundValues[index];
      if (isa<IndexType>(hostBoundType))
        bound =
            arith::IndexCastOp::create(rewriter, location, scalarType, bound);

      Value hostStorage;
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(allocaBlock);
        hostStorage =
            memref::AllocaOp::create(rewriter, location, hostMemrefType);
      }
      memref::StoreOp::create(rewriter, location, bound, hostStorage,
                              ValueRange());
      enzymexla::TempAllocOp allocation;
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToEnd(module.getBody());
        allocation = enzymexla::TempAllocOp::create(
            rewriter, location, name + "_bound_" + std::to_string(index),
            rewriter.getStringAttr("private"), deviceMemrefType);
        symbols.insert(allocation);
      }
      Value deviceStorage = enzymexla::GetGlobalTempOp::create(
          rewriter, location, deviceMemrefType,
          FlatSymbolRefAttr::get(allocation));
      enzymexla::MemcpyOp::create(rewriter, location,
                                  /*asyncToken=*/(Type) nullptr,
                                  /*asyncDependencies=*/ValueRange(),
                                  deviceStorage, hostStorage, copySize);
      newWrapperInputs.push_back(deviceStorage);
    }

    // Copy the pure operations before the new call. Keep values from
    // outside the loop unchanged.
    IRMapping mapping;
    for (Operation &operation : body->without_terminator()) {
      if (&operation != wrapper.getOperation())
        rewriter.clone(operation, mapping);
    }
    for (Value input : wrapper.getInputs())
      newWrapperInputs.push_back(mapping.lookupOrDefault(input));

    auto newWrapper = cast<enzymexla::XLAWrapperOp>(rewriter.clone(*wrapper));
    newWrapper->setLoc(location);
    newWrapper.setFnAttr(SymbolRefAttr::get(megakernel));
    newWrapper.getInputsMutable().assign(newWrapperInputs);
    if (auto attrs = wrapper.getArgAttrsAttr())
      newWrapper.setArgAttrsAttr(
          prependBoundAttrs(attrs, dynamicBoundIndices.size()));
    if (auto attrs = wrapper.getResAttrsAttr())
      newWrapper.setResAttrsAttr(
          prependBoundAttrs(attrs, dynamicBoundIndices.size()));
    rewriter.eraseOp(loop.getOperation());
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
    patterns.add<FuseSequentialXLAWrappers, LiftXLAWrapperLoop<scf::ForOp>,
                 LiftXLAWrapperLoop<affine::AffineForOp>>(&getContext());
    enzymexla::XLAWrapperOp::getCanonicalizationPatterns(patterns,
                                                         &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace
