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

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
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

static bool isMemoryEffectFreeRaisedFunction(FunctionOpInterface function) {
  return !function.getFunctionBody()
              .walk([](Operation *operation) {
                return isMemoryEffectFree(operation) ? WalkResult::advance()
                                                     : WalkResult::interrupt();
              })
              .wasInterrupted();
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
    if (!wrapper || wrapper.getArgAttrsAttr() || wrapper.getResAttrsAttr())
      return failure();

    auto function = dyn_cast_or_null<FunctionOpInterface>(
        SymbolTable::lookupNearestSymbolFrom(wrapper, wrapper.getFnAttr()));
    if (!function || !isRaisedWrapperFunction(function, wrapper) ||
        !isMemoryEffectFreeRaisedFunction(function))
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
    // Keep symbol references in the same scope. Do not move argument or
    // result attributes to the new bound slots.
    if (!module || function->getParentOp() != module.getOperation() ||
        hasMetadata(function.getAllArgAttrs()) ||
        hasMetadata(function.getAllResultAttrs()))
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
    Type megakernelType =
        function.cloneTypeWith(megakernelTypes, megakernelTypes);
    std::string name = getUniqueMegakernelName(module, nextMegakernelId);

    FunctionOpInterface megakernel;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(module.getBody());
      megakernel = createRaisedFunction(rewriter, function, location, name,
                                        megakernelType);
      Block *entry = &megakernel.front();
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

      SmallVector<Value> initialState(hloBounds.begin(), hloBounds.end());
      llvm::append_range(initialState, entry->getArguments().drop_front(
                                           dynamicBoundIndices.size()));

      SmallVector<Type> stateTypes(3, scalarTensorType);
      stateTypes.append(function.getArgumentTypes().begin(),
                        function.getArgumentTypes().end());
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
          rewriter, location, condition->getArgument(0),
          condition->getArgument(1), stablehlo::ComparisonDirection::LT,
          comparisonType);
      stablehlo::ReturnOp::create(rewriter, location, keepGoing);

      Block *whileBody = rewriter.createBlock(&whileOp.getBody());
      for (Type type : stateTypes)
        whileBody->addArgument(type, location);
      rewriter.setInsertionPointToStart(whileBody);

      Value nextInduction = stablehlo::AddOp::create(rewriter, location,
                                                     whileBody->getArgument(0),
                                                     whileBody->getArgument(2));
      SmallVector<Value> yielded = {nextInduction, whileBody->getArgument(1),
                                    whileBody->getArgument(2)};
      SmallVector<Value> functionArguments(
          whileBody->getArguments().drop_front(3));
      llvm::append_range(yielded, cloneRaisedFunctionBody(rewriter, function,
                                                          functionArguments));
      stablehlo::ReturnOp::create(rewriter, location, yielded);

      rewriter.setInsertionPointAfter(whileOp);
      SmallVector<Value> results;
      for (unsigned index : dynamicBoundIndices)
        results.push_back(whileOp.getResult(index));
      llvm::append_range(results, whileOp.getResults().drop_front(3));
      createRaisedReturn(rewriter, function, results);
    }

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
    Value copySize;
    if (!dynamicBoundIndices.empty())
      copySize = arith::ConstantIndexOp::create(
          rewriter, location, static_cast<int64_t>(scalarByteWidth));
    for (unsigned index : dynamicBoundIndices) {
      Value bound = boundValues[index];
      if (isa<IndexType>(hostBoundType))
        bound =
            arith::IndexCastOp::create(rewriter, location, scalarType, bound);

      Value hostStorage =
          memref::AllocaOp::create(rewriter, location, hostMemrefType);
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

    enzymexla::XLAWrapperOp::create(
        rewriter, location, SymbolRefAttr::get(megakernel), newWrapperInputs,
        /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr);
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
