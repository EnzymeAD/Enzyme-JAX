//===- XLAMegakernelize.cpp - Fuse raised XLA wrappers -------------------===//
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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_XLAMEGAKERNELIZEPASS
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

static bool isMemoryEffectFreeRaisedFunction(func::FuncOp function) {
  return !function.getBody()
              .walk([](Operation *operation) {
                return isMemoryEffectFree(operation) ? WalkResult::advance()
                                                     : WalkResult::interrupt();
              })
              .wasInterrupted();
}

static std::optional<AliasResult> queryAlias(DataFlowSolver &aliasSolver,
                                             Value lhs, Value rhs) {
  auto *lhsState = aliasSolver.lookupState<enzyme::AliasClassLattice>(lhs);
  auto *rhsState = aliasSolver.lookupState<enzyme::AliasClassLattice>(rhs);
  if (!lhsState || !rhsState || lhsState->isUndefined() ||
      rhsState->isUndefined())
    return std::nullopt;
  return lhsState->alias(*rhsState);
}

static bool arePairwiseNoAlias(DataFlowSolver &aliasSolver,
                               ArrayRef<Value> identities) {
  for (unsigned i = 0; i < identities.size(); ++i) {
    for (unsigned j = i + 1; j < identities.size(); ++j) {
      std::optional<AliasResult> result =
          queryAlias(aliasSolver, identities[i], identities[j]);
      if (!result || !result->isNo())
        return false;
    }
  }
  return true;
}

/// Return true when a host scalar expression can be reconstructed inside a
/// StableHLO function from constants and one or more device_mirror leaves.
/// Keep this list in sync with raiseMirroredBoundExpression below.  These are
/// the integer operations handled by ArithRaising as StableHLO elementwise
/// operations as well.
static bool collectMirroredBoundLeaves(Value value, Type hostBoundType,
                                       llvm::SetVector<Value> &mirrors) {
  if (value.getType() != hostBoundType)
    return false;

  if (auto mirror = value.getDefiningOp<enzymexla::DeviceMirrorOp>()) {
    mirrors.insert(mirror.getResult());
    return true;
  }

  APInt constant;
  if (matchPattern(value, m_ConstantInt(&constant)))
    return true;

  Operation *operation = value.getDefiningOp();
  if (!operation || operation->getNumOperands() != 2 ||
      !isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivSIOp,
           arith::DivUIOp, arith::MaxSIOp, arith::MaxUIOp, arith::MinSIOp,
           arith::MinUIOp, arith::ShLIOp, arith::ShRSIOp, arith::ShRUIOp,
           arith::AndIOp, arith::OrIOp, arith::XOrIOp, arith::RemSIOp,
           arith::RemUIOp>(operation))
    return false;

  return collectMirroredBoundLeaves(operation->getOperand(0), hostBoundType,
                                    mirrors) &&
         collectMirroredBoundLeaves(operation->getOperand(1), hostBoundType,
                                    mirrors);
}

static Value makeStableHLOIntegerConstant(OpBuilder &builder, Location loc,
                                          RankedTensorType tensorType,
                                          APInt value) {
  auto scalarType = cast<IntegerType>(tensorType.getElementType());
  Attribute scalarAttr = IntegerAttr::get(
      scalarType, value.sextOrTrunc(scalarType.getWidth()));
  auto tensorAttr = SplatElementsAttr::get(
      tensorType, ArrayRef<Attribute>{scalarAttr});
  return stablehlo::ConstantOp::create(builder, loc, tensorType, tensorAttr);
}

/// Rebuild an expression accepted by collectMirroredBoundLeaves using rank-0
/// StableHLO tensors.  mirrorArguments maps each annotation result to the XLA
/// function argument backed by its existing device allocation.
static Value raiseMirroredBoundExpression(
    OpBuilder &builder, Value value, RankedTensorType tensorType,
    const DenseMap<Value, Value> &mirrorArguments,
    DenseMap<Value, Value> &raisedValues) {
  if (auto found = raisedValues.find(value); found != raisedValues.end())
    return found->second;

  if (value.getDefiningOp<enzymexla::DeviceMirrorOp>()) {
    auto found = mirrorArguments.find(value);
    if (found == mirrorArguments.end())
      return {};
    raisedValues[value] = found->second;
    return found->second;
  }

  APInt constant;
  if (matchPattern(value, m_ConstantInt(&constant))) {
    Value result = makeStableHLOIntegerConstant(
        builder, value.getLoc(), tensorType, constant);
    raisedValues[value] = result;
    return result;
  }

  Operation *operation = value.getDefiningOp();
  if (!operation || operation->getNumOperands() != 2)
    return {};
  Value lhs = raiseMirroredBoundExpression(
      builder, operation->getOperand(0), tensorType, mirrorArguments,
      raisedValues);
  Value rhs = raiseMirroredBoundExpression(
      builder, operation->getOperand(1), tensorType, mirrorArguments,
      raisedValues);
  if (!lhs || !rhs)
    return {};

  Location loc = operation->getLoc();
  Value result;
  if (isa<arith::AddIOp>(operation))
    result = stablehlo::AddOp::create(builder, loc, lhs, rhs);
  else if (isa<arith::SubIOp>(operation))
    result = stablehlo::SubtractOp::create(builder, loc, lhs, rhs);
  else if (isa<arith::MulIOp>(operation))
    result = stablehlo::MulOp::create(builder, loc, lhs, rhs);
  else if (isa<arith::DivSIOp, arith::DivUIOp>(operation))
    result = stablehlo::DivOp::create(builder, loc, lhs, rhs);
  else if (isa<arith::MaxSIOp, arith::MaxUIOp>(operation))
    result = stablehlo::MaxOp::create(builder, loc, lhs, rhs);
  else if (isa<arith::MinSIOp, arith::MinUIOp>(operation))
    result = stablehlo::MinOp::create(builder, loc, lhs, rhs);
  else if (isa<arith::ShLIOp>(operation))
    result = stablehlo::ShiftLeftOp::create(builder, loc, lhs, rhs);
  else if (isa<arith::ShRSIOp>(operation))
    result = stablehlo::ShiftRightArithmeticOp::create(builder, loc, lhs, rhs);
  else if (isa<arith::ShRUIOp>(operation))
    result = stablehlo::ShiftRightLogicalOp::create(builder, loc, lhs, rhs);
  else if (isa<arith::AndIOp>(operation))
    result = stablehlo::AndOp::create(builder, loc, lhs, rhs);
  else if (isa<arith::OrIOp>(operation))
    result = stablehlo::OrOp::create(builder, loc, lhs, rhs);
  else if (isa<arith::XOrIOp>(operation))
    result = stablehlo::XorOp::create(builder, loc, lhs, rhs);
  else if (isa<arith::RemSIOp, arith::RemUIOp>(operation))
    result = stablehlo::RemOp::create(builder, loc, lhs, rhs);
  else
    return {};

  raisedValues[value] = result;
  return result;
}

static bool isUsableScalarDeviceMirror(enzymexla::DeviceMirrorOp mirror,
                                       Type scalarType) {
  Type deviceType = mirror.getDevice().getType();
  if (isa<LLVM::LLVMPointerType>(deviceType))
    return true;
  auto memrefType = dyn_cast<MemRefType>(deviceType);
  return memrefType && memrefType.getRank() == 1 &&
         memrefType.getDimSize(0) == 1 &&
         memrefType.getElementType() == scalarType;
}

static SmallVector<Value> cloneRaisedFunctionBody(PatternRewriter &rewriter,
                                                  func::FuncOp function,
                                                  ArrayRef<Value> arguments) {
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
}

static std::string getUniqueMegakernelName(ModuleOp module,
                                           unsigned &nextMegakernelId) {
  std::string name;
  do {
    name = "rxla$megakernel_" + std::to_string(nextMegakernelId++);
  } while (SymbolTable::lookupSymbolIn(module, name));
  return name;
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

    if (!arePairwiseNoAlias(aliasSolver, firstIdentities) ||
        !arePairwiseNoAlias(aliasSolver, secondIdentities))
      return failure();

    SmallVector<unsigned> secondToFirst;
    llvm::BitVector matchedFirstBuffers(firstIdentities.size());
    for (auto [secondIndex, identity] : llvm::enumerate(secondIdentities)) {
      std::optional<unsigned> mappedIndex;
      for (auto [firstIndex, firstIdentity] :
           llvm::enumerate(firstIdentities)) {
        std::optional<AliasResult> result =
            queryAlias(aliasSolver, identity, firstIdentity);
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

    std::string name = getUniqueMegakernelName(module, nextMegakernelId);

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

    bufferState = cloneRaisedFunctionBody(rewriter, firstFunction, bufferState);

    SmallVector<Value> secondArguments;
    for (unsigned firstIndex : secondToFirst)
      secondArguments.push_back(bufferState[firstIndex]);
    SmallVector<Value> secondResults =
        cloneRaisedFunctionBody(rewriter, secondFunction, secondArguments);
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
  DataFlowSolver &aliasSolver;
  mutable unsigned nextMegakernelId = 0;
};

/// Move a serial loop containing one raised XLA wrapper into a new raised
/// function as a stablehlo.while. The wrapper's tensor results become the
/// loop-carried state, so no intermediate iteration is materialized back to
/// host memory.
class LiftXLAWrapperLoop final : public OpRewritePattern<scf::ForOp> {
public:
  LiftXLAWrapperLoop(MLIRContext *context, DataFlowSolver &aliasSolver)
      : OpRewritePattern(context), aliasSolver(aliasSolver) {}

  LogicalResult matchAndRewrite(scf::ForOp loop,
                                PatternRewriter &rewriter) const override {
    if (loop.getNumResults() != 0 || !loop.getInitArgs().empty())
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

      // Only clone region-free, loop-invariant bookkeeping (for example,
      // pointer2memref) out of the loop. It must be safe to speculate because
      // the original loop may execute zero times. The IV-use check above
      // ensures these operations do not secretly vary across iterations.
      if (operation.getNumRegions() != 0 || !isPure(&operation))
        return failure();
    }
    if (!wrapper || wrapper.getArgAttrsAttr() || wrapper.getResAttrsAttr())
      return failure();

    auto function = dyn_cast_or_null<func::FuncOp>(
        SymbolTable::lookupNearestSymbolFrom(wrapper, wrapper.getFnAttr()));
    if (!function || !isRaisedWrapperFunction(function, wrapper) ||
        !isMemoryEffectFreeRaisedFunction(function))
      return failure();

    SmallVector<Value> identities;
    for (Value input : wrapper.getInputs())
      identities.push_back(getBufferIdentity(input));
    if (!arePairwiseNoAlias(aliasSolver, identities))
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

    SmallVector<Value, 3> boundValues = {loop.getLowerBound(),
                                         loop.getUpperBound(), loop.getStep()};
    struct BoundInfo {
      std::optional<APInt> constant;
      llvm::SetVector<Value> mirrorLeaves;
      std::optional<unsigned> scalarArgumentIndex;
      bool useMirrorExpression = false;
    };
    SmallVector<BoundInfo, 3> bounds(3);
    unsigned scalarBitWidth = cast<IntegerType>(scalarType).getWidth();
    for (auto [index, bound] : llvm::enumerate(boundValues)) {
      APInt constant;
      if (matchPattern(bound, m_ConstantInt(&constant))) {
        bounds[index].constant = constant.sextOrTrunc(scalarBitWidth);
        continue;
      }

      llvm::SetVector<Value> mirrorLeaves;
      if (collectMirroredBoundLeaves(bound, hostBoundType, mirrorLeaves) &&
          !mirrorLeaves.empty())
        bounds[index].mirrorLeaves = std::move(mirrorLeaves);
    }

    // A mirror is only a replacement for the temporary allocation when its
    // storage has the scalar representation this transform expects and alias
    // analysis proves that it is disjoint from every tensor state passed to
    // the XLA wrapper (including other mirrors).  MayAlias therefore takes the
    // existing allocation/copy fallback, preserving correctness.
    llvm::SetVector<Value> uniqueMirrorLeaves;
    for (BoundInfo &bound : bounds)
      for (Value mirror : bound.mirrorLeaves)
        uniqueMirrorLeaves.insert(mirror);

    bool canUseMirrors = !uniqueMirrorLeaves.empty();
    SmallVector<Value> allDeviceIdentities(identities.begin(),
                                           identities.end());
    for (Value mirrorValue : uniqueMirrorLeaves) {
      auto mirror = mirrorValue.getDefiningOp<enzymexla::DeviceMirrorOp>();
      if (!mirror || !isUsableScalarDeviceMirror(mirror, scalarType)) {
        canUseMirrors = false;
        break;
      }
      allDeviceIdentities.push_back(getBufferIdentity(mirror.getDevice()));
    }
    if (canUseMirrors &&
        !arePairwiseNoAlias(aliasSolver, allDeviceIdentities))
      canUseMirrors = false;

    SmallVector<Value> scalarSources;
    DenseMap<Value, unsigned> mirrorArgumentIndices;
    if (canUseMirrors) {
      for (Value mirror : uniqueMirrorLeaves) {
        mirrorArgumentIndices[mirror] = scalarSources.size();
        scalarSources.push_back(mirror);
      }
      for (BoundInfo &bound : bounds)
        bound.useMirrorExpression = !bound.mirrorLeaves.empty();
    }

    // Bounds that are neither constants nor reconstructible from proven-safe
    // mirrors retain the conservative host-to-device staging path.
    for (auto [index, bound] : llvm::enumerate(bounds)) {
      if (bound.constant || bound.useMirrorExpression)
        continue;
      bound.scalarArgumentIndex = scalarSources.size();
      scalarSources.push_back(boundValues[index]);
    }

    // The memcpy size is expressed in bytes. Sub-byte integer types still
    // occupy at least one byte in host memory.
    uint64_t scalarByteWidth = (scalarBitWidth + 7) / 8;

    ModuleOp module = loop->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    Location location =
        FusedLoc::get(loop->getContext(), {loop->getLoc(), wrapper.getLoc()});
    auto scalarTensorType = RankedTensorType::get({}, scalarType);
    // PJRT buffers have a physical one-element shape. Use the usual dynamic
    // rank-1 wrapper type so argument refinement can discover [1] from the
    // runtime byte buffer and erase its shape barrier. Then reshape to a
    // rank-0 tensor inside StableHLO for scalar arithmetic. A rank-0 wrapper
    // type would instead ask shape refinement to reconcile [1] with [].
    auto scalarBufferTensorType =
        RankedTensorType::get({ShapedType::kDynamic}, scalarType);

    SmallVector<Type> megakernelTypes(scalarSources.size(),
                                      scalarBufferTensorType);
    megakernelTypes.append(function.getArgumentTypes().begin(),
                           function.getArgumentTypes().end());
    auto megakernelType =
        FunctionType::get(loop->getContext(), megakernelTypes, megakernelTypes);
    std::string name = getUniqueMegakernelName(module, nextMegakernelId);

    func::FuncOp megakernel;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(module.getBody());
      megakernel =
          func::FuncOp::create(rewriter, location, name, megakernelType);
      megakernel.setPrivate();
      Block *entry = megakernel.addEntryBlock();
      rewriter.setInsertionPointToEnd(entry);

      SmallVector<Value, 3> hloBounds(3);
      SmallVector<Value> hloScalarArguments;
      for (Value argument :
           entry->getArguments().take_front(scalarSources.size()))
        hloScalarArguments.push_back(stablehlo::ReshapeOp::create(
            rewriter, location, scalarTensorType, argument));
      DenseMap<Value, Value> mirrorArguments;
      for (auto [mirror, argumentIndex] : mirrorArgumentIndices)
        mirrorArguments[mirror] = hloScalarArguments[argumentIndex];
      DenseMap<Value, Value> raisedBoundValues;
      for (unsigned index = 0; index < boundValues.size(); ++index) {
        BoundInfo &bound = bounds[index];
        if (bound.constant) {
          hloBounds[index] = makeStableHLOIntegerConstant(
              rewriter, location, scalarTensorType, *bound.constant);
          continue;
        }
        if (bound.useMirrorExpression) {
          hloBounds[index] = raiseMirroredBoundExpression(
              rewriter, boundValues[index], scalarTensorType,
              mirrorArguments, raisedBoundValues);
          assert(hloBounds[index] &&
                 "validated mirrored bound expression failed to raise");
          continue;
        }
        hloBounds[index] = hloScalarArguments[*bound.scalarArgumentIndex];
      }

      SmallVector<Value> initialState(hloBounds.begin(), hloBounds.end());
      llvm::append_range(initialState, entry->getArguments().drop_front(
                                           scalarSources.size()));

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
          loop.getUnsignedCmp() ? stablehlo::ComparisonType::UNSIGNED
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
      // Scalar inputs are read-only. In particular, do not write the final
      // induction value back into a persistent device mirror.
      SmallVector<Value> results;
      llvm::append_range(
          results, entry->getArguments().take_front(scalarSources.size()));
      llvm::append_range(results, whileOp.getResults().drop_front(3));
      func::ReturnOp::create(rewriter, location, results);
    }

    rewriter.setInsertionPoint(loop.getOperation());
    SmallVector<Value> newWrapperInputs;
    SmallVector<Value, 3> boundAllocations;
    auto hostMemrefType = MemRefType::get({1}, scalarType);
    auto deviceMemrefType =
        MemRefType::get({1}, scalarType, MemRefLayoutAttrInterface{},
                        rewriter.getI64IntegerAttr(1));

    // XLA wrapper operands must refer to allocations registered with the XLA
    // runtime. Reuse an explicitly associated device mirror where possible;
    // otherwise stage the dynamic host scalar through a temporary allocation.
    Value copySize;
    if (llvm::any_of(bounds, [](const BoundInfo &bound) {
          return bound.scalarArgumentIndex.has_value();
        }))
      copySize = arith::ConstantIndexOp::create(
          rewriter, location, static_cast<int64_t>(scalarByteWidth));
    for (Value scalarSource : scalarSources) {
      if (mirrorArgumentIndices.contains(scalarSource)) {
        auto mirror =
            scalarSource.getDefiningOp<enzymexla::DeviceMirrorOp>();
        assert(mirror && "mirror scalar source must be a device_mirror result");
        Value deviceStorage = mirror.getDevice();
        if (auto pointerType =
                dyn_cast<LLVM::LLVMPointerType>(deviceStorage.getType())) {
          Attribute memorySpace;
          if (pointerType.getAddressSpace() != 0)
            memorySpace =
                rewriter.getI64IntegerAttr(pointerType.getAddressSpace());
          auto mirrorMemrefType = MemRefType::get(
              {ShapedType::kDynamic}, scalarType,
              MemRefLayoutAttrInterface{}, memorySpace);
          deviceStorage = enzymexla::Pointer2MemrefOp::create(
                              rewriter, location, mirrorMemrefType,
                              deviceStorage)
                              .getResult();
        }
        newWrapperInputs.push_back(deviceStorage);
        continue;
      }

      Value bound = scalarSource;
      if (isa<IndexType>(hostBoundType))
        bound =
            arith::IndexCastOp::create(rewriter, location, scalarType, bound);

      Value hostStorage =
          memref::AllocaOp::create(rewriter, location, hostMemrefType);
      Value zero = arith::ConstantIndexOp::create(rewriter, location, 0);
      memref::StoreOp::create(rewriter, location, bound, hostStorage, zero);
      Value deviceStorage =
          gpu::AllocOp::create(rewriter, location, deviceMemrefType,
                               /*asyncToken=*/(Type) nullptr,
                               /*asyncDependencies=*/ValueRange(),
                               /*dynamicSizes=*/ValueRange(),
                               /*symbolOperands=*/ValueRange())
              .getMemref();
      enzymexla::MemcpyOp::create(rewriter, location,
                                  /*asyncToken=*/(Type) nullptr,
                                  /*asyncDependencies=*/ValueRange(),
                                  deviceStorage, hostStorage, copySize);
      boundAllocations.push_back(deviceStorage);
      newWrapperInputs.push_back(deviceStorage);
    }

    // Recreate the wrapper's pure operand setup just before the replacement
    // call. Values defined outside the loop are reused directly.
    IRMapping mapping;
    for (Operation &operation : body->without_terminator()) {
      if (&operation != wrapper.getOperation())
        rewriter.clone(operation, mapping);
    }
    for (Value input : wrapper.getInputs())
      newWrapperInputs.push_back(mapping.lookupOrDefault(input));

    auto newWrapper = enzymexla::XLAWrapperOp::create(
        rewriter, location, SymbolRefAttr::get(megakernel), newWrapperInputs,
        /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr);
    rewriter.setInsertionPointAfter(newWrapper);
    for (Value allocation : boundAllocations)
      gpu::DeallocOp::create(rewriter, location,
                             /*asyncToken=*/(Type) nullptr,
                             /*asyncDependencies=*/ValueRange(), allocation);
    rewriter.eraseOp(loop.getOperation());
    return success();
  }

private:
  DataFlowSolver &aliasSolver;
  mutable unsigned nextMegakernelId = 0;
};

struct XLAMegakernelizePass
    : public enzyme::impl::XLAMegakernelizePassBase<XLAMegakernelizePass> {
  using XLAMegakernelizePassBase::XLAMegakernelizePassBase;

  void runOnOperation() override {
    DataFlowSolver aliasSolver(DataFlowConfig().setInterprocedural(false));
    dataflow::loadBaselineAnalyses(aliasSolver);
    aliasSolver.load<enzyme::AliasAnalysis>(&getContext());
    aliasSolver.load<enzyme::PointsToPointerAnalysis>();
    if (failed(aliasSolver.initializeAndRun(getOperation()))) {
      signalPassFailure();
      return;
    }

    RewritePatternSet fusionPatterns(&getContext());
    fusionPatterns.add<FuseSequentialXLAWrappers>(&getContext(), aliasSolver);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(fusionPatterns)))) {
      signalPassFailure();
      return;
    }

    // Run loop lifting after sequential fusion so an scf.for containing the
    // ping-pong pair becomes a single wrapper before we inspect its body.
    RewritePatternSet loopPatterns(&getContext());
    loopPatterns.add<LiftXLAWrapperLoop>(&getContext(), aliasSolver);
    if (failed(applyPatternsGreedily(getOperation(), std::move(loopPatterns))))
      signalPassFailure();
  }
};

} // namespace
