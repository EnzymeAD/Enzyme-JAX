#include "src/enzyme_ad/jax/Passes/AutoBatching.h"

#include "Enzyme/MLIR/Dialect/Dialect.h"
#include "Enzyme/MLIR/Passes/EnzymeBatchPass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/InliningUtils.h"
#include "src/enzyme_ad/jax/Implementations/WhileLoopInfo.h"
#include "src/enzyme_ad/jax/Passes/AlwaysInliner.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Passes/StructuredTensors.h"
#include "src/enzyme_ad/jax/Utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "auto-batching"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_AUTOBATCHINGPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

static int64_t concatReshapeToBatchCounter = 0;
static int64_t sliceToBatchCount = 0;

// This function checks if any 2 ops in the list are data-dependent on each
// other. We exploit the fact that while traversing the dep graph if we are at a
// position before the other ops in the set, we know that the other ops are not
// data dependent.
bool anyOpsAreDataDependent(ArrayRef<Operation *> ops) {
  if (ops.size() <= 1)
    return false;

  // ops are sorted based on ordering of slices. we need to sort here based on
  // op ordering
  SmallVector<Operation *> sortedOps(ops.begin(), ops.end());
  std::sort(sortedOps.begin(), sortedOps.end(),
            [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });

  SmallPtrSet<Operation *, 8> subsetOps(ops.begin(), ops.end());
  Block *parentBlock = ops[0]->getBlock();

  // For each op, we only need to check if it depends on any earlier op in our
  // subset We can use a worklist approach but only traverse backwards in
  // program order
  for (int i = 1; i < sortedOps.size(); ++i) {
    Operation *laterOp = sortedOps[i];

    // Track all operations this op transitively depends on
    SmallPtrSet<Operation *, 16> dependencies;
    SmallVector<Operation *> worklist;

    // Start with direct operands
    for (Value operand : laterOp->getOperands()) {
      Operation *definingOp = operand.getDefiningOp();
      if (definingOp && definingOp->getBlock() == parentBlock) {
        if (dependencies.insert(definingOp).second) {
          worklist.push_back(definingOp);
        }
      }
    }

    // Expand transitively, but only for ops that come before laterOp
    while (!worklist.empty()) {
      Operation *curr = worklist.pop_back_val();

      // Early termination: if we found a dependency in our subset, we're done
      if (subsetOps.contains(curr)) {
        return true;
      }

      // Only explore dependencies of operations that come before laterOp
      if (!curr->isBeforeInBlock(laterOp))
        continue;

      for (Value operand : curr->getOperands()) {
        Operation *definingOp = operand.getDefiningOp();
        if (definingOp && definingOp->getBlock() == parentBlock &&
            definingOp->isBeforeInBlock(laterOp) &&
            dependencies.insert(definingOp).second) {
          worklist.push_back(definingOp);
        }
      }
    }
  }

  return false;
}

func::FuncOp createWrapperUnbatchedFunction(PatternRewriter &rewriter,
                                            const std::string &funcName,
                                            ArrayRef<Value> operands,
                                            ArrayRef<int32_t> operandIndexMap,
                                            Operation *templateOp) {
  OpBuilder::InsertionGuard guard(rewriter);
  auto modOp = templateOp->getParentOfType<mlir::ModuleOp>();
  if (!modOp)
    return nullptr;
  rewriter.setInsertionPointToStart(modOp.getBody());

  SmallVector<mlir::Type> argTypes;
  for (auto v : operands) {
    auto vType = cast<RankedTensorType>(v.getType());
    auto shape = vType.getShape();
    SmallVector<int64_t> argShape;
    for (int i = 1; i < shape.size(); i++)
      argShape.push_back(shape[i]);
    argTypes.push_back(RankedTensorType::get(argShape, vType.getElementType()));
  }

  FunctionType calleeType =
      rewriter.getFunctionType(argTypes, {templateOp->getResult(0).getType()});
  func::FuncOp func =
      rewriter.create<func::FuncOp>(templateOp->getLoc(), funcName, calleeType);
  func.setPrivate();

  auto &entryBlock = *func.addEntryBlock();
  rewriter.setInsertionPointToStart(&entryBlock);

  IRMapping mapper;
  int batchArgIndex = 0;
  for (int i = 0; i < templateOp->getNumOperands(); i++) {
    Value originalOperand = templateOp->getOperand(i);

    auto copyConstOp = originalOperand.getDefiningOp<stablehlo::ConstantOp>();
    if (operandIndexMap[i] == -1 && copyConstOp) {
      // This is a constant - clone it directly in the function
      auto clonedConst = rewriter.clone(*copyConstOp);
      mapper.map(originalOperand, clonedConst->getResult(0));
    } else {
      // Map to corresponding function argument
      mapper.map(originalOperand, entryBlock.getArguments()[batchArgIndex++]);
    }
  }

  auto unbatchedOp = rewriter.clone(*templateOp, mapper);
  rewriter.create<func::ReturnOp>(templateOp->getLoc(),
                                  ValueRange(unbatchedOp->getResult(0)));

  return func;
}

std::tuple<SmallVector<Value>, SmallVector<int32_t>>
constructAndExtractBatchOperands(PatternRewriter &rewriter,
                                 ArrayRef<Operation *> batchOps, Location loc,
                                 BatchOperandConstructionInfo batchInfo) {
  SmallVector<Value> operands;
  SmallVector<int32_t> operandIndexMap;

  // Analyze operands to find equivalences
  for (int i = 0; i < batchOps[0]->getNumOperands(); i++) {
    SmallVector<Value> currentOperands;
    bool allEquivalent = true;
    auto firstOperand = batchOps[0]->getOperand(i);
    auto firstDefOp = firstOperand.getDefiningOp();

    for (auto v : batchOps) {
      auto currentOperand = v->getOperand(i);
      currentOperands.push_back(currentOperand);

      // Check equivalence with first operand
      if (firstDefOp) {
        if (auto currentDefOp = currentOperand.getDefiningOp()) {
          if (!OperationEquivalence::isEquivalentTo(
                  firstDefOp, currentDefOp,
                  OperationEquivalence::ignoreValueEquivalence, nullptr,
                  OperationEquivalence::IgnoreLocations, nullptr)) {
            allEquivalent = false;
          }
        } else {
          allEquivalent = false;
        }
      } else {
        // Handle block arguments or other cases
        if (firstOperand != currentOperand) {
          allEquivalent = false;
        }
      }
    }

    auto constOp = firstOperand.getDefiningOp<stablehlo::ConstantOp>();
    if (allEquivalent && constOp) {
      // Mark as constant (no batch operand needed)
      operandIndexMap.push_back(-1);
    } else if (i == batchInfo.sliceOperandIndex) {
      auto sliceOp = batchInfo.sliceOp;
      assert(sliceOp);
      assert(batchInfo.sliceDim >= 0);

      auto sliceOpType = cast<RankedTensorType>(sliceOp.getResult().getType());
      auto sliceOpShape = sliceOpType.getShape();
      assert(sliceOpShape[batchInfo.sliceDim] == 1);

      if (batchInfo.intermediateReshape) {
        // in this case we just need a transpose
        SmallVector<int64_t> permutation(sliceOpShape.size(), -1);
        for (int i = 0; i < sliceOpShape.size(); i++) {
          if (i < batchInfo.sliceDim)
            permutation[i + 1] = i;
          else if (i > batchInfo.sliceDim)
            permutation[i] = i;
          else
            permutation[0] = i;
        }
        auto transposeOp = rewriter.create<stablehlo::TransposeOp>(
            loc, sliceOp.getOperand(),
            rewriter.getDenseI64ArrayAttr(permutation));
        operands.push_back(transposeOp.getResult());
      } else {
        SmallVector<int64_t> bcastShape;
        SmallVector<int64_t> mapping(sliceOpShape.size(), -1);
        bcastShape.push_back(batchInfo.nbatches);
        for (int i = 0; i < sliceOpShape.size(); i++) {
          if (i != batchInfo.sliceDim) {
            bcastShape.push_back(sliceOpShape[i]);
            mapping[i] = i + 1;
          } else {
            bcastShape.push_back(1);
            mapping[i] = 0;
          }
        }
        auto bcastOp = rewriter.create<stablehlo::BroadcastInDimOp>(
            loc,
            RankedTensorType::get(bcastShape, sliceOpType.getElementType()),
            sliceOp.getOperand(), rewriter.getDenseI64ArrayAttr(mapping));
        operands.push_back(bcastOp.getResult());
      }

      operandIndexMap.push_back(operands.size() - 1);
    } else {
      // Non-equivalent operands - need to concatenate them
      SmallVector<Value> newConcatOperands;
      for (Value operand : currentOperands) {
        auto inputType = cast<RankedTensorType>(operand.getType());
        auto inputShape = inputType.getShape();

        SmallVector<int64_t> outputShape;
        outputShape.push_back(1);
        outputShape.append(inputShape.begin(), inputShape.end());

        // expand the batch dim (== 0) or move the batch dim from `batchDim` to
        // `0`
        auto newReshapeOp = rewriter.create<stablehlo::ReshapeOp>(
            loc, RankedTensorType::get(outputShape, inputType.getElementType()),
            operand);
        newConcatOperands.push_back(newReshapeOp.getResult());
      }

      auto newConcatOp =
          rewriter.create<stablehlo::ConcatenateOp>(loc, newConcatOperands, 0);

      operandIndexMap.push_back(operands.size());
      operands.push_back(newConcatOp.getResult());
    }
  }

  return std::make_tuple(operands, operandIndexMap);
}

std::tuple<bool, bool> allSameBool(const SmallVector<bool> &bools) {
  return {
      llvm::all_of(bools, [&](bool b) { return b == bools.front(); }),
      bools.front(),
  };
}

bool allOpsAreUnique(const SmallVector<Operation *> &ops) {
  SmallPtrSet<Operation *, 8> seen;
  return llvm::all_of(ops,
                      [&](Operation *op) { return seen.insert(op).second; });
}

static SliceInfo<stablehlo::SliceOp>
defaultUnsupportedSliceInfo(stablehlo::SliceOp sliceOp) {
  return SliceInfo<stablehlo::SliceOp>(sliceOp, {}, {}, {}, -1, -1, false);
}

SliceInfo<stablehlo::SliceOp> constructSliceInfo(stablehlo::SliceOp sliceOp) {
  auto startIndices = llvm::to_vector(sliceOp.getStartIndices());
  auto limitIndices = llvm::to_vector(sliceOp.getLimitIndices());
  auto strides = llvm::to_vector(sliceOp.getStrides());

  if (!llvm::all_of(strides, [](int64_t i) { return i == 1; }))
    return defaultUnsupportedSliceInfo(sliceOp);

  auto inputType = cast<RankedTensorType>(sliceOp.getOperand().getType());
  auto inputShape = llvm::to_vector(inputType.getShape());

  bool supported = true, found = false;
  int64_t sliceDim, sliceStart;

  for (size_t i = 0; i < startIndices.size(); ++i) {
    if (startIndices[i] == limitIndices[i] - 1 &&
        !(startIndices[i] == 0 && limitIndices[i] == inputShape[i])) {
      if (found) {
        // multiple singleton slices not supported
        supported = false;
        break;
      }
      sliceDim = i;
      sliceStart = startIndices[i];
      found = true;
    } else { // For all other dims, must be a full slice
      if (!(startIndices[i] == 0 && limitIndices[i] == inputShape[i] &&
            strides[i] == 1)) {
        supported = false;
        break;
      }
    }
  }
  if (!found) { // If no singleton found, not supported
    supported = false;
  }

  if (!supported)
    return defaultUnsupportedSliceInfo(sliceOp);

  return SliceInfo<stablehlo::SliceOp>(sliceOp, {}, startIndices, inputShape,
                                       sliceDim, sliceStart, true);
}

bool areSlicesContiguous(SmallVector<SliceInfo<stablehlo::SliceOp>> &slices) {
  if (slices.empty())
    return false;

  int64_t sliceDim = slices[0].sliceDim;
  for (const auto &slice : slices) {
    if (slice.sliceDim != sliceDim)
      return false;
    if (!slice.supported)
      return false;
  }

  // Sort slices by start index
  llvm::sort(slices, [](const SliceInfo<stablehlo::SliceOp> &a,
                        const SliceInfo<stablehlo::SliceOp> &b) {
    return a.sliceStart < b.sliceStart;
  });

  int64_t expectedStart = slices[0].sliceStart;
  // TODO: partial slice support
  if (expectedStart != 0)
    return false;

  for (const auto &slice : slices) {
    if (slice.sliceStart != expectedStart)
      return false;
    expectedStart++;
  }

  // TODO: we should support cases where the entire batch is not being sliced
  //       but introducing an intermediate slice.
  if (expectedStart !=
      slices[0].sliceOp.getOperand().getType().getShape()[sliceDim])
    return false;

  return true;
}

LogicalResult batchOperationAndInline(PatternRewriter &rewriter,
                                      enzyme::BatchOp batchOp,
                                      func::FuncOp func) {
  auto funcOpInterface = cast<FunctionOpInterface>(func.getOperation());
  auto key = enzyme::batchutils::BatchCacheKey{
      funcOpInterface, llvm::to_vector(batchOp.getBatchShape())};
  auto modOp = batchOp->getParentOfType<ModuleOp>();
  if (!modOp)
    return rewriter.notifyMatchFailure(batchOp, "parent module not found");

  std::map<enzyme::batchutils::BatchCacheKey, FunctionOpInterface>
      batchedFunctionCache;
  auto batchingResult = enzyme::batchutils::batchOperation(
      rewriter, batchOp, funcOpInterface, batchedFunctionCache);
  if (failed(batchingResult))
    return batchingResult;

  // If calling this function, we assume that we can erase the unbatched
  // function.
  rewriter.eraseOp(func);

  // Inline the batched function
  auto batchedFuncOp = batchedFunctionCache[key];
  auto symbolUses = SymbolTable::getSymbolUses(batchedFuncOp, modOp);
  if (symbolUses) {
    SmallVector<Operation *> callSites;
    for (auto &use : *symbolUses)
      callSites.push_back(use.getUser());

    bool hasRemainingUses = false;
    for (Operation *callSite : callSites) {
      if (auto callOp = dyn_cast<func::CallOp>(callSite)) {
        alwaysInlineCall(callOp);
      } else {
        hasRemainingUses = true;
      }
    }

    if (!hasRemainingUses)
      rewriter.eraseOp(batchedFuncOp);
  }

  return success();
}

LogicalResult ConcatInsertDimToBatchBase::matchAndRewriteImpl(
    stablehlo::ConcatenateOp concatOp, PatternRewriter &rewriter) const {
  if (concatOp.getNumOperands() <= 1)
    return failure();

  auto concatDim = concatOp.getDimension();
  auto concatType = cast<RankedTensorType>(concatOp.getResult().getType());
  auto concatShape = concatType.getShape();

  SmallVector<Operation *> concatOpOperands;

  for (auto [i, v] : llvm::enumerate(concatOp.getOperands())) {
    auto definingOp = v.getDefiningOp();
    if (!definingOp)
      return rewriter.notifyMatchFailure(concatOp, "operand is not a valid op");

    bool isReshapeOpInsertDim =
        validReshapeOpInsertDimForBatching(definingOp, concatDim);

    bool isBroadcastInDimOpInsertDim = false;
    if (!isReshapeOpInsertDim)
      isBroadcastInDimOpInsertDim =
          validBroadcastInDimOpInsertDimForBatching(definingOp, concatDim);

    if (!isReshapeOpInsertDim && !isBroadcastInDimOpInsertDim)
      return rewriter.notifyMatchFailure(concatOp, "operand is not a valid op");

    auto vdefOp = isValidTargetOp(definingOp->getOperand(0).getDefiningOp());
    if (!vdefOp)
      return rewriter.notifyMatchFailure(concatOp, "not a valid target op");

    if (concatOpOperands.size() != 0) {
      if (!OperationEquivalence::isEquivalentTo(
              concatOpOperands[0], vdefOp,
              OperationEquivalence::ignoreValueEquivalence, nullptr,
              OperationEquivalence::IgnoreLocations, nullptr))
        return rewriter.notifyMatchFailure(concatOp,
                                           "op is not equivalent to first");
    }

    if (!isOnlyUsedInOperation(vdefOp, definingOp))
      return rewriter.notifyMatchFailure(concatOp,
                                         "op is not only used in reshape op");

    concatOpOperands.push_back(vdefOp);
  }

  auto [batchOpOperands, operandIndexMap] = constructAndExtractBatchOperands(
      rewriter, concatOpOperands, concatOp.getLoc(),
      BatchOperandConstructionInfo{nullptr, -1, -1, -1, false});

  std::string wrapperFuncName = "enzymexla_unbatched_ConcatInsertDimToBatch_" +
                                (std::to_string(concatReshapeToBatchCounter++));

  func::FuncOp func =
      createWrapperUnbatchedFunction(rewriter, wrapperFuncName, batchOpOperands,
                                     operandIndexMap, concatOpOperands[0]);
  if (!func)
    return rewriter.notifyMatchFailure(concatOp,
                                       "failed to create wrapper function");

  SmallVector<int64_t> outputShape;
  outputShape.push_back(concatShape[concatDim]);
  for (int i = 0; i < concatShape.size(); i++) {
    if (i == concatDim)
      continue;
    outputShape.push_back(concatShape[i]);
  }

  auto batchOp = rewriter.create<enzyme::BatchOp>(
      concatOp.getLoc(),
      RankedTensorType::get(outputShape, concatType.getElementType()),
      mlir::FlatSymbolRefAttr::get(concatOp.getContext(), wrapperFuncName),
      ValueRange(batchOpOperands),
      rewriter.getDenseI64ArrayAttr({concatShape[concatDim]}));

  SmallVector<int64_t> permutation;
  for (int i = 1; i <= concatDim; i++)
    permutation.push_back(i);
  permutation.push_back(0);
  for (int i = concatDim + 1; i < concatShape.size(); i++)
    permutation.push_back(i);

  rewriter.replaceOpWithNewOp<stablehlo::TransposeOp>(
      concatOp, batchOp->getResult(0), permutation);

  return batchOperationAndInline(rewriter, batchOp, func);
}

bool ConcatInsertDimToBatchBase::validReshapeOpInsertDimForBatching(
    Operation *op, int64_t dim) const {
  auto reshapeOp = dyn_cast<stablehlo::ReshapeOp>(op);
  if (!reshapeOp)
    return false;

  return areValidInsertionDims(
      cast<RankedTensorType>(reshapeOp.getOperand().getType()),
      cast<RankedTensorType>(reshapeOp.getResult().getType()), {dim});
}

bool ConcatInsertDimToBatchBase::validBroadcastInDimOpInsertDimForBatching(
    Operation *op, int64_t dim) const {
  auto broadcastInDimOp = dyn_cast<stablehlo::BroadcastInDimOp>(op);
  if (!broadcastInDimOp)
    return false;

  auto inputType =
      cast<RankedTensorType>(broadcastInDimOp.getOperand().getType());
  auto outputType =
      cast<RankedTensorType>(broadcastInDimOp.getResult().getType());

  // single insert dim
  if (inputType.getRank() != outputType.getRank() - 1)
    return false;

  // If concat dim is present in broadcast dims, then it is not a valid insert
  auto broadcastInDimOpDims =
      llvm::to_vector(broadcastInDimOp.getBroadcastDimensions());
  for (auto bDim : broadcastInDimOpDims) {
    if (bDim == dim)
      return false;
  }

  // Broadcast dims must be sorted
  if (!llvm::is_sorted(broadcastInDimOpDims))
    return false;

  // insert dim must be of size 1
  return outputType.getShape()[dim] == 1;
}

LogicalResult
SliceToBatchBase::matchAndRewriteImpl(stablehlo::SliceOp sliceOp,
                                      PatternRewriter &rewriter) const {
  Value sliceInput = sliceOp.getOperand();
  // Find all slices of the same input that feed into equivalent operations
  SmallVector<SliceInfo<stablehlo::SliceOp>> relatedSlices;
  SmallVector<Operation *> relatedOps;
  SmallVector<bool> allHaveIntermediateReshapes;
  Operation *targetOp = nullptr;
  int64_t sliceOperandIndex = -1;

  // Build worklist of all slice operations on the same input
  for (auto [idx, user] : llvm::enumerate(sliceInput.getUsers())) {
    auto candidateSlice = dyn_cast<stablehlo::SliceOp>(user);
    if (!candidateSlice || !candidateSlice.getResult().hasOneUse())
      continue;

    auto sliceInfo = constructSliceInfo(candidateSlice);

    Operation *onlyUser = *candidateSlice.getResult().getUsers().begin();

    bool isIntermediateReshape = false;
    auto candidateTargetOp = isValidTargetOp(onlyUser);

    Operation *preceedingOp = candidateSlice;
    if (!candidateTargetOp) {
      // check for reshape
      auto intermediateReshape = dyn_cast<stablehlo::ReshapeOp>(onlyUser);
      if (!intermediateReshape)
        continue;

      auto reshapeInputShape =
          cast<RankedTensorType>(intermediateReshape.getOperand().getType());
      auto reshapeOutputShape =
          cast<RankedTensorType>(intermediateReshape.getResult().getType());

      if (!areValidInsertionDims(reshapeOutputShape, reshapeInputShape,
                                 {sliceInfo.sliceDim}))
        continue;

      if (!intermediateReshape.getResult().hasOneUse())
        continue;

      isIntermediateReshape = true;
      candidateTargetOp =
          isValidTargetOp(*intermediateReshape.getResult().getUsers().begin());
      if (!candidateTargetOp)
        continue;
      preceedingOp = intermediateReshape;
    }

    if (targetOp) {
      if (!OperationEquivalence::isEquivalentTo(
              targetOp, candidateTargetOp,
              OperationEquivalence::ignoreValueEquivalence, nullptr,
              OperationEquivalence::IgnoreLocations, nullptr)) {
        continue;
      }
    } else {
      for (auto [i, opOperand] :
           llvm::enumerate(candidateTargetOp->getOperands())) {
        if (opOperand == preceedingOp->getResult(0)) {
          sliceOperandIndex = i;
          break;
        }
      }
      targetOp = candidateTargetOp;
    }

    relatedSlices.push_back(sliceInfo);
    relatedOps.push_back(candidateTargetOp);
    allHaveIntermediateReshapes.push_back(isIntermediateReshape);
  }

  if (relatedSlices.size() <= 1)
    return rewriter.notifyMatchFailure(sliceOp, "no related slices found");

  // Check that all related slices and ops are in the same block
  // TODO: we can run this pass by constructing common subsets and optimizing
  // each of those
  Block *commonBlock = relatedSlices[0].sliceOp->getBlock();
  for (const auto &sliceInfo : relatedSlices) {
    if (sliceInfo.sliceOp->getBlock() != commonBlock)
      return rewriter.notifyMatchFailure(
          sliceOp, "not all related slices in same block");
  }
  for (Operation *op : relatedOps) {
    if (op->getBlock() != commonBlock)
      return rewriter.notifyMatchFailure(sliceOp,
                                         "not all related ops in same block");
  }

  // Sort all three vectors together based on sliceStart
  SmallVector<size_t> indices(relatedSlices.size());
  std::iota(indices.begin(), indices.end(), 0);

  std::sort(indices.begin(), indices.end(), [&](size_t i, size_t j) {
    return relatedSlices[i].sliceStart < relatedSlices[j].sliceStart;
  });

  // Reorder all three vectors according to the sorted indices
  SmallVector<SliceInfo<stablehlo::SliceOp>> sortedSlices;
  SmallVector<Operation *> sortedOps;
  SmallVector<bool> sortedReshapes;

  for (size_t idx : indices) {
    sortedSlices.push_back(relatedSlices[idx]);
    sortedOps.push_back(relatedOps[idx]);
    sortedReshapes.push_back(allHaveIntermediateReshapes[idx]);
  }

  relatedSlices = std::move(sortedSlices);
  relatedOps = std::move(sortedOps);
  allHaveIntermediateReshapes = std::move(sortedReshapes);

  // Validate that slices are compatible for batching
  if (!areSlicesContiguous(relatedSlices))
    return rewriter.notifyMatchFailure(sliceOp,
                                       "slices not compatible for batching");

  if (!allOpsAreUnique(relatedOps))
    return rewriter.notifyMatchFailure(sliceOp, "ops are not unique");

  auto [validReshapes, intermediateReshape] =
      allSameBool(allHaveIntermediateReshapes);
  if (!validReshapes)
    return rewriter.notifyMatchFailure(
        sliceOp, "not all ops have same intermediate reshape");

  if (anyOpsAreDataDependent(relatedOps))
    return rewriter.notifyMatchFailure(sliceOp, "ops are data dependent");

  assert(sliceOperandIndex >= 0);

  // move all the related ops and their operands (and theirs) s.t they are
  // together
  Operation *firstRelatedOp = relatedOps[0];
  for (auto &op : relatedOps) {
    if (op->isBeforeInBlock(firstRelatedOp))
      firstRelatedOp = op;
  }
  // if we have to move around too many ops we avoid applying this pattern
  llvm::SmallSetVector<Operation *, 8> opsToMoveWorklist;
  for (auto &op : relatedOps) {
    if (op == firstRelatedOp)
      continue;
    SmallVector<Operation *> opsToMove;
    opsToMove.push_back(op);
    while (!opsToMove.empty()) {
      auto currOp = opsToMove.pop_back_val();
      bool notSeen = opsToMoveWorklist.insert(currOp);
      if (notSeen) {
        for (auto operand : currOp->getOperands()) {
          if (auto operandOp = operand.getDefiningOp()) {
            if (!operandOp->isBeforeInBlock(firstRelatedOp)) {
              opsToMove.push_back(operandOp);
            }
          }
        }
      }
    }
  }

  SmallVector<Operation *> opsToMoveWorklistVec(opsToMoveWorklist.begin(),
                                                opsToMoveWorklist.end());

  std::sort(opsToMoveWorklistVec.begin(), opsToMoveWorklistVec.end(),
            [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
  for (auto op : opsToMoveWorklistVec) {
    rewriter.moveOpAfter(op, firstRelatedOp);
    firstRelatedOp = op;
  }

  Operation *insertionPoint = relatedOps[0];
  for (auto &op : relatedOps) {
    if (op->isBeforeInBlock(insertionPoint))
      continue;
    insertionPoint = op;
  }
  rewriter.setInsertionPoint(insertionPoint);

  auto [batchOpOperands, operandIndexMap] = constructAndExtractBatchOperands(
      rewriter, relatedOps, sliceOp.getLoc(),
      BatchOperandConstructionInfo{
          relatedSlices[0].sliceOp, static_cast<int32_t>(sliceOperandIndex),
          static_cast<int32_t>(relatedSlices[0].sliceDim),
          static_cast<int32_t>(relatedSlices.size()), intermediateReshape});

  std::string sliceToBatchName =
      "enzymexla_unbatched_SliceToBatch_" + std::to_string(sliceToBatchCount++);

  SmallVector<int64_t> retShape;

  func::FuncOp func = createWrapperUnbatchedFunction(
      rewriter, sliceToBatchName, batchOpOperands, operandIndexMap,
      relatedOps[0]);
  if (!func)
    return rewriter.notifyMatchFailure(sliceOp, "failed to create function");

  SmallVector<int64_t> outputShape;
  outputShape.push_back(relatedSlices.size());
  auto relatedOpsType =
      cast<RankedTensorType>(relatedOps[0]->getResult(0).getType());
  auto funcRetShape = relatedOpsType.getShape();
  outputShape.append(funcRetShape.begin(), funcRetShape.end());

  auto batchOp = rewriter.create<enzyme::BatchOp>(
      sliceOp.getLoc(),
      RankedTensorType::get(outputShape, relatedOpsType.getElementType()),
      mlir::FlatSymbolRefAttr::get(sliceOp.getContext(), sliceToBatchName),
      ValueRange(batchOpOperands),
      rewriter.getDenseI64ArrayAttr(
          {static_cast<int64_t>(relatedSlices.size())}));

  SmallVector<int64_t> startIndices(outputShape.size(), 0);
  SmallVector<int64_t> endIndices;
  endIndices.append(outputShape.begin(), outputShape.end());
  SmallVector<int64_t> strides(outputShape.size(), 1);
  for (auto [sliceInfo, otherOp] : llvm::zip(relatedSlices, relatedOps)) {
    startIndices[0] = sliceInfo.sliceStart;
    endIndices[0] = sliceInfo.sliceStart + 1;

    auto slicedOp = rewriter.create<stablehlo::SliceOp>(
        sliceOp.getLoc(), batchOp->getResult(0),
        rewriter.getDenseI64ArrayAttr(startIndices),
        rewriter.getDenseI64ArrayAttr(endIndices),
        rewriter.getDenseI64ArrayAttr(strides));
    rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(
        otherOp, otherOp->getResult(0).getType(), slicedOp);
  }

  return batchOperationAndInline(rewriter, batchOp, func);
}

LogicalResult GreedyWhileLoopBatchFission::matchAndRewriteImpl(
    stablehlo::WhileOp whileOp, PatternRewriter &rewriter) const {
  auto info = WhileLoopInfo(whileOp);
  auto computeInfoSuccess = info.computeInfo();
  if (computeInfoSuccess.failed())
    return computeInfoSuccess;

  if (!info.isValid() || !info.isConstant())
    return failure();

  auto numIters = info.getConstantNumIters();
  if (numIters == 1) // should get unrolled
    return failure();

  int64_t start = info.getConstantStart().value();
  int64_t limit = info.getConstantLimit().value();
  int64_t step = info.getConstantStep().value();

  // TODO: we can support start != 0 for applying a function to part of the
  // tensor
  if (start != 0 || step != 1)
    return failure();

  auto parentBlock = whileOp->getBlock();
  auto &whileBody = whileOp.getBody().front();

  info.propagateInductionVarOffsets();
  DenseMap<Value, APInt> inductionVarOffsets = info.getInductionVarOffsets();

  auto parentFunc = whileOp->getParentOp();
  if (!parentFunc)
    return rewriter.notifyMatchFailure(whileOp, "parent function not found");

  // Find all dynamic slices in the loop body that meet the criteria:
  // 1. All slice variables are outside the loop body
  // 2. Only one variable in the body is a direct descendant of the induction
  // variable
  // 3. The size of that dimension equals the limit
  SmallVector<DynamicSliceInfo> candidateSlices;

  for (auto [value, offset] : inductionVarOffsets) {
    if (!offset.isZero())
      continue;
    for (auto user : value.getUsers()) {
      if (auto sliceOp = dyn_cast<stablehlo::DynamicSliceOp>(user)) {
        auto result = isDynamicSliceValidForBatching(sliceOp, value, limit,
                                                     whileBody, parentBlock);
        if (result.result == IsValidForBatchingResult::VALID) {
          candidateSlices.push_back(
              DynamicSliceInfo{sliceOp, result.sliceDim, false});
        }
      }
    }
  }

  if (candidateSlices.empty())
    return rewriter.notifyMatchFailure(whileOp, "no candidate slices found");

  bool anyOpRewritten = false;

  // iota [idx] where iota starts at 0 and iter var also starts at 0
  // replace this with idx
  // If we do a successful rewrite here, we remove the DynamicSliceInfo from
  // the candidateSlices vector (a later invocation will handle the rest)
  SmallVector<DynamicSliceInfo> retainedSlices;
  for (auto [i, slice] : llvm::enumerate(candidateSlices)) {
    auto iotaDetection = detectIotaLikeTensor(slice.sliceOp.getOperand());
    if (iotaDetection &&
        slice.inductionVarDimension == iotaDetection.value().dimension &&
        iotaDetection.value().start == 0 &&
        iotaDetection.value().limit == limit) {
      anyOpRewritten = true;

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(slice.sliceOp);
      Value newOperand = info.getInductionVariable();
      auto sliceType =
          cast<RankedTensorType>(slice.sliceOp.getResult().getType());
      auto outElemType = sliceType.getElementType();
      if (cast<TensorType>(newOperand.getType()).getElementType() !=
          outElemType) {
        newOperand = rewriter
                         .create<stablehlo::ConvertOp>(
                             slice.sliceOp.getLoc(),
                             RankedTensorType::get({}, outElemType), newOperand)
                         .getResult();
      }
      rewriter.replaceOpWithNewOp<stablehlo::BroadcastInDimOp>(
          slice.sliceOp, sliceType, newOperand,
          rewriter.getDenseI64ArrayAttr({}));
    } else {
      retainedSlices.push_back(slice);
    }
  }
  candidateSlices = std::move(retainedSlices);

  // Create a map of user operations to their corresponding dynamic slices
  DenseMap<Operation *, SmallVector<DynamicSliceInfo>> userOpToSlicesMap;
  for (auto ds : candidateSlices) {
    for (auto op : ds.sliceOp->getUsers()) {
      userOpToSlicesMap[op].push_back(ds);

      if (isa<stablehlo::ReshapeOp>(op)) {
        if (!areValidInsertionDims(
                cast<RankedTensorType>(op->getResult(0).getType()),
                cast<RankedTensorType>(op->getOperand(0).getType()),
                {ds.inductionVarDimension})) {
          continue;
        }

        for (auto user : op->getUsers()) {
          userOpToSlicesMap[user].push_back(
              DynamicSliceInfo{ds.sliceOp, ds.inductionVarDimension, true});
        }
      }
    }
  }

  if (userOpToSlicesMap.empty())
    return anyOpRewritten ? success() : failure();

  for (auto &[op, slices] : userOpToSlicesMap) {
    SmallVector<bool> allIntermediateReshapes(slices.size());
    for (auto [i, slice] : llvm::enumerate(slices))
      allIntermediateReshapes[i] = slice.intermediateReshape;
    auto [validReshapes, intermediateReshape] =
        allSameBool(allIntermediateReshapes);

    if (!validReshapes)
      continue;

    // TODO: add scatter here once batch interface is
    if (isa<stablehlo::DotGeneralOp, stablehlo::GatherOp, stablehlo::ReduceOp,
            stablehlo::SortOp, stablehlo::TransposeOp,
            stablehlo::BroadcastInDimOp, stablehlo::ReduceWindowOp>(op) ||
        op->hasTrait<OpTrait::Elementwise>()) {
      if (liftOperationByBatching(rewriter, whileOp, slices, op, info,
                                  intermediateReshape)) {
        anyOpRewritten = true;
      }
    } else if (!intermediateReshape && isa<stablehlo::ReshapeOp>(op)) {
      if (liftSpecialReshapeOp(rewriter, whileOp, slices,
                               dyn_cast<stablehlo::ReshapeOp>(op), info)) {
        anyOpRewritten = true;
      }
    }
  }

  return anyOpRewritten ? success() : failure();
};

bool GreedyWhileLoopBatchFission::liftSpecialReshapeOp(
    PatternRewriter &rewriter, stablehlo::WhileOp whileOp,
    ArrayRef<DynamicSliceInfo> sliceOps, stablehlo::ReshapeOp reshapeOp,
    WhileLoopInfo info) const {
  auto moduleOp = reshapeOp->getParentOfType<ModuleOp>();

  if (sliceOps.size() != 1)
    return false;

  auto sliceInfo = sliceOps[0];
  auto sliceOp = sliceInfo.sliceOp;

  auto inputType = cast<RankedTensorType>(reshapeOp.getOperand().getType());
  auto outputType = cast<RankedTensorType>(reshapeOp.getType());

  auto deletionDims = findReshapeInsertionDims(outputType, inputType);

  llvm::SmallSet<int64_t, 4> deletionDimsWithoutInductionVarDim;
  bool foundSliceDim = false;
  for (auto dim : deletionDims) {
    if (dim == sliceInfo.inductionVarDimension) {
      foundSliceDim = true;
      continue;
    }
    deletionDimsWithoutInductionVarDim.insert(dim);
  }
  if (!foundSliceDim || deletionDimsWithoutInductionVarDim.empty())
    return false;

  // check that the ds start indices for these dims are all zero
  SmallVector<int64_t> reshapeShape;
  auto sliceOperandType =
      cast<RankedTensorType>(sliceOp.getOperand().getType());
  auto sliceOperandShape = sliceOperandType.getShape();

  auto sliceStartIndices = sliceOp.getStartIndices();
  auto sliceSizes = sliceOp.getSliceSizes();
  SmallVector<Value> newStartIndices;
  SmallVector<int64_t> newSliceSizes;
  for (int32_t i = 0; i < sliceStartIndices.size(); i++) {
    if (deletionDimsWithoutInductionVarDim.contains(i)) {
      if (matchPattern(sliceStartIndices[i], m_Zero()))
        continue;
      return false;
    }
    reshapeShape.push_back(sliceOperandShape[i]);
    newStartIndices.push_back(sliceStartIndices[i]);
    newSliceSizes.push_back(sliceSizes[i]);
  }

  // hoist the reshape op
  rewriter.setInsertionPoint(whileOp);
  auto outsideReshapeOp = rewriter.create<stablehlo::ReshapeOp>(
      whileOp->getLoc(),
      RankedTensorType::get(reshapeShape, inputType.getElementType()),
      sliceOp.getOperand());

  // then reconstruct the slice op with removed start indices
  rewriter.setInsertionPoint(reshapeOp);
  auto newSliceOp = rewriter.create<stablehlo::DynamicSliceOp>(
      whileOp->getLoc(), outsideReshapeOp->getResult(0), newStartIndices,
      rewriter.getDenseI64ArrayAttr(newSliceSizes));

  // introduce another reshape op to only delete the induction var dim
  rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(
      reshapeOp, reshapeOp.getType(), newSliceOp);
  return true;
}

bool GreedyWhileLoopBatchFission::liftOperationByBatching(
    PatternRewriter &rewriter, stablehlo::WhileOp whileOp,
    ArrayRef<DynamicSliceInfo> sliceOps, Operation *op, WhileLoopInfo info,
    bool intermediateReshape) const {
  auto moduleOp = op->getParentOfType<ModuleOp>();

  SmallVector<BatchLiftingMode> batchLiftingModes(op->getNumOperands());
  SmallVector<Value> batchOperands(op->getNumOperands());
  SmallVector<int32_t> sliceDims(op->getNumOperands());
  for (int i = 0; i < op->getNumOperands(); i++) {
    auto operand = op->getOperand(i);
    auto defOp = operand.getDefiningOp();

    if (operand.getParentBlock() != &whileOp.getBody().front()) {
      SplatElementsAttr splat;
      if (defOp && matchPattern(defOp, m_Constant(&splat))) {
        batchLiftingModes[i] = BatchLiftingMode::CONSTANT;
      } else {
        batchLiftingModes[i] = BatchLiftingMode::DEFINED_OUTSIDE_WHILE;
      }
      batchOperands[i] = operand;
      continue;
    }

    if (!defOp)
      return false;

    Operation *dsOp;
    if (intermediateReshape) {
      if (auto reshapeOp = dyn_cast<stablehlo::ReshapeOp>(defOp)) {
        dsOp = reshapeOp.getOperand().getDefiningOp();
      } else {
        return false;
      }
    } else {
      dsOp = defOp;
    }

    if (auto ds = dyn_cast<stablehlo::DynamicSliceOp>(dsOp)) {
      auto itr = llvm::find_if(sliceOps, [&](const DynamicSliceInfo &info) {
        return info.sliceOp == ds;
      });
      if (itr != sliceOps.end()) {
        batchLiftingModes[i] = BatchLiftingMode::DYNAMIC_SLICE;
        sliceDims[i] = itr->inductionVarDimension;
        batchOperands[i] = ds->getOperand(0);
        continue;
      } else {
        return false;
      }
    }

    return false;
  }

  // emit a funcOp that we will batch later
  static int64_t batchCounter = 0;
  std::string batchFnName =
      "enzymexla_loop_batch_fission_" + std::to_string(batchCounter++);
  func::FuncOp func;

  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());

    SmallVector<Type> argTypes;
    for (int i = 0; i < op->getNumOperands(); i++) {
      if (batchLiftingModes[i] == BatchLiftingMode::CONSTANT)
        continue;
      argTypes.push_back(op->getOperand(i).getType());
    }

    FunctionType calleeType =
        rewriter.getFunctionType(TypeRange(argTypes), op->getResultTypes());
    func = rewriter.create<func::FuncOp>(moduleOp.getLoc(), batchFnName,
                                         calleeType);
    func.setPrivate();

    auto &entryBlock = *func.addEntryBlock();
    rewriter.setInsertionPointToStart(&entryBlock);

    IRMapping mapper;
    for (int i = 0; i < op->getNumOperands(); i++) {
      auto operand = op->getOperand(i);
      if (batchLiftingModes[i] == BatchLiftingMode::CONSTANT) {
        auto clonedConst = rewriter.clone(*operand.getDefiningOp());
        mapper.map(operand, clonedConst->getResult(0));
        continue;
      }
      mapper.map(operand, entryBlock.getArguments()[i]);
    }

    auto unbatchedOp = rewriter.clone(*op, mapper);
    rewriter.create<func::ReturnOp>(op->getLoc(), unbatchedOp->getResults());
  }

  rewriter.setInsertionPoint(whileOp);

  SmallVector<Value> newOperands;
  for (auto [consType, baseOp, sliceDim] :
       llvm::zip(batchLiftingModes, batchOperands, sliceDims)) {
    auto operandType = cast<RankedTensorType>(baseOp.getType());
    int operandRank = cast<RankedTensorType>(baseOp.getType()).getRank();
    auto operandShape = operandType.getShape();

    switch (consType) {
    case BatchLiftingMode::DYNAMIC_SLICE: {
      if (intermediateReshape) {
        SmallVector<int64_t> permutation(operandRank);
        permutation[0] = sliceDim;
        for (int i = 0; i < sliceDim; i++)
          permutation[i + 1] = i;
        for (int i = sliceDim + 1; i < operandRank; i++)
          permutation[i] = i;

        auto transposedOperand = rewriter.create<stablehlo::TransposeOp>(
            whileOp->getLoc(), baseOp,
            rewriter.getDenseI64ArrayAttr(permutation));
        newOperands.push_back(transposedOperand->getResult(0));
      } else {
        SmallVector<int64_t> mapping(operandRank);
        for (size_t i = 0; i < sliceDim; i++)
          mapping[i] = i + 1;
        mapping[sliceDim] = 0;
        for (size_t i = sliceDim + 1; i < operandRank; i++)
          mapping[i] = i + 1;

        SmallVector<int64_t> resultShape(operandRank + 1);
        resultShape[0] = info.getConstantLimit().value();
        resultShape[sliceDim + 1] = 1;
        for (size_t i = 0; i < sliceDim; i++)
          resultShape[i + 1] = operandShape[i];
        for (size_t i = sliceDim + 1; i < operandRank; i++)
          resultShape[i + 1] = operandShape[i];

        auto broadcastedOperand = rewriter.create<stablehlo::BroadcastInDimOp>(
            whileOp->getLoc(),
            RankedTensorType::get(resultShape, operandType.getElementType()),
            baseOp, rewriter.getDenseI64ArrayAttr(mapping));
        newOperands.push_back(broadcastedOperand->getResult(0));
      }
      break;
    }
    case BatchLiftingMode::DEFINED_OUTSIDE_WHILE: {
      SmallVector<int64_t> newOperandShape(operandRank + 1);
      newOperandShape[0] = info.getConstantLimit().value();
      for (int i = 0; i < operandRank; i++)
        newOperandShape[i + 1] = operandShape[i];

      SmallVector<int64_t> mapping(operandRank);
      std::iota(mapping.begin(), mapping.end(), 1);

      auto broadcastedOperand = rewriter.create<stablehlo::BroadcastInDimOp>(
          whileOp->getLoc(),
          RankedTensorType::get(newOperandShape, operandType.getElementType()),
          baseOp, rewriter.getDenseI64ArrayAttr(mapping));
      newOperands.push_back(broadcastedOperand->getResult(0));
      break;
    }
    case BatchLiftingMode::CONSTANT: {
      continue; // copied into the function body no need to include in operands
    }
    default: {
      assert(false && "not implemented");
      break;
    }
    }
  }

  auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
  auto resultShape = resultType.getShape();
  SmallVector<int64_t> outputShape(resultShape.size() + 1);
  outputShape[0] = info.getConstantLimit().value();
  for (int i = 0; i < resultShape.size(); i++)
    outputShape[i + 1] = resultShape[i];

  auto inductionVar = info.getInductionVariable();
  auto inductionVarType = cast<RankedTensorType>(inductionVar.getType());

  auto constZero = rewriter.create<stablehlo::ConstantOp>(
      whileOp->getLoc(), inductionVarType,
      cast<ElementsAttr>(makeAttr(inductionVarType, 0)));

  auto batchOp = rewriter.create<enzyme::BatchOp>(
      whileOp->getLoc(),
      RankedTensorType::get(outputShape, resultType.getElementType()),
      mlir::FlatSymbolRefAttr::get(func.getContext(), batchFnName),
      ValueRange(newOperands),
      rewriter.getDenseI64ArrayAttr({info.getConstantLimit().value()}));

  rewriter.setInsertionPointAfter(op);
  SmallVector<Value> dynamicSliceStarts(outputShape.size(), constZero);
  dynamicSliceStarts[0] = info.getInductionVariable();

  SmallVector<int64_t> dynamicSliceSizes(outputShape.size());
  dynamicSliceSizes[0] = 1;
  for (int i = 1; i < outputShape.size(); i++)
    dynamicSliceSizes[i] = outputShape[i];

  auto dynamicSlice = rewriter.create<stablehlo::DynamicSliceOp>(
      whileOp->getLoc(), batchOp->getResult(0), dynamicSliceStarts,
      dynamicSliceSizes);
  rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(
      op, op->getResult(0).getType(), dynamicSlice);

  return succeeded(batchOperationAndInline(rewriter, batchOp, func));
}

GreedyWhileLoopBatchFission::ValidBatchingInfo
GreedyWhileLoopBatchFission::isDynamicSliceValidForBatching(
    stablehlo::DynamicSliceOp sliceOp, Value iterVar, int64_t limit,
    Block &whileBody, Block *parentBlock) const {
  auto startIndices = sliceOp.getStartIndices();
  auto sliceSizes = sliceOp.getSliceSizes();
  auto operand = sliceOp.getOperand();
  auto operandShape = cast<RankedTensorType>(operand.getType()).getShape();

  if (operand.getParentBlock() == &whileBody)
    return ValidBatchingInfo{
        IsValidForBatchingResult::OPERAND_NOT_ACCESSIBLE_FROM_PARENT, -1};

  // Track which start index corresponds to the induction variable descendant
  int32_t inductionVarDimension = -1;

  for (size_t i = 0; i < startIndices.size(); i++) {
    Value startIndex = startIndices[i];

    if (startIndex == iterVar) {
      // Multiple dimensions are descendants of induction var - invalid
      if (inductionVarDimension != -1)
        return ValidBatchingInfo{
            IsValidForBatchingResult::MULTIPLE_INDUCTION_VARIABLE_SLICE_DIMS,
            -1};
      inductionVarDimension = i;

      // Check if the slice size in this dimension equals the limit
      // TODO: relax the limit check at some point
      if (operandShape[i] != limit || sliceSizes[i] != 1) {
        ;
        return ValidBatchingInfo{IsValidForBatchingResult::NOT_FULL_SLICE, -1};
      }

      continue;
    }

    Operation *definingOp = startIndex.getDefiningOp();
    if (!definingOp) {
      // TODO: defining op might be false here since it could be the induction
      // variable itself
      return ValidBatchingInfo{IsValidForBatchingResult::NOT_FULL_SLICE, -1};
    }

    // Check if this start index is defined within the loop body
    if (definingOp->getBlock() != &whileBody) {
      // TODO: we are only considering the full slice case for now. we can
      // generalize this
      if (!matchPattern(definingOp, m_Zero()) ||
          sliceSizes[i] != operandShape[i]) {
        return ValidBatchingInfo{IsValidForBatchingResult::NOT_FULL_SLICE, -1};
      }
      continue;
    }

    return ValidBatchingInfo{IsValidForBatchingResult::NOT_FULL_SLICE, -1};
  }

  // We should have exactly one index from the body, and it should be
  // a descendant of the induction variable
  return ValidBatchingInfo{IsValidForBatchingResult::VALID,
                           inductionVarDimension};
}

struct AutoBatchingPass
    : public enzyme::impl::AutoBatchingPassBase<AutoBatchingPass> {
  using Base::Base;

  void runOnOperation() override {
    auto context = getOperation()->getContext();
    RewritePatternSet patterns(context);

    if (concat_insert_dim_passes) {
      patterns.add<ConcatInsertDimToBatch<stablehlo::DotGeneralOp>,
                   ConcatInsertDimToBatch<stablehlo::GatherOp>,
                   ConcatInsertDimToBatch<stablehlo::IotaOp>,
                   ConcatInsertDimToBatch<stablehlo::ReduceOp>,
                   // ConcatInsertDimToBatch<stablehlo::ScatterOp>, after batch
                   // op interface is implemented
                   ConcatInsertDimToBatch<stablehlo::SortOp>,
                   ConcatInsertDimToBatch<stablehlo::ReduceWindowOp>,
                   ConcatInsertDimElementwiseToBatch>(context);
    }

    if (slice_to_batch_passes) {
      patterns.add<
          SliceToBatch<stablehlo::DotGeneralOp>,
          SliceToBatch<stablehlo::GatherOp>, SliceToBatch<stablehlo::IotaOp>,
          SliceToBatch<stablehlo::ReduceOp>, SliceToBatch<stablehlo::SortOp>,
          SliceToBatch<stablehlo::TransposeOp>,
          SliceToBatch<stablehlo::BroadcastInDimOp>,
          SliceToBatch<stablehlo::ReduceWindowOp>,
          // SliceToBatchReshape,
          SliceToBatchElementwise>(context);
    }

    if (while_loop_batching_mode == "greedy") {
      patterns.add<GreedyWhileLoopBatchFission>(context);
    } else if (while_loop_batching_mode != "none") {
      llvm::errs() << "Unknown while loop batching mode: "
                   << while_loop_batching_mode << "\n";
      signalPassFailure();
    }

    GreedyRewriteConfig config;
    config.setMaxIterations(max_iterations);
    config.setUseTopDownTraversal(top_down);
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }
  }
};
