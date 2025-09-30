#include "src/enzyme_ad/jax/Passes/AutoBatching.h"

#include "Enzyme/MLIR/Dialect/Dialect.h"
#include "Enzyme/MLIR/Passes/EnzymeBatchPass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/SetVector.h"
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

  SmallVector<Type> argTypes;
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
  std::map<enzyme::batchutils::BatchCacheKey, FunctionOpInterface>
      batchedFunctionCache;
  return enzyme::batchutils::batchOperation(
      rewriter, batchOp, cast<FunctionOpInterface>(func.getOperation()),
      batchedFunctionCache);
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
  for (auto bDim : broadcastInDimOp.getBroadcastDimensions()) {
    if (bDim == dim)
      return false;
  }

  // insert dim must be of size 1
  return outputType.getShape()[dim] == 1;
}

LogicalResult
SliceToBatchBase::matchAndRewriteImpl(stablehlo::SliceOp sliceOp,
                                      PatternRewriter &rewriter) const {
  Value sliceInput = sliceOp.getOperand();
  // Find all slices of the same input that feed into equivalent operations
  SmallVector<SliceInfo> relatedSlices;
  SmallVector<Operation *> relatedOps;
  SmallVector<bool> allHaveIntermediateReshapes;
  Operation *targetOp = nullptr;
  int64_t sliceOperandIndex = -1;

  // Build worklist of all slice operations on the same input
  for (auto [idx, user] : llvm::enumerate(sliceInput.getUsers())) {
    auto candidateSlice = dyn_cast<stablehlo::SliceOp>(user);
    if (!candidateSlice || !candidateSlice.getResult().hasOneUse())
      continue;

    auto sliceInfo = extractSliceInfo(candidateSlice);

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
  SmallVector<SliceInfo> sortedSlices;
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

  std::map<enzyme::batchutils::BatchCacheKey, FunctionOpInterface>
      batchedFunctionCache;
  return enzyme::batchutils::batchOperation(
      rewriter, batchOp, cast<FunctionOpInterface>(func.getOperation()),
      batchedFunctionCache);
}

SliceToBatchBase::SliceInfo
SliceToBatchBase::extractSliceInfo(stablehlo::SliceOp slice) const {
  auto startIndices = llvm::to_vector(slice.getStartIndices());
  auto limitIndices = llvm::to_vector(slice.getLimitIndices());
  auto strides = llvm::to_vector(slice.getStrides());

  SmallVector<int64_t> infoStartIndices;
  SmallVector<int64_t> infoEndIndices;
  SmallVector<int64_t> infoStrides;

  for (auto [start, end, stride] :
       llvm::zip(startIndices, limitIndices, strides)) {
    infoStartIndices.push_back(start);
    infoEndIndices.push_back(end);
    infoStrides.push_back(stride);
  }

  // Find the sliceDim
  auto inputType = cast<RankedTensorType>(slice.getOperand().getType());
  ArrayRef<int64_t> inputShape = inputType.getShape();

  bool found = false;
  bool supported;
  int64_t sliceDim, sliceStart;

  supported = true;
  for (size_t i = 0; i < infoStartIndices.size(); ++i) {
    if (infoStartIndices[i] == infoEndIndices[i] - 1 &&
        !(infoStartIndices[i] == 0 && infoEndIndices[i] == inputShape[i])) {
      if (found) {
        // multiple singleton slices not supported
        supported = false;
        break;
      }
      sliceDim = i;
      sliceStart = infoStartIndices[i];
      found = true;
    } else { // For all other dims, must be a full slice
      if (!(infoStartIndices[i] == 0 && infoEndIndices[i] == inputShape[i] &&
            infoStrides[i] == 1)) {
        supported = false;
        break;
      }
    }
  }
  if (!found) { // If no singleton found, not supported
    supported = false;
  }

  return SliceToBatchBase::SliceInfo{
      slice,    infoStartIndices, infoEndIndices, infoStrides,
      sliceDim, sliceStart,       supported};
}

bool SliceToBatchBase::areSlicesContiguous(
    SmallVector<SliceInfo> &slices) const {
  if (slices.empty())
    return false;

  int64_t sliceDim = slices[0].sliceDim;
  for (const auto &slice : slices) {
    if (slice.sliceDim != sliceDim) {
      return false;
    }
    if (!slice.supported) {
      return false;
    }
  }

  // Sort slices by start index
  llvm::sort(slices, [](const SliceInfo &a, const SliceInfo &b) {
    return a.sliceStart < b.sliceStart;
  });

  int64_t expectedStart = slices[0].sliceStart;
  // TODO: partial slice support
  if (expectedStart != 0)
    return false;

  for (const auto &slice : slices) {
    if (slice.sliceStart != expectedStart) {
      return false;
    }
    expectedStart++;
  }

  // TODO: we should support cases where the entire batch is not being sliced
  //       but introducing an intermediate slice.
  if (expectedStart !=
      slices[0].sliceOp.getOperand().getType().getShape()[sliceDim])
    return false;

  return true;
}

std::tuple<bool, bool>
SliceToBatchBase::allSameBool(const SmallVector<bool> &bools) const {
  bool first = bools[0];
  for (auto b : bools) {
    if (b != first)
      return {false, first};
  }
  return {true, first};
}

bool SliceToBatchBase::allOpsAreUnique(
    const SmallVector<Operation *> &ops) const {
  SmallPtrSet<Operation *, 8> seen;
  for (auto op : ops) {
    if (!seen.insert(op).second)
      return false;
  }
  return true;
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

    GreedyRewriteConfig config;
    config.setMaxIterations(max_iterations);
    config.setUseTopDownTraversal(top_down);
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }
  }
};
