#include "src/enzyme_ad/jax/Passes/AutoBatching.h"

#include "Enzyme/MLIR/Passes/EnzymeBatchPass.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Implementations/WhileLoopInfo.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Passes/StructuredTensors.h"
#include "src/enzyme_ad/jax/Utils.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include <algorithm>
#include <iterator>

#define DEBUG_TYPE "auto-batching"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_AUTOBATCHINGPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

static int64_t batchCounter = 0;

namespace utils {

// This function checks if any 2 ops in the list are data-dependent on each
// other. We exploit the fact that while traversing the dep graph if we are at a
// position before the other ops in the set, we know that the other ops are not
// data dependent.
bool anyOpsAreDataDependent(ArrayRef<Operation *> ops) {
  if (ops.size() <= 1) {
    return false;
  }

  Block *parentBlock = ops[0]->getBlock();
  // dependency analysis for ops in different blocks is hard. conservatively
  // assume that all ops are data dependent
  if (llvm::any_of(
          ops, [&](Operation *op) { return op->getBlock() != parentBlock; })) {
    return true;
  }

  llvm::SetVector<Operation *> sortedOpsTmp(ops.begin(), ops.end());
  auto sortedOps = mlir::topologicalSort(sortedOpsTmp);

  llvm::SmallSet<Value, 8> dependentValues;
  for (auto op : sortedOps) {
    for (auto result : op->getResults()) {
      dependentValues.insert(result);
    }
  }

  SmallVector<Operation *> worklist;

  // For each op, we only need to check if it depends on any earlier op in our
  // subset. We can use a worklist approach but only traverse backwards in
  // program order
  for (auto op : llvm::drop_begin(sortedOps)) {
    for (Value operand : op->getOperands()) {
      if (dependentValues.contains(operand)) {
        return true;
      }
      if (auto definingOp = operand.getDefiningOp()) {
        worklist.push_back(definingOp);
      }
    }

    while (!worklist.empty()) {
      Operation *curr = worklist.pop_back_val();

      // Only explore dependencies of operations that come after first op
      if (curr->isBeforeInBlock(sortedOps.front())) {
        continue;
      }

      for (Value operand : curr->getOperands()) {
        if (dependentValues.contains(operand)) {
          return true;
        }

        Operation *definingOp = operand.getDefiningOp();
        if (definingOp && definingOp->getBlock() == parentBlock &&
            definingOp->isBeforeInBlock(sortedOps.front())) {
          worklist.push_back(definingOp);
        }
      }
    }
  }

  return false;
}

func::FuncOp CreateWrapperUnbatchedFunction(
    mlir::ModuleOp modOp, PatternRewriter &rewriter, std::string funcName,
    SmallVectorImpl<BatchLiftingMode> &batchLiftingModes, Operation *op,
    std::optional<SmallVector<int64_t>> outShape) {
  assert(op->getNumResults() == 1 && "only support single result ops");

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(modOp.getBody());

  SmallVector<Type> argTypes;
  for (auto [i, v] : llvm::enumerate(op->getOperands())) {
    if (batchLiftingModes[i] == BatchLiftingMode::CONSTANT) {
      continue;
    }
    argTypes.push_back(v.getType());
  }

  FunctionType calleeType =
      rewriter.getFunctionType(argTypes, op->getResultTypes());
  func::FuncOp funcOp = func::FuncOp::create(
      rewriter, op->getLoc(), funcName + std::to_string(batchCounter++),
      calleeType);
  funcOp.setPrivate();

  auto &entryBlock = *funcOp.addEntryBlock();
  rewriter.setInsertionPointToStart(&entryBlock);

  IRMapping mapper;
  size_t argIdx = 0;
  for (auto [batchLiftMode, operand] :
       llvm::zip(batchLiftingModes, op->getOperands())) {
    if (batchLiftMode == BatchLiftingMode::CONSTANT) { // clone into fn body
      auto clonedConst = rewriter.clone(*operand.getDefiningOp());
      mapper.map(operand, clonedConst->getResult(0));
      continue;
    }
    mapper.map(operand, entryBlock.getArguments()[argIdx++]);
  }

  Value result = rewriter.clone(*op, mapper)->getResult(0);
  if (outShape.has_value()) {
    result = stablehlo::ReshapeOpCreate(rewriter, op->getLoc(), result,
                                        outShape.value());
  }
  func::ReturnOp::create(rewriter, op->getLoc(), ValueRange(result));

  return funcOp;
}

void ConstructAndExtractBatchOperands(
    PatternRewriter &rewriter, ArrayRef<Operation *> batchOps, Location loc,
    std::optional<BatchOperandConstructionInfo<stablehlo::SliceOp>> batchInfo,
    SmallVectorImpl<Value> &operands,
    SmallVectorImpl<BatchLiftingMode> &liftingModes) {
  for (int i = 0; i < batchOps[0]->getNumOperands(); i++) {
    SmallVector<Value> currentOperands(batchOps.size());
    for (auto [idx, op] : llvm::enumerate(batchOps)) {
      currentOperands[idx] = op->getOperand(i);
    }

    // all the values are same
    SplatElementsAttr firstSplat;
    if (matchPattern(currentOperands[0], m_Constant(&firstSplat))) {
      if (llvm::all_of(currentOperands, [&](Value v) {
            SplatElementsAttr splatAttr;
            if (matchPattern(v, m_Constant(&splatAttr))) {
              return splatAttr == firstSplat;
            }
            return false;
          })) {
        liftingModes.push_back(BatchLiftingMode::CONSTANT);
        continue;
      }
    }

    if (batchInfo.has_value() && i == batchInfo->sliceOperandIndex) {
      SmallVector<Value> concatOperands(batchInfo->slices.size());
      for (auto [idx, slice] : llvm::enumerate(batchInfo->slices)) {
        assert(slice.dimensions.size() == 1);
        concatOperands[idx] = slice.sliceOp.getResult();
      }
      auto sliceDim = batchInfo->slices[0].dimensions[0];

      SmallVector<Value> newConcatOperands;
      auto canSimplify = stablehlo::concatSliceSimplify(
          rewriter, concatOperands, sliceDim, newConcatOperands);
      Value concatResult = stablehlo::ConcatenateOpCreate(
          rewriter, loc,
          canSimplify.succeeded() ? newConcatOperands : concatOperands,
          sliceDim);

      auto nbatchSize = static_cast<int64_t>(batchInfo->slices.size());
      auto resTy = cast<RankedTensorType>(concatResult.getType());
      assert(resTy.getDimSize(sliceDim) == nbatchSize);

      SmallVector<int64_t> permutation(resTy.getRank());
      permutation[0] = sliceDim;
      for (int64_t i = 0; i < sliceDim; i++) {
        permutation[i + 1] = i;
      }
      for (int64_t i = sliceDim + 1; i < resTy.getRank(); i++) {
        permutation[i] = i;
      }

      Value newOperand = stablehlo::TransposeOpCreate(
          rewriter, loc, concatResult, permutation);

      bool needsReshape = false;
      SmallVector<int64_t> outShape;
      if (batchInfo->intermediateReshape) {
        if (batchInfo->slices[0].explicitReshapeShape.has_value()) {
          needsReshape = true;
          outShape = batchInfo->slices[0].explicitReshapeShape.value();
          outShape.insert(outShape.begin(), nbatchSize);
        }
      } else {
        needsReshape = true;
        outShape =
            llvm::to_vector(cast<ShapedType>(newOperand.getType()).getShape());
        outShape.insert(outShape.begin() + sliceDim + 1, 1);
      }

      if (needsReshape) {
        newOperand =
            stablehlo::ReshapeOpCreate(rewriter, loc, newOperand, outShape);
      }

      liftingModes.push_back(BatchLiftingMode::DEFINED_OUTSIDE_WHILE);
      operands.push_back(newOperand);
    } else { // Non-equivalent operands - need to concatenate them
      SmallVector<Value> newConcatOperands;
      for (Value operand : currentOperands) {
        auto inputType = cast<RankedTensorType>(operand.getType());
        auto inputShape = inputType.getShape();

        SmallVector<int64_t> outputShape(inputShape.begin(), inputShape.end());
        outputShape.insert(outputShape.begin(), 1);

        // expand the batch dim (== 0) to `1`
        newConcatOperands.push_back(
            stablehlo::ReshapeOpCreate(rewriter, loc, operand, outputShape));
      }

      liftingModes.push_back(BatchLiftingMode::DEFINED_OUTSIDE_WHILE);
      operands.push_back(
          stablehlo::ConcatenateOpCreate(rewriter, loc, newConcatOperands, 0));
    }
  }
}

bool IsEquivalentToIgnoringValueEquivalence(Operation *op1, Operation *op2) {
  return OperationEquivalence::isEquivalentTo(
      op1, op2, OperationEquivalence::ignoreValueEquivalence, nullptr,
      OperationEquivalence::IgnoreLocations, nullptr);
}

// Implicitly assumes before in block if ops are in different blocks
bool isBeforeInBlock(Operation *op, Operation *otherOp) {
  if (op->getBlock() != otherOp->getBlock())
    return true;
  return op->isBeforeInBlock(otherOp);
}

bool allOpsAreUnique(const SmallVector<Operation *> &ops) {
  SmallPtrSet<Operation *, 8> seen;
  return llvm::all_of(ops,
                      [&](Operation *op) { return seen.insert(op).second; });
}

// dim == -1 => ignore the dimension check
bool CheckIsValidForBatching(
    stablehlo::ReshapeOp op, int64_t dim,
    llvm::SmallVectorImpl<int64_t> &intermediateInsertions,
    bool checkInsertion) {
  auto inTy = cast<RankedTensorType>(op.getOperand().getType());
  auto outTy = cast<RankedTensorType>(op.getType());

  if (!checkInsertion) {
    std::swap(inTy, outTy);
  }

  auto insertionDims = findReshapeInsertionDims(inTy, outTy);
  if (insertionDims.empty()) {
    return false;
  }

  if (dim == -1 || llvm::is_contained(insertionDims, dim)) {
    for (auto i : insertionDims) {
      if (i != dim) {
        intermediateInsertions.push_back(i);
      }
    }
    return true;
  }
  return false;
}

bool CheckIsValidForBatching(
    stablehlo::BroadcastInDimOp op, int64_t dim,
    llvm::SmallVectorImpl<int64_t> &intermediateInsertions,
    bool checkInsertion) {
  if (!checkInsertion) {
    return false; // bcast in dim cannot perform deletion
  }

  auto inputType = cast<RankedTensorType>(op.getOperand().getType());
  auto outputType = cast<RankedTensorType>(op.getType());

  // If concat dim is present in broadcast dims, then it is not a valid insert
  if ((dim != -1 && llvm::is_contained(op.getBroadcastDimensions(), dim)) ||
      !llvm::is_sorted(op.getBroadcastDimensions())) {
    return false;
  }

  // all bcasted dim but preserve size
  for (auto [i, bDim] : llvm::enumerate(op.getBroadcastDimensions())) {
    if (outputType.getDimSize(bDim) != inputType.getDimSize(i)) {
      return false;
    }
  }

  bool found = false;
  for (size_t i = 0; i < outputType.getRank(); i++) {
    if (!llvm::is_contained(op.getBroadcastDimensions(), i) &&
        outputType.getDimSize(i) == 1) {
      if (i == dim) {
        found = true;
        continue;
      }
      intermediateInsertions.push_back(i);
    }
  }
  return dim == -1 || found;
}

} // namespace utils

SliceInfo<stablehlo::SliceOp> constructSliceInfo(stablehlo::SliceOp sliceOp) {
  auto startIndices = llvm::to_vector(sliceOp.getStartIndices());
  auto limitIndices = llvm::to_vector(sliceOp.getLimitIndices());

  SmallVector<int64_t> dimensions;
  for (size_t i = 0; i < startIndices.size(); ++i) {
    if (startIndices[i] == limitIndices[i] - 1) {
      dimensions.push_back(i);
    }
  }

  if (dimensions.empty()) {
    return SliceInfo<stablehlo::SliceOp>();
  }

  return SliceInfo<stablehlo::SliceOp>{sliceOp, dimensions, false,
                                       std::nullopt};
}

void ComputeSliceDimension(ArrayRef<SliceInfo<mlir::stablehlo::SliceOp>> slices,
                           int64_t *sliceDim) {
  // find potential slice dimensions by constructing the intersection of all
  llvm::SmallSetVector<int64_t, 4> dimensions(slices[0].dimensions.begin(),
                                              slices[0].dimensions.end());
  for (auto &slice : llvm::drop_begin(slices)) {
    llvm::SmallSetVector<int64_t, 4> curDims(slice.dimensions.begin(),
                                             slice.dimensions.end());
    llvm::set_intersect(dimensions, curDims);
  }
  if (dimensions.empty()) {
    *sliceDim = -1;
    return;
  }

  // find the first dimension for which all conditions on the slice ops are
  // satisfied
  for (auto dim : dimensions) {
    auto baseSliceOp = slices[0].sliceOp;
    auto explicitReshapeShape = slices[0].explicitReshapeShape;

    for (auto slice : llvm::drop_begin(slices)) {
      auto curSliceOp = slice.sliceOp;

      if (explicitReshapeShape.has_value()) {
        if (!slice.explicitReshapeShape.has_value() ||
            explicitReshapeShape.value() !=
                slice.explicitReshapeShape.value()) {
          *sliceDim = -1;
          return;
        }
      }

      for (int64_t i = 0; i < baseSliceOp.getStartIndices().size(); ++i) {
        if (i == dim) {
          continue;
        }

        if (baseSliceOp.getStartIndices()[i] !=
                curSliceOp.getStartIndices()[i] ||
            baseSliceOp.getLimitIndices()[i] !=
                curSliceOp.getLimitIndices()[i] ||
            baseSliceOp.getStrides()[i] != curSliceOp.getStrides()[i]) {
          *sliceDim = -1;
          return;
        }
      }
    }

    *sliceDim = dim;
    return;
  }

  *sliceDim = -1;
  return;
}

LogicalResult ConcatInsertDimToBatchBase::matchAndRewriteImpl(
    stablehlo::ConcatenateOp concatOp, PatternRewriter &rewriter) const {
  if (concatOp.getNumOperands() <= 1) {
    return failure();
  }

  auto concatDim = concatOp.getDimension();
  auto concatType = cast<RankedTensorType>(concatOp.getResult().getType());
  auto concatShape = concatType.getShape();

  SmallVector<Operation *> concatOpOperands;
  SmallVector<int64_t> extraIntermediateInsertions;

  for (auto [i, v] : llvm::enumerate(concatOp.getOperands())) {
    auto definingOp = v.getDefiningOp();
    if (!definingOp) {
      return rewriter.notifyMatchFailure(concatOp, "operand is not a valid op");
    }

    SmallVector<int64_t> intermediateInsertions;
    bool validIntermediate =
        TypeSwitch<Operation *, bool>(definingOp)
            .Case<stablehlo::ReshapeOp, stablehlo::BroadcastInDimOp>(
                [&](auto op) {
                  return ::utils::CheckIsValidForBatching(
                      op, concatDim, intermediateInsertions, true);
                })
            .Default([](auto op) { return false; });

    if (i == 0) {
      extraIntermediateInsertions = std::move(intermediateInsertions);
    } else if (extraIntermediateInsertions != intermediateInsertions) {
      return rewriter.notifyMatchFailure(concatOp,
                                         "not all ops have same intermediate "
                                         "reshape");
    }

    if (!validIntermediate) {
      return rewriter.notifyMatchFailure(concatOp, "operand is not a valid op");
    }

    auto vdefOp = isValidTargetOp(definingOp->getOperand(0).getDefiningOp());
    if (!vdefOp) {
      return rewriter.notifyMatchFailure(concatOp, "not a valid target op");
    }

    if (concatOpOperands.size() != 0) {
      if (!::utils::IsEquivalentToIgnoringValueEquivalence(concatOpOperands[0],
                                                           vdefOp)) {
        return rewriter.notifyMatchFailure(concatOp,
                                           "op is not equivalent to first");
      }
    }

    if (!isOnlyUsedInOperation(vdefOp, definingOp)) {
      return rewriter.notifyMatchFailure(concatOp,
                                         "op is not only used in reshape op");
    }

    concatOpOperands.push_back(vdefOp);
  }

  SmallVector<Value> batchOpOperands;
  SmallVector<BatchLiftingMode> liftingModes;
  ::utils::ConstructAndExtractBatchOperands(rewriter, concatOpOperands,
                                            concatOp.getLoc(), std::nullopt,
                                            batchOpOperands, liftingModes);

  SmallVector<int64_t> outputShape;
  for (int i = 0; i < concatShape.size(); i++) {
    if (i == concatDim) {
      continue;
    }
    outputShape.push_back(concatShape[i]);
  }

  std::optional<SmallVector<int64_t>> explicitReshapeShape;
  if (!extraIntermediateInsertions.empty()) {
    explicitReshapeShape = outputShape;
  }
  auto moduleOp = concatOp->getParentOfType<ModuleOp>();
  func::FuncOp func = ::utils::CreateWrapperUnbatchedFunction(
      moduleOp, rewriter, "enzymexla_unbatched_ConcatInsertDimToBatch_",
      liftingModes, concatOpOperands[0], explicitReshapeShape);

  outputShape.insert(outputShape.begin(), concatShape[concatDim]);
  auto batchOp = enzyme::BatchOp::create(
      rewriter, concatOp.getLoc(),
      RankedTensorType::get(outputShape, concatType.getElementType()),
      mlir::FlatSymbolRefAttr::get(concatOp.getContext(), func.getName()),
      ValueRange(batchOpOperands),
      rewriter.getDenseI64ArrayAttr({concatShape[concatDim]}));

  SmallVector<int64_t> permutation;
  for (int i = 1; i <= concatDim; i++) {
    permutation.push_back(i);
  }
  permutation.push_back(0);
  for (int i = concatDim + 1; i < concatShape.size(); i++) {
    permutation.push_back(i);
  }

  rewriter.replaceOpWithNewOp<stablehlo::TransposeOp>(
      concatOp, batchOp->getResult(0), permutation);

  enzyme::batchutils::batchOperationInline(
      rewriter, batchOp, cast<FunctionOpInterface>(func.getOperation()));
  return success();
}

LogicalResult
SliceToBatchBase::matchAndRewriteImpl(stablehlo::SliceOp sliceOp,
                                      PatternRewriter &rewriter) const {
  Value sliceInput = sliceOp.getOperand();
  auto block = sliceOp->getBlock();

  // Find all slices of the same input that feed into equivalent operations
  SmallVector<SliceInfo<stablehlo::SliceOp>> relatedSlices;
  SmallVector<Operation *> relatedOps;
  SmallVector<bool> allHaveIntermediateReshapes;
  Operation *targetOp = nullptr;
  int64_t sliceOperandIndex = -1;

  // Build worklist of all slice operations on the same input
  for (auto [idx, user] : llvm::enumerate(sliceInput.getUsers())) {
    auto candidateSlice = dyn_cast<stablehlo::SliceOp>(user);
    if (!candidateSlice || !candidateSlice.getResult().hasOneUse()) {
      continue;
    }

    auto sliceInfo = constructSliceInfo(candidateSlice);

    Operation *onlyUser = *candidateSlice.getResult().getUsers().begin();
    Operation *candidateTargetOp = isValidTargetOp(onlyUser);

    bool isIntermediateReshape = false;
    Operation *preceedingOp = candidateSlice;
    if (!candidateTargetOp) {
      SmallVector<int64_t> intermediateInsertions;
      isIntermediateReshape =
          TypeSwitch<Operation *, bool>(onlyUser)
              .Case<stablehlo::ReshapeOp, stablehlo::BroadcastInDimOp>(
                  [&](auto op) {
                    auto res = op.getResult();
                    preceedingOp = op;
                    if (res.hasOneUse() &&
                        ::utils::CheckIsValidForBatching(
                            op, -1, intermediateInsertions, false)) {
                      candidateTargetOp =
                          isValidTargetOp(*res.getUsers().begin());
                      if (candidateTargetOp) {
                        return true;
                      }
                      return true;
                    }
                    return false;
                  })
              .Default([](auto op) { return false; });

      if (isIntermediateReshape) {
        // resolve the intermediate reshape
        sliceInfo.intermediateReshape = true;
        sliceInfo.explicitReshapeShape = llvm::to_vector(
            cast<ShapedType>(preceedingOp->getResult(0).getType()).getShape());
      }
    }

    if (!candidateTargetOp || candidateTargetOp->getBlock() != block) {
      continue; // only consider ops in the same block
    }

    // check that all of the ops are equivalent and that the slice operand is
    // at the same location
    if (targetOp) {
      if (!::utils::IsEquivalentToIgnoringValueEquivalence(targetOp,
                                                           candidateTargetOp)) {
        continue;
      }
      if (candidateTargetOp->getOperand(sliceOperandIndex) !=
          preceedingOp->getResult(0)) {
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

  if (relatedSlices.size() <= 1) {
    return rewriter.notifyMatchFailure(sliceOp, "no related slices found");
  }

  if (!::utils::allOpsAreUnique(relatedOps)) {
    return rewriter.notifyMatchFailure(sliceOp, "ops are not unique");
  }

  if (sliceOperandIndex < 0) {
    return rewriter.notifyMatchFailure(sliceOp, "slice operand not found");
  }

  if (llvm::any_of(allHaveIntermediateReshapes, [=](bool b) {
        return b != allHaveIntermediateReshapes[0];
      })) {
    return rewriter.notifyMatchFailure(
        sliceOp, "slices have different intermediate reshape");
  }

  int64_t sliceDim = -1;
  ComputeSliceDimension(relatedSlices, &sliceDim);

  if (sliceDim < 0) {
    return rewriter.notifyMatchFailure(sliceOp, "slice dimension not found");
  }

  // Sort all vectors together based on sliceStart to ensure better locality
  SmallVector<size_t> indices(relatedSlices.size());
  std::iota(indices.begin(), indices.end(), 0);

  std::sort(indices.begin(), indices.end(), [&](size_t i, size_t j) {
    return relatedSlices[i].sliceOp.getStartIndices()[sliceDim] <
           relatedSlices[j].sliceOp.getStartIndices()[sliceDim];
  });

  // Reorder all three vectors according to the sorted indices
  SmallVector<SliceInfo<stablehlo::SliceOp>> sortedSlices;
  SmallVector<Operation *> sortedOps;

  for (size_t idx : indices) {
    sortedSlices.push_back(relatedSlices[idx]);
    sortedOps.push_back(relatedOps[idx]);
  }

  relatedSlices = std::move(sortedSlices);
  relatedOps = std::move(sortedOps);

  // quite an expensive check, so run at the very end
  if (::utils::anyOpsAreDataDependent(relatedOps)) {
    return rewriter.notifyMatchFailure(sliceOp, "ops are data dependent");
  }

  // move all the related ops and their operands (and theirs) s.t they are
  // together
  Operation *firstRelatedOp = relatedOps[0];
  for (auto &op : relatedOps) {
    if (::utils::isBeforeInBlock(op, firstRelatedOp)) {
      firstRelatedOp = op;
    }
  }
  // if we have to move around too many ops we avoid applying this pattern
  llvm::SetVector<Operation *> opsToMoveWorklist;
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
            if (!::utils::isBeforeInBlock(operandOp, firstRelatedOp)) {
              opsToMove.push_back(operandOp);
            }
          }
        }
      }
    }
  }

  auto opsToMoveSorted = mlir::topologicalSort(opsToMoveWorklist);
  for (auto op : opsToMoveSorted) {
    rewriter.moveOpAfter(op, firstRelatedOp);
    firstRelatedOp = op;
  }

  Operation *insertionPoint = relatedOps[0];
  for (auto &op : relatedOps) {
    if (::utils::isBeforeInBlock(op, insertionPoint)) {
      continue;
    }
    insertionPoint = op;
  }
  rewriter.setInsertionPoint(insertionPoint);

  for (auto &slice : relatedSlices) {
    slice.dimensions = {sliceDim};
  }

  SmallVector<Value> batchOpOperands;
  SmallVector<BatchLiftingMode> liftingModes;
  ::utils::ConstructAndExtractBatchOperands(
      rewriter, relatedOps, sliceOp.getLoc(),
      BatchOperandConstructionInfo<stablehlo::SliceOp>{
          relatedSlices, static_cast<int32_t>(sliceOperandIndex),
          allHaveIntermediateReshapes[0]},
      batchOpOperands, liftingModes);

  auto moduleOp = sliceOp->getParentOfType<ModuleOp>();
  func::FuncOp func = ::utils::CreateWrapperUnbatchedFunction(
      moduleOp, rewriter, "enzymexla_unbatched_SliceToBatch_", liftingModes,
      relatedOps[0], std::nullopt);

  SmallVector<int64_t> outputShape;
  outputShape.push_back(relatedSlices.size());
  auto relatedOpsType =
      cast<RankedTensorType>(relatedOps[0]->getResult(0).getType());
  auto funcRetShape = relatedOpsType.getShape();
  outputShape.append(funcRetShape.begin(), funcRetShape.end());

  auto batchOp = enzyme::BatchOp::create(
      rewriter, sliceOp.getLoc(),
      RankedTensorType::get(outputShape, relatedOpsType.getElementType()),
      mlir::FlatSymbolRefAttr::get(sliceOp.getContext(), func.getName()),
      ValueRange(batchOpOperands),
      rewriter.getDenseI64ArrayAttr(
          {static_cast<int64_t>(relatedSlices.size())}));

  SmallVector<int64_t> startIndices(outputShape.size(), 0);
  SmallVector<int64_t> endIndices;
  endIndices.append(outputShape.begin(), outputShape.end());
  SmallVector<int64_t> strides(outputShape.size(), 1);
  for (auto [idx, sliceInfoAndOp] :
       llvm::enumerate(llvm::zip(relatedSlices, relatedOps))) {
    auto &[sliceInfo, otherOp] = sliceInfoAndOp;
    startIndices[0] = idx;
    endIndices[0] = idx + 1;

    auto slicedOp = stablehlo::SliceOp::create(
        rewriter, sliceOp.getLoc(), batchOp->getResult(0),
        rewriter.getDenseI64ArrayAttr(startIndices),
        rewriter.getDenseI64ArrayAttr(endIndices),
        rewriter.getDenseI64ArrayAttr(strides));
    rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(
        otherOp, otherOp->getResult(0).getType(), slicedOp);
  }

  enzyme::batchutils::batchOperationInline(
      rewriter, batchOp, cast<FunctionOpInterface>(func.getOperation()));
  return success();
}

static bool definedOutside(Value v, Operation *op) {
  return !op->isAncestor(v.getParentBlock()->getParentOp());
}

LogicalResult GreedyWhileLoopBatchFission::matchAndRewriteImpl(
    stablehlo::WhileOp whileOp, PatternRewriter &rewriter) const {
  auto info = WhileLoopInfo(whileOp);
  auto computeInfoSuccess = info.computeInfo();
  if (computeInfoSuccess.failed())
    return computeInfoSuccess;

  // TODO: the only actual restriction is that the total loop iterations be
  // constant and the indexing must be affine
  if (!info.isValid() || !info.isConstant())
    return failure();

  auto &whileBody = whileOp.getBody().front();

  auto affineIndexInfoMap = info.getAffineIndexInfo();

  auto parentFunc = whileOp->getParentOp();
  if (!parentFunc)
    return rewriter.notifyMatchFailure(whileOp, "parent function not found");

  // Find all dynamic slices in the loop body that meet the criteria:
  // 1. All slice variables are constant across iterations
  // 2. Only one variable in the body is a direct descendant of the induction
  // variable
  SmallPtrSet<Operation *, 8> seenOps;
  SmallVector<SliceInfo<stablehlo::DynamicSliceOp>> candidateSlices;

  for (auto [value, affineIndexInfo] : affineIndexInfoMap) {
    for (auto user : value.getUsers()) {
      if (user->getBlock() != &whileBody || seenOps.contains(user)) {
        continue;
      }

      seenOps.insert(user);

      if (auto sliceOp = dyn_cast<stablehlo::DynamicSliceOp>(user)) {
        auto result = isDynamicSliceValidForBatching(
            sliceOp, affineIndexInfoMap, whileBody, whileOp);

        if (isValidForBatchingResult(result.result)) {
          candidateSlices.push_back(SliceInfo<stablehlo::DynamicSliceOp>{
              sliceOp, result.dimensions, false, std::nullopt});
        }
      }
    }
  }

  // Create a map of user operations to their corresponding dynamic slices
  llvm::MapVector<Operation *,
                  SmallVector<SliceInfo<stablehlo::DynamicSliceOp>>>
      userOpToSlicesMap;
  for (auto ds : candidateSlices) {
    for (auto op : ds.sliceOp->getUsers()) {
      userOpToSlicesMap[op].push_back(ds);

      if (isa<stablehlo::ReshapeOp>(op)) {
        auto operandTy = cast<RankedTensorType>(op->getOperand(0).getType());
        auto resultTy = cast<RankedTensorType>(op->getResult(0).getType());

        std::optional<SmallVector<int64_t>> reshapeShape;
        if (!areValidInsertionDims(resultTy, operandTy, ds.dimensions)) {
          reshapeShape = llvm::to_vector(resultTy.getShape());
        }

        for (auto user : op->getUsers()) {
          userOpToSlicesMap[user].push_back(
              SliceInfo<stablehlo::DynamicSliceOp>{ds.sliceOp, ds.dimensions,
                                                   true, reshapeShape});
        }
      }
    }
  }

  // for certain operations on index variables it is more efficient to hoist
  // those out of the loop and then perform indirect indexing
  for (auto &[val, slices] : affineIndexInfoMap) {
    for (auto user : val.getUsers()) {
      if (isa<stablehlo::CompareOp, stablehlo::BroadcastInDimOp>(user)) {
        userOpToSlicesMap[user].push_back(
            SliceInfo<stablehlo::DynamicSliceOp>{});
      }
    }
  }

  if (userOpToSlicesMap.empty())
    return failure();

  bool anyOpRewritten = false;

  for (auto &[op, slices] : userOpToSlicesMap) {
    bool avoidBatching =
        llvm::TypeSwitch<Operation *, bool>(op)
            .Case<stablehlo::DynamicSliceOp, stablehlo::DynamicUpdateSliceOp,
                  stablehlo::ReshapeOp>([=](auto op) { return true; })
            .Case<stablehlo::BroadcastInDimOp>(
                [=](auto op) { return stablehlo::broadcastInDimIsReshape(op); })
            .Default([](auto op) { return false; });
    if (avoidBatching) {
      continue;
    }

    if ((dyn_cast<BatchOpInterface>(op) ||
         stablehlo::hasTraitElementwise(op)) &&
        op->getNumResults() == 1) {
      if (liftOperationByBatching(rewriter, whileOp, slices, op, info)) {
        anyOpRewritten = true;
      }
    }
  }

  return anyOpRewritten ? success() : failure();
};

bool GreedyWhileLoopBatchFission::liftOperationByBatching(
    PatternRewriter &rewriter, stablehlo::WhileOp whileOp,
    ArrayRef<SliceInfo<stablehlo::DynamicSliceOp>> slices, Operation *op,
    WhileLoopInfo info) const {
  auto moduleOp = op->getParentOfType<ModuleOp>();
  auto affineIndexInfoMap = info.getAffineIndexInfo();

  SmallVector<BatchLiftingMode> batchLiftingModes(op->getNumOperands());
  SmallVector<Value> batchOperands(op->getNumOperands());
  SmallVector<SmallVector<int64_t>> sliceDims(op->getNumOperands());
  SmallVector<int64_t> hoistedDims(op->getNumOperands());
  SmallVector<SliceInfo<stablehlo::DynamicSliceOp>> mappedSliceInfos(
      op->getNumOperands());
  DenseMap<Value, SmallVector<Operation *>> hoistMap;

  for (int i = 0; i < op->getNumOperands(); i++) {
    auto operand = op->getOperand(i);

    Value outerValue;
    SmallVector<Operation *> canBeHoisted;
    if (info.isConstantAcrossIterations(operand, outerValue, canBeHoisted)) {
      if (outerValue) {
        SplatElementsAttr splat;
        if (matchPattern(operand, m_Constant(&splat))) {
          batchLiftingModes[i] = BatchLiftingMode::CONSTANT;
        } else {
          batchLiftingModes[i] = BatchLiftingMode::DEFINED_OUTSIDE_WHILE;
        }
        batchOperands[i] = outerValue;
      } else {
        hoistMap[operand] = canBeHoisted;
        hoistedDims[i] = cast<mlir::OpResult>(operand).getResultNumber();
        batchLiftingModes[i] = BatchLiftingMode::NEEDS_HOISTING_OUTSIDE_WHILE;
        batchOperands[i] = operand;
      }
      continue;
    }

    if (affineIndexInfoMap.contains(operand) &&
        !cast<RankedTensorType>(operand.getType())
             .getElementType()
             .isInteger(1)) {
      batchLiftingModes[i] = BatchLiftingMode::AFFINE_INDEX;
      batchOperands[i] = operand;
      continue;
    }

    auto defOp = operand.getDefiningOp();
    if (!defOp) {
      return false;
    }

    Operation *dsOp;
    bool mustBeIntermediateReshape = false;
    if (auto reshapeOp = dyn_cast<stablehlo::ReshapeOp>(defOp)) {
      mustBeIntermediateReshape = true;
      dsOp = reshapeOp.getOperand().getDefiningOp();
    } else {
      dsOp = defOp;
    }

    if (auto ds = dyn_cast<stablehlo::DynamicSliceOp>(dsOp)) {
      auto itr = llvm::find_if(
          slices, [&](const SliceInfo<stablehlo::DynamicSliceOp> &info) {
            return info.sliceOp == ds;
          });
      if (itr != slices.end()) {
        batchLiftingModes[i] = BatchLiftingMode::DYNAMIC_SLICE;
        sliceDims[i] = itr->dimensions;

        auto dsOperand = ds->getOperand(0);
        if (definedOutside(dsOperand, whileOp)) {
          batchOperands[i] = dsOperand;
        } else {
          auto blockArg = dyn_cast<BlockArgument>(dsOperand);
          assert(blockArg && "expected block arg");
          batchOperands[i] = whileOp->getOperand(blockArg.getArgNumber());
        }

        mappedSliceInfos[i] = *itr;
        if (mustBeIntermediateReshape && !itr->intermediateReshape) {
          return false;
        }
        continue;
      } else {
        return false;
      }
    }

    return false;
  }

  func::FuncOp func = ::utils::CreateWrapperUnbatchedFunction(
      moduleOp, rewriter, "enzymexla_unbatched_WhileLoopBatchFission_",
      batchLiftingModes, op, std::nullopt);

  rewriter.setInsertionPoint(whileOp);

  // hoist any operations that can be hoisted
  DenseMap<Value, Value> hoistedValues;
  IRMapping mapper;
  for (auto &[val, ops] : hoistMap) {
    llvm::SetVector<Operation *> toHoist(ops.begin(), ops.end());
    auto sorted = mlir::topologicalSort(toHoist);

    for (auto &op : sorted) {
      if (llvm::all_of(op->getResults(),
                       [&](Value v) { return mapper.contains(v); }))
        continue;

      for (auto operand : op->getOperands()) {
        if (mapper.contains(operand))
          continue;

        if (!definedOutside(operand, whileOp)) {
          Value outerValue;
          SmallVector<Operation *> canBeHoisted;
          if (info.isConstantAcrossIterations(operand, outerValue, canBeHoisted,
                                              false)) {
            mapper.map(operand, outerValue);
          }
        }
      }
      auto hoisted = rewriter.clone(*op, mapper);
      for (auto [origRes, newRes] :
           llvm::zip(op->getResults(), hoisted->getResults())) {
        mapper.map(origRes, newRes);
      }
    }

    hoistedValues[val] = mapper.lookup(val);
  }

  SmallVector<Value> newOperands;
  for (auto [consType, baseOp, sliceDim, sliceInfo, hoistDim] :
       llvm::zip(batchLiftingModes, batchOperands, sliceDims, mappedSliceInfos,
                 hoistedDims)) {
    auto operandType = cast<RankedTensorType>(baseOp.getType());
    int operandRank = cast<RankedTensorType>(baseOp.getType()).getRank();

    switch (consType) {
    case BatchLiftingMode::DYNAMIC_SLICE: {
      // hoist the dynamic slice out of the loop and replace the sliceDim
      // with full slice.
      Value newSlice;
      bool successfulHoist = info.hoistOperationFromLoop(
          rewriter, baseOp, sliceInfo.sliceOp, sliceDim, newSlice);
      assert(successfulHoist && "Expected DS hoist to succeed");
      (void)successfulHoist;
      auto originalShape =
          cast<RankedTensorType>(sliceInfo.sliceOp.getType()).getShape();

      auto DSType = cast<RankedTensorType>(newSlice.getType());
      SmallVector<int64_t> permutation(DSType.getRank());
      permutation[0] = sliceDim[0];
      for (size_t i = 0; i < sliceDim[0]; i++)
        permutation[i + 1] = i;
      for (size_t i = sliceDim[0] + 1; i < DSType.getRank(); i++)
        permutation[i] = i;

      Value newOperand = stablehlo::TransposeOpCreate(
          rewriter, whileOp->getLoc(), newSlice, permutation);

      bool applyReshape = true;
      SmallVector<int64_t> reshapeShape;
      if (sliceInfo.intermediateReshape) {
        if (sliceInfo.explicitReshapeShape.has_value()) {
          reshapeShape = sliceInfo.explicitReshapeShape.value();
          reshapeShape.insert(reshapeShape.begin(), info.getConstantNumIters());
        } else {
          applyReshape = false;
        }
      } else {
        reshapeShape = llvm::to_vector(originalShape);
        reshapeShape.insert(reshapeShape.begin(), info.getConstantNumIters());
        for (auto dim : sliceDim)
          reshapeShape[dim + 1] = 1;
      }

      if (applyReshape) {
        newOperand = stablehlo::ReshapeOpCreate(rewriter, whileOp->getLoc(),
                                                newOperand, reshapeShape);
      }

      newOperands.push_back(newOperand);
      break;
    }
    case BatchLiftingMode::NEEDS_HOISTING_OUTSIDE_WHILE: {
      baseOp = hoistedValues[baseOp];
      LLVM_FALLTHROUGH;
    }
    case BatchLiftingMode::DEFINED_OUTSIDE_WHILE: {
      auto operandShape = operandType.getShape();
      SmallVector<int64_t> newOperandShape(operandRank + 1);
      newOperandShape[0] = info.getConstantNumIters();
      for (int i = 0; i < operandRank; i++)
        newOperandShape[i + 1] = operandShape[i];

      SmallVector<int64_t> mapping(operandRank);
      std::iota(mapping.begin(), mapping.end(), 1);

      auto broadcastedOperand = stablehlo::BroadcastInDimOp::create(
          rewriter, whileOp->getLoc(),
          RankedTensorType::get(newOperandShape, operandType.getElementType()),
          baseOp, rewriter.getDenseI64ArrayAttr(mapping));
      newOperands.push_back(broadcastedOperand->getResult(0));
      break;
    }
    case BatchLiftingMode::CONSTANT: {
      break; // copied into the function body no need to include in operands
    }
    case BatchLiftingMode::AFFINE_INDEX: {
      auto hoistedTy = RankedTensorType::get({info.getConstantNumIters()},
                                             operandType.getElementType());
      Value loopIndices = stablehlo::IotaOp::create(
          rewriter, whileOp->getLoc(),
          RankedTensorType::get({info.getConstantNumIters()},
                                operandType.getElementType()),
          0);

      auto createConst = [&](int64_t val) {
        return stablehlo::ConstantOp::create(
            rewriter, whileOp->getLoc(), hoistedTy,
            cast<ElementsAttr>(makeAttr(hoistedTy, val)));
      };

      auto startVal = createConst(info.getConstantStart().value());
      auto stepVal = createConst(info.getConstantStep().value());
      loopIndices = stablehlo::AddOp::create(
          rewriter, whileOp->getLoc(), loopIndices,
          stablehlo::MulOp::create(rewriter, whileOp->getLoc(), stepVal,
                                   startVal));

      auto affineIndexInfo = info.getAffineIndexInfo()[baseOp];
      auto scale = createConst(affineIndexInfo.scale.getSExtValue());
      auto offset = createConst(affineIndexInfo.offset.getSExtValue());
      auto res = stablehlo::AddOp::create(
          rewriter, whileOp->getLoc(),
          stablehlo::MulOp::create(rewriter, whileOp->getLoc(), scale,
                                   loopIndices),
          offset);
      newOperands.push_back(res);
      break;
    }
    }
  }

  auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
  auto resultShape = resultType.getShape();
  SmallVector<int64_t> outputShape(resultShape.size() + 1);
  outputShape[0] = info.getConstantNumIters();
  for (int i = 0; i < resultShape.size(); i++)
    outputShape[i + 1] = resultShape[i];

  auto inductionVar = info.getInductionVariable();
  auto inductionVarType = cast<RankedTensorType>(inductionVar.getType());

  auto constZero = stablehlo::ConstantOp::create(
      rewriter, whileOp->getLoc(), inductionVarType,
      cast<ElementsAttr>(makeAttr(inductionVarType, 0)));

  auto batchOp = enzyme::BatchOp::create(
      rewriter, whileOp->getLoc(),
      RankedTensorType::get(outputShape, resultType.getElementType()),
      mlir::FlatSymbolRefAttr::get(func.getContext(), func.getName()),
      ValueRange(newOperands),
      rewriter.getDenseI64ArrayAttr({info.getConstantNumIters()}));

  rewriter.setInsertionPointAfter(op);
  SmallVector<Value> dynamicSliceStarts(outputShape.size(), constZero);
  Value resIndex = stablehlo::SubtractOp::create(rewriter, whileOp->getLoc(),
                                                 info.getInductionVariable(),
                                                 info.getStart());
  if (!info.isStepOne()) {
    resIndex = stablehlo::DivOp::create(rewriter, whileOp->getLoc(), resIndex,
                                        info.getStep(rewriter));
  }
  dynamicSliceStarts[0] = resIndex;

  SmallVector<int64_t> dynamicSliceSizes(outputShape.begin(),
                                         outputShape.end());
  dynamicSliceSizes[0] = 1;

  auto dynamicSlice = stablehlo::DynamicSliceOp::create(
      rewriter, whileOp->getLoc(), batchOp->getResult(0), dynamicSliceStarts,
      dynamicSliceSizes);
  rewriter.replaceOpWithNewOp<stablehlo::ReshapeOp>(
      op, op->getResult(0).getType(), dynamicSlice);

  enzyme::batchutils::batchOperationInline(
      rewriter, batchOp, cast<FunctionOpInterface>(func.getOperation()));

  return true;
}

GreedyWhileLoopBatchFission::ValidBatchingInfo
GreedyWhileLoopBatchFission::isDynamicSliceValidForBatching(
    stablehlo::DynamicSliceOp sliceOp,
    llvm::MapVector<Value, mlir::enzyme::WhileLoopInfo::AffineIndexInfo>
        &affineIndexInfoMap,
    Block &whileBody, stablehlo::WhileOp whileOp) const {
  auto operand = sliceOp.getOperand();

  if (operand.getParentBlock() == &whileBody) {
    auto failureRetVal = ValidBatchingInfo{
        IsValidForBatchingResult::OPERAND_NOT_ACCESSIBLE_FROM_PARENT, {}};
    if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
      auto terminator = whileBody.getTerminator();
      if (!terminator ||
          terminator->getOperand(blockArg.getArgNumber()) != operand) {
        return failureRetVal;
      }
    } else {
      return failureRetVal;
    }
  }

  SmallVector<int64_t> dimensions;
  auto sliceSizes = sliceOp.getSliceSizes();

  for (auto [i, startIndex] : llvm::enumerate(sliceOp.getStartIndices())) {
    if (affineIndexInfoMap.contains(startIndex) && sliceSizes[i] == 1) {
      dimensions.push_back(i);
      continue;
    }

    if (definedOutside(startIndex, whileOp))
      continue;

    return ValidBatchingInfo{IsValidForBatchingResult::DYNAMIC_START_INDEX, {}};
  }

  if (dimensions.empty())
    return ValidBatchingInfo{
        IsValidForBatchingResult::NO_INDUCTION_VARIABLE_DETECTED, {}};

  // We should have exactly one index from the body, and it should be
  // a descendant of the induction variable
  return ValidBatchingInfo{IsValidForBatchingResult::VALID, dimensions};
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
                   ConcatInsertDimToBatch<stablehlo::ConcatenateOp>,
                   ConcatInsertDimToBatch<stablehlo::GetDimensionSizeOp>,
                   ConcatInsertDimToBatch<stablehlo::ReverseOp>,
                   ConcatInsertDimToBatch<stablehlo::ReduceWindowOp>,
                   ConcatInsertDimToBatch<stablehlo::ConvolutionOp>,
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
          SliceToBatch<stablehlo::ConcatenateOp>,
          SliceToBatch<stablehlo::GetDimensionSizeOp>,
          SliceToBatch<stablehlo::ReverseOp>,
          SliceToBatch<stablehlo::ReduceWindowOp>,
          SliceToBatch<stablehlo::ConvolutionOp>, SliceToBatchElementwise>(
          context);
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
